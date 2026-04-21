"""Student ``R_Actor`` and ``R_Critic`` — non-attention, non-MLP path only.

Stripped from ``external/paper_reference/marl_tmlr/onpolicy/algorithms/r_mappo/
algorithm/r_actor_critic.py``. Keeps the default forward path that our
:class:`maps.experiments.marl.policy.MAPPOActor` / :class:`MAPPOCritic` target.

Stripped branches :
- ``use_attention`` RIM / SCOFF.
- ``MLPBase`` (1D obs).
- ``PopArt`` (``use_popart=False``).
- HATRPO / TRPO branches in ``evaluate_actions``.
- ``rnn_cells`` — unused, but preserved in the signature for shape parity.

Ported verbatim otherwise.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .act import ACTLayer
from .cnn import CNNBase
from .rnn import RNNLayer
from .util import check, init


def _get_shape_from_obs_space(obs_space):
    """Student ``onpolicy.utils.util.get_shape_from_obs_space`` inlined."""
    if hasattr(obs_space, "shape"):
        return tuple(obs_space.shape)
    return tuple(obs_space)


class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._recurrent_N = args.recurrent_N
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.cascade_one = args.cascade_iterations1
        self.cascade_rate_one = float(1 / self.cascade_one)

        obs_shape = _get_shape_from_obs_space(obs_space)
        self.base = CNNBase(obs_shape, self.hidden_size, self._use_orthogonal, args.use_ReLU)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal
            )

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)
        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        output_cascade1 = None
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        for _ in range(self.cascade_one):
            actor_features, rnn_states, output_cascade1 = self.rnn(
                actor_features, rnn_states, masks, output_cascade1, self.cascade_rate_one
            )

        # Student applies ``rnn_states = rnn_states.permute(1, 0, 2)`` on exit ;
        # we preserve it to match the tensor layout our port is compared against.
        rnn_states = rnn_states.permute(1, 0, 2)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self,
        obs,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        """Student L125-196 — returns (action_log_probs, dist_entropy)."""
        output_cascade1 = None
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)
        for _ in range(self.cascade_one):
            actor_features, rnn_states, output_cascade1 = self.rnn(
                actor_features, rnn_states, masks, output_cascade1, self.cascade_rate_one
            )

        # Student L199-204 — Discrete ACT eval.
        action_logits = self.act.action_out(actor_features, available_actions)
        action_log_probs = action_logits.log_probs(action)
        if active_masks is not None:
            dist_entropy = (
                action_logits.entropy() * active_masks.squeeze(-1)
            ).sum() / active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()
        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._recurrent_N = args.recurrent_N
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.cascade_one = args.cascade_iterations1
        self.cascade_rate_one = float(1 / self.cascade_one)

        cent_obs_shape = _get_shape_from_obs_space(cent_obs_space)
        self.base = CNNBase(cent_obs_shape, self.hidden_size, self._use_orthogonal, args.use_ReLU)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal
            )

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=1.0)

        self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        output_cascade1 = None
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)

        for _ in range(self.cascade_one):
            critic_features, rnn_states, output_cascade1 = self.rnn(
                critic_features, rnn_states, masks, output_cascade1, self.cascade_rate_one
            )

        rnn_states = rnn_states.permute(1, 0, 2)
        values = self.v_out(critic_features)
        return values, rnn_states
