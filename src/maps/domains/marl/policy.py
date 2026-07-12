"""MARL actor & critic (baseline) — MAPS §5 (MeltingPot MAPPO).

Ported from ``onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py`` (R_Actor,
R_Critic), the **GRU non-attention** path (config: use_recurrent_policy=true, no
attention → RIM/SCOFF out of scope). ``rnn_cells`` (LSTM) dropped.

Both nets: CNN encoder → cascade loop over the GRU (``cascade_iterations_1`` steps,
threading ``output_cascade1``) → action head (actor) / value head (critic). The
actor is what selects env actions (M-C2: the meta nets do NOT act — see the meta
variants, added separately).

References
----------
Vargas et al. (2025), MAPS, TMLR submission §5, Table 7.
"""

from __future__ import annotations

import torch
from torch import nn

from maps.domains.marl.act import ACTLayer
from maps.domains.marl.encoder import CNNBase
from maps.domains.marl.rnn import RNNLayer
from maps.domains.marl.util import check, init


class R_Actor(nn.Module):  # noqa: N801 (source class name)
    """Baseline actor: CNN → cascade-GRU → ACTLayer. Port of R_Actor (GRU path)."""

    def __init__(
        self,
        obs_shape,
        action_space,
        *,
        hidden_size: int,
        recurrent_n: int,
        cascade_iterations_1: int,
        use_orthogonal: bool = True,
        gain: float = 0.01,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cascade_one = cascade_iterations_1
        self.cascade_rate_one = float(1.0 / cascade_iterations_1)
        self.device = torch.device(device)
        self.base = CNNBase(obs_shape, hidden_size, use_orthogonal, use_relu=True)
        self.rnn = RNNLayer(hidden_size, hidden_size, recurrent_n, use_orthogonal)
        self.act = ACTLayer(action_space, hidden_size, use_orthogonal, gain)
        self.to(self.device)

    def _cascade(self, features, rnn_states, masks):
        output_cascade1 = None
        for _ in range(self.cascade_one):
            features, rnn_states, output_cascade1 = self.rnn(
                features, rnn_states, masks, output_cascade1, self.cascade_rate_one
            )
        return features, rnn_states

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(self.device).float()
        rnn_states = check(rnn_states).to(self.device).float()
        masks = check(masks).to(self.device).float()
        if available_actions is not None:
            available_actions = check(available_actions).to(self.device).float()
        features, rnn_states = self._cascade(self.base(obs), rnn_states, masks)
        rnn_states = rnn_states.permute(1, 0, 2)
        actions, action_log_probs = self.act(features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self, obs, rnn_states, action, masks, available_actions=None, active_masks=None
    ):
        obs = check(obs).to(self.device).float()
        rnn_states = check(rnn_states).to(self.device).float()
        action = check(action).to(self.device).float()
        masks = check(masks).to(self.device).float()
        if available_actions is not None:
            available_actions = check(available_actions).to(self.device).float()
        if active_masks is not None:
            active_masks = check(active_masks).to(self.device).float()
        features, _ = self._cascade(self.base(obs), rnn_states, masks)
        return self.act.evaluate_actions(features, action, available_actions, active_masks)


class R_Critic(nn.Module):  # noqa: N801 (source class name)
    """Baseline critic: CNN → cascade-GRU → value head. Port of R_Critic (GRU path)."""

    def __init__(
        self,
        share_obs_shape,
        *,
        hidden_size: int,
        recurrent_n: int,
        cascade_iterations_1: int,
        use_orthogonal: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cascade_one = cascade_iterations_1
        self.cascade_rate_one = float(1.0 / cascade_iterations_1)
        self.device = torch.device(device)
        self.base = CNNBase(share_obs_shape, hidden_size, use_orthogonal, use_relu=True)
        self.rnn = RNNLayer(hidden_size, hidden_size, recurrent_n, use_orthogonal)

        def init_(m):
            return init(
                m,
                [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal],
                lambda x: nn.init.constant_(x, 0),
            )

        self.v_out = init_(nn.Linear(hidden_size, 1))
        self.to(self.device)

    def forward(self, cent_obs, rnn_states, masks):
        cent_obs = check(cent_obs).to(self.device).float()
        rnn_states = check(rnn_states).to(self.device).float()
        masks = check(masks).to(self.device).float()
        features = self.base(cent_obs)
        output_cascade1 = None
        for _ in range(self.cascade_one):
            features, rnn_states, output_cascade1 = self.rnn(
                features, rnn_states, masks, output_cascade1, self.cascade_rate_one
            )
        rnn_states = rnn_states.permute(1, 0, 2)
        values = self.v_out(features)
        return values, rnn_states
