"""MARL MAPPO policy wrapper — MAPS §5 (MeltingPot MAPPO).

Ported from ``onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py`` (``R_MAPPOPolicy``),
the **GRU non-attention** path. Wraps the ``actor`` and ``critic`` (+ their optimizers).

**Fix (a) — the judge acts (2026-07, run/marl-20seeds-1M).** The paper's separate
``actor_meta``/``critic_meta`` ghost networks are REMOVED. The confidence judge
(``SecondOrderNetwork``) now lives inside ``actor`` and reads the acting network's own
representation, so ``actor_optimizer`` covers it and the wager loss co-trains the acting
policy (Blindsight/AGL pattern). See ``docs/reviews/fix-a-meta-cotraining-marl.md``.
Monitoring only: ``get_actions`` is unchanged (the wager never gates env actions).

- **rnn_cells (LSTM) dropped**: GRU path only (D16); 5-tuples where the source used 7-tuples.

The argparse ``args`` of the source is replaced by explicit keyword parameters (the
rebuild's convention); the optimizer dispatch reproduces the source's exact
per-optimizer construction.

References
----------
Vargas et al. (2025), MAPS, TMLR submission §5, Table 7.
Yu et al. (2022). The surprising effectiveness of PPO in cooperative MARL (MAPPO).
"""

from __future__ import annotations

import torch
import torch_optimizer as optim2

from maps.domains.marl.policy import R_Actor, R_Critic
from maps.domains.marl.util import get_shape_from_obs_space, update_linear_schedule


def _build_optimizer(name: str, params, lr: float, opti_eps: float, weight_decay: float):
    """Reproduce rMAPPOPolicy.py's per-optimizer construction (same args as source)."""
    if name == "ADAM":
        return torch.optim.Adam(
            params, lr=lr, eps=opti_eps, weight_decay=weight_decay, amsgrad=False
        )
    if name == "ADAMAX":
        return torch.optim.Adamax(params, lr=lr, eps=opti_eps, weight_decay=weight_decay)
    if name == "RANGERVA":
        return optim2.RangerVA(params, lr=lr, eps=opti_eps, weight_decay=weight_decay)
    if name == "AMS":
        return torch.optim.Adam(
            params, lr=lr, eps=opti_eps, weight_decay=weight_decay, amsgrad=True
        )
    if name == "ADAMW":
        return torch.optim.AdamW(
            params, lr=lr, eps=opti_eps, weight_decay=weight_decay, amsgrad=False
        )
    if name == "AMSW":
        return torch.optim.AdamW(
            params, lr=lr, eps=opti_eps, weight_decay=weight_decay, amsgrad=True
        )
    if name == "RMS":
        return torch.optim.RMSprop(params, lr=lr, eps=opti_eps, weight_decay=weight_decay)
    if name == "POO":
        return optim2.Shampoo(params, lr=lr, epsilon=opti_eps, weight_decay=weight_decay)
    if name == "SWT":
        return optim2.SWATS(params, lr=lr, eps=opti_eps, weight_decay=weight_decay)
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


class R_MAPPOPolicy:  # noqa: N801 (source class name)
    """Wraps actor/critic + meta actor/critic and their optimizers. Port of R_MAPPOPolicy."""

    def __init__(
        self,
        *,
        obs_space,
        cent_obs_space,
        act_space,
        hidden_size: int,
        recurrent_n: int,
        cascade_iterations_1: int,
        cascade_iterations_2: int,
        lr: float,
        critic_lr: float,
        opti_eps: float,
        weight_decay: float,
        optimizer: str = "ADAM",
        use_orthogonal: bool = False,
        gain: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.lr = lr
        self.critic_lr = critic_lr
        self.opti_eps = opti_eps
        self.weight_decay = weight_decay
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.optimizer = optimizer

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        self.actor = R_Actor(
            obs_shape,
            act_space,
            hidden_size=hidden_size,
            recurrent_n=recurrent_n,
            cascade_iterations_1=cascade_iterations_1,
            cascade_iterations_2=cascade_iterations_2,
            use_orthogonal=use_orthogonal,
            gain=gain,
            device=device,
        )
        self.critic = R_Critic(
            share_obs_shape,
            hidden_size=hidden_size,
            recurrent_n=recurrent_n,
            cascade_iterations_1=cascade_iterations_1,
            use_orthogonal=use_orthogonal,
            device=device,
        )
        # Fix (a): no ghost meta nets. The judge lives INSIDE self.actor (self.actor.second_order),
        # so actor_optimizer covers it → the wager loss co-trains the acting network.
        self.actor_optimizer = _build_optimizer(
            optimizer, self.actor.parameters(), lr, opti_eps, weight_decay
        )
        self.critic_optimizer = _build_optimizer(
            optimizer, self.critic.parameters(), critic_lr, opti_eps, weight_decay
        )

    def lr_decay(self, episode: int, episodes: int) -> None:
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(
        self,
        cent_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=False,
    ):
        """Rollout action selection — baseline ``actor`` only (M-C2). GRU 5-tuple return."""
        cent_obs = torch.tensor(cent_obs).to(self.device)
        obs = torch.tensor(obs).to(self.device)
        rnn_states_actor = torch.tensor(rnn_states_actor).to(self.device)
        rnn_states_critic = torch.tensor(rnn_states_critic).to(self.device)
        masks = torch.tensor(masks).to(self.device)
        if available_actions is not None:
            available_actions = torch.tensor(available_actions).to(self.device)

        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(
        self,
        cent_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        """Baseline actor log-probs/entropy + co-training wager + critic values.

        Fix (a): the actor also returns the wager (from its own representation); the trainer
        adds its BCE (vs advantage>0) to the actor loss so it co-trains the acting network."""
        action_log_probs, dist_entropy, wager = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy, wager
