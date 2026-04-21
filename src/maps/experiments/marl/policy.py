"""MARL policy networks — paper Fig.4 actor + critic + MAPS additions.

Module layout (per ``docs/reviews/marl-maps-additions.md §(i)``) :

- :class:`MarlSecondOrderNetwork` — paper §2.2 eq.1-3 + MARL's extra
  ``comparison_layer`` (Linear + ReLU + Dropout(0.1)) + 2-unit wager head.
- :class:`MAPPOActor` — baseline actor (settings meta=False). CNN → RNN → ACT.
- :class:`MAPPOCritic` — baseline critic (settings meta=False). CNN → RNN → v_out.
- :class:`MAPSActor` — meta actor (settings meta=True). Same as MAPPOActor
  plus :class:`MarlSecondOrderNetwork` on the wager path. Includes the student
  ``layer_input`` extra Linear projection.
- :class:`MAPSCritic` — meta critic (settings meta=True). Uses ``RNNLayerMeta``
  but no 2nd-order (paper Fig.4 puts wager only on actor side).
- :class:`MAPPOPolicy` — wrapper managing the 4 networks + 2 pairs of
  optimizers. Only Adam and RangerVA supported (E.5 scope lock).

Port references :
- Student ``r_actor_critic.py``, ``r_actor_critic_meta.py``, ``rMAPPOPolicy.py``.
- Paper §2.2 eq.1-6 + Table 12 + Fig.4.
- Docs: ``docs/reviews/marl-architecture.md``, ``docs/reviews/marl-maps-additions.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from maps.experiments.marl.act import ACTLayer
from maps.experiments.marl.encoder import CNNBase
from maps.experiments.marl.rnn import RNNLayer, RNNLayerMeta
from maps.experiments.marl.util import init

__all__ = [
    "MarlSecondOrderNetwork",
    "MAPPOActor",
    "MAPPOCritic",
    "MAPSActor",
    "MAPSCritic",
    "MAPPOPolicy",
]

log = logging.getLogger(__name__)

# RangerVA is optional (see E.6 — lives in .venv-marl only).
try:
    import torch_optimizer as _torch_optim

    _RANGERVA_AVAILABLE = True
except ImportError:
    _RANGERVA_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# MarlSecondOrderNetwork
# ─────────────────────────────────────────────────────────────────────────────


class MarlSecondOrderNetwork(nn.Module):
    """MARL-variant 2nd-order network (student ``r_actor_critic_meta.py:17-42``).

    Forward (paper §2.2 eq.2-3 + MARL ``comparison_layer``) :

    ``comparison_out = Dropout(ReLU(comparison_layer(C)))``
    ``if prev_comparison is not None:``
    ``    comparison_out = α · comparison_out + (1-α) · prev_comparison``  (eq.6)
    ``wager = wager_layer(comparison_out)``                                (eq.3)

    Weight init matches student L29-32 : ``comparison_layer`` uniform(-1, 1),
    ``wager`` uniform(0, 0.1).

    Outputs **raw 2-unit logits** (paper eq.3). Downstream loss applies
    ``binary_cross_entropy_with_logits`` (eq.5).
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.comparison_layer = nn.Linear(hidden_size, hidden_size)
        self.wager = nn.Linear(hidden_size, 2)  # paper eq.3 : 2 raw logits
        self.dropout = nn.Dropout(p=dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.comparison_layer.weight, -1.0, 1.0)
        nn.init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(
        self,
        comparison_matrix: torch.Tensor,
        prev_comparison: torch.Tensor | None,
        cascade_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(wager_logits, comparison_out)``.

        Note : the student's ``comparison_matrix`` here is NOT ``obs - recon``
        (paper eq.1) but rather ``h_pre_rnn - h_post_rnn`` — see
        ``docs/reviews/marl-maps-additions.md §(c)``. Semantics handled by the
        caller (``MAPSActor.evaluate_actions``).
        """
        comparison_out = self.dropout(F.relu(self.comparison_layer(comparison_matrix)))
        if prev_comparison is not None:
            comparison_out = cascade_rate * comparison_out + (1.0 - cascade_rate) * prev_comparison
        wager_logits = self.wager(comparison_out)
        return wager_logits, comparison_out


# ─────────────────────────────────────────────────────────────────────────────
# Baseline actor / critic (setting.meta=False)
# ─────────────────────────────────────────────────────────────────────────────


class MAPPOActor(nn.Module):
    """Baseline MAPPO actor — student ``R_Actor`` L15-179 with attention paths
    removed (per E.5 scope : no RIM / SCOFF — plain GRU only).

    Pipeline : ``obs → CNN → [cascade loop] × N × RNN → ACT``.
    """

    def __init__(
        self,
        cfg: DictConfig,
        obs_shape: tuple[int, int, int],
        action_space,
        cascade_iterations: int = 1,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.hidden_size = int(cfg.model.hidden_size)
        self._cascade_iterations = max(1, int(cascade_iterations))
        self._cascade_rate = 1.0 / float(self._cascade_iterations)

        self.base = CNNBase(
            obs_shape=obs_shape,
            hidden_size=self.hidden_size,
            use_orthogonal=bool(cfg.model.use_orthogonal),
            use_ReLU=bool(cfg.model.use_ReLU),
        )
        self.rnn = RNNLayer(
            inputs_dim=self.hidden_size,
            outputs_dim=self.hidden_size,
            recurrent_n=int(cfg.model.recurrent_n),
            use_orthogonal=bool(cfg.model.use_orthogonal),
        )
        self.act = ACTLayer(
            action_space=action_space,
            inputs_dim=self.hidden_size,
            use_orthogonal=bool(cfg.model.use_orthogonal),
            gain=float(cfg.model.gain),
        )
        self.to(self.device)

    def forward(
        self,
        obs: torch.Tensor,
        rnn_states: torch.Tensor,
        masks: torch.Tensor,
        available_actions: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(actions, action_log_probs, rnn_states)``."""
        actor_features = self.base(obs)

        cascade_state: torch.Tensor | None = None
        for _ in range(self._cascade_iterations):
            actor_features, rnn_states, cascade_state = self.rnn(
                actor_features, rnn_states, masks, cascade_state, self._cascade_rate
            )

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        rnn_states: torch.Tensor,
        action: torch.Tensor,
        masks: torch.Tensor,
        available_actions: torch.Tensor | None = None,
        active_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(action_log_probs, dist_entropy)``."""
        actor_features = self.base(obs)

        cascade_state: torch.Tensor | None = None
        for _ in range(self._cascade_iterations):
            actor_features, rnn_states, cascade_state = self.rnn(
                actor_features, rnn_states, masks, cascade_state, self._cascade_rate
            )

        return self.act.evaluate_actions(actor_features, action, available_actions, active_masks)


class MAPPOCritic(nn.Module):
    """Baseline MAPPO critic — student ``R_Critic`` L183-294 with attention paths
    removed. Centralized-obs encoder + GRU + linear value head."""

    def __init__(
        self,
        cfg: DictConfig,
        cent_obs_shape: tuple[int, int, int],
        cascade_iterations: int = 1,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.hidden_size = int(cfg.model.hidden_size)
        self._cascade_iterations = max(1, int(cascade_iterations))
        self._cascade_rate = 1.0 / float(self._cascade_iterations)

        self.base = CNNBase(
            obs_shape=cent_obs_shape,
            hidden_size=self.hidden_size,
            use_orthogonal=bool(cfg.model.use_orthogonal),
            use_ReLU=bool(cfg.model.use_ReLU),
        )
        self.rnn = RNNLayer(
            inputs_dim=self.hidden_size,
            outputs_dim=self.hidden_size,
            recurrent_n=int(cfg.model.recurrent_n),
            use_orthogonal=bool(cfg.model.use_orthogonal),
        )

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][int(bool(cfg.model.use_orthogonal))]

        def init_(m: nn.Module) -> nn.Module:
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=1.0)

        self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(self.device)

    def forward(
        self,
        cent_obs: torch.Tensor,
        rnn_states: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(values, rnn_states)``."""
        critic_features = self.base(cent_obs)

        cascade_state: torch.Tensor | None = None
        for _ in range(self._cascade_iterations):
            critic_features, rnn_states, cascade_state = self.rnn(
                critic_features, rnn_states, masks, cascade_state, self._cascade_rate
            )

        values = self.v_out(critic_features)
        return values, rnn_states


# ─────────────────────────────────────────────────────────────────────────────
# MAPS (meta) actor / critic (setting.meta=True)
# ─────────────────────────────────────────────────────────────────────────────


class MAPSActor(nn.Module):
    """MAPS actor — adds :class:`MarlSecondOrderNetwork` + student's
    ``layer_input`` extra Linear projection (see ``docs/reviews/marl-maps-additions.md §(e)``).

    Pipeline (student ``r_actor_critic_meta.py:104-149`` for rollout,
    L151-196 for evaluate_actions / wager) :

    Rollout :
        obs → CNN → layer_input → RNNLayerMeta × cascade1 → ACT

    evaluate_actions (train-time wager) :
        obs → CNN → layer_input → snapshot (pre-RNN) → RNNLayerMeta × cascade1 →
             comparator = snapshot − post-RNN → MarlSecondOrderNetwork × cascade2 → wager

    ``layer_output`` from student (L67) is DROPPED (dead code per E.4).
    """

    def __init__(
        self,
        cfg: DictConfig,
        obs_shape: tuple[int, int, int],
        action_space,
        cascade_iterations1: int = 1,
        cascade_iterations2: int = 1,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.hidden_size = int(cfg.model.hidden_size)
        self._cascade1 = max(1, int(cascade_iterations1))
        self._cascade2 = max(1, int(cascade_iterations2))
        self._cascade_rate1 = 1.0 / float(self._cascade1)
        self._cascade_rate2 = 1.0 / float(self._cascade2)

        self.base = CNNBase(
            obs_shape=obs_shape,
            hidden_size=self.hidden_size,
            use_orthogonal=bool(cfg.model.use_orthogonal),
            use_ReLU=bool(cfg.model.use_ReLU),
        )
        # Student's ``layer_input`` — extra Linear(H, H) pre-RNN. Kept for
        # student-parity per E.4 decision ; paper Fig.4 silent on this.
        self.layer_input = nn.Linear(self.hidden_size, self.hidden_size)

        self.rnn = RNNLayerMeta(
            inputs_dim=self.hidden_size,
            outputs_dim=self.hidden_size,
            recurrent_n=int(cfg.model.recurrent_n),
            use_orthogonal=bool(cfg.model.use_orthogonal),
        )
        self.act = ACTLayer(
            action_space=action_space,
            inputs_dim=self.hidden_size,
            use_orthogonal=bool(cfg.model.use_orthogonal),
            gain=float(cfg.model.gain),
        )
        self.second_order = MarlSecondOrderNetwork(
            hidden_size=self.hidden_size,
            dropout=float(cfg.model.second_order_dropout),
        )
        self.to(self.device)

    def forward(
        self,
        obs: torch.Tensor,
        rnn_states: torch.Tensor,
        masks: torch.Tensor,
        available_actions: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rollout path (student L104-149). Returns ``(actions, log_probs, rnn_states)``."""
        actor_features = self.base(obs)
        # Note: student also projects `rnn_states = self.layer_input(rnn_states)`
        # but ONLY in ``evaluate_actions`` (L181). We preserve that asymmetry.
        cascade_state: torch.Tensor | None = None
        for _ in range(self._cascade1):
            actor_features, rnn_states, cascade_state = self.rnn(
                actor_features, rnn_states, masks, cascade_state, self._cascade_rate1
            )
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        rnn_states: torch.Tensor,
        action: torch.Tensor,  # noqa: ARG002 — student signature ; unused here (returns wager only)
        masks: torch.Tensor,
        available_actions: torch.Tensor | None = None,  # noqa: ARG002
        active_masks: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Meta-path : compute the wager (student L151-196).

        Student's ``evaluate_actions`` on ``R_Actor_Meta`` returns ONLY the
        wager (not action_log_probs like the baseline). Named misleadingly —
        but that's the student contract and we preserve it (see
        ``docs/reviews/marl-maps-additions.md §(f)``).
        """
        actor_features = self.base(obs)
        actor_features = self.layer_input(actor_features)
        rnn_states = self.layer_input(rnn_states)

        initial_states = actor_features

        cascade_state: torch.Tensor | None = None
        actor_features_out = actor_features
        rnn_states_out = rnn_states
        for _ in range(self._cascade1):
            actor_features_out, rnn_states_out, cascade_state = self.rnn(
                actor_features_out, rnn_states_out, masks, cascade_state, self._cascade_rate1
            )

        # eq.1 MARL variant : comparator = pre-RNN − post-RNN
        comparison_matrix = initial_states - actor_features_out

        prev_comparison: torch.Tensor | None = None
        wager = None
        for _ in range(self._cascade2):
            wager, prev_comparison = self.second_order(
                comparison_matrix, prev_comparison, self._cascade_rate2
            )

        assert wager is not None
        return wager


class MAPSCritic(nn.Module):
    """MAPS critic — student ``R_Critic_Meta`` L199-299.

    Uses :class:`RNNLayerMeta` + ``layer_input`` (same as :class:`MAPSActor`)
    but **no 2nd-order** (paper Fig.4 wagers only on actor side).
    """

    def __init__(
        self,
        cfg: DictConfig,
        cent_obs_shape: tuple[int, int, int],
        cascade_iterations1: int = 1,
        cascade_iterations2: int = 1,  # noqa: ARG002 — unused on critic (kept for API symmetry)
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.hidden_size = int(cfg.model.hidden_size)
        self._cascade1 = max(1, int(cascade_iterations1))
        self._cascade_rate1 = 1.0 / float(self._cascade1)

        self.base = CNNBase(
            obs_shape=cent_obs_shape,
            hidden_size=self.hidden_size,
            use_orthogonal=bool(cfg.model.use_orthogonal),
            use_ReLU=bool(cfg.model.use_ReLU),
        )
        self.layer_input = nn.Linear(self.hidden_size, self.hidden_size)
        self.rnn = RNNLayerMeta(
            inputs_dim=self.hidden_size,
            outputs_dim=self.hidden_size,
            recurrent_n=int(cfg.model.recurrent_n),
            use_orthogonal=bool(cfg.model.use_orthogonal),
        )

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][int(bool(cfg.model.use_orthogonal))]

        def init_(m: nn.Module) -> nn.Module:
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=1.0)

        self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(self.device)

    def forward(
        self,
        cent_obs: torch.Tensor,
        rnn_states: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(values, rnn_states)``."""
        critic_features = self.base(cent_obs)
        # Student L285-286 commented out layer_input on forward path. We
        # leave it out for forward-parity. Keep the Linear available for
        # future experiments.

        cascade_state: torch.Tensor | None = None
        for _ in range(self._cascade1):
            critic_features, rnn_states, cascade_state = self.rnn(
                critic_features, rnn_states, masks, cascade_state, self._cascade_rate1
            )

        values = self.v_out(critic_features)
        return values, rnn_states


# ─────────────────────────────────────────────────────────────────────────────
# Policy wrapper (4 networks + optimizers)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _PolicyOptimizers:
    actor: torch.optim.Optimizer
    critic: torch.optim.Optimizer
    actor_meta: torch.optim.Optimizer | None
    critic_meta: torch.optim.Optimizer | None


class MAPPOPolicy:
    """Wraps the 4 networks + their optimizers.

    Only Adam and RangerVA supported (per E.5 scope). Optimizer type is
    determined by ``cfg.optimizer.name``. RangerVA requires the optional
    ``torch_optimizer`` dep (available in ``.venv-marl``).

    The ``actor_meta`` / ``critic_meta`` pair is only built when at least one
    setting uses ``meta=True``. For the baseline-only setting, the meta
    networks + their optimizers are ``None`` to save GPU memory.
    """

    def __init__(
        self,
        cfg: DictConfig,
        obs_shape: tuple[int, int, int],
        cent_obs_shape: tuple[int, int, int],
        action_space,
        *,
        meta: bool = False,
        cascade_iterations1: int = 1,
        cascade_iterations2: int = 1,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.meta_enabled = bool(meta)

        # Baseline networks — always built (used for settings 1/2).
        self.actor = MAPPOActor(
            cfg, obs_shape, action_space, cascade_iterations=cascade_iterations1, device=device
        )
        self.critic = MAPPOCritic(
            cfg, cent_obs_shape, cascade_iterations=cascade_iterations1, device=device
        )

        # Meta networks — only when meta is enabled.
        self.actor_meta: MAPSActor | None = None
        self.critic_meta: MAPSCritic | None = None
        if self.meta_enabled:
            self.actor_meta = MAPSActor(
                cfg,
                obs_shape,
                action_space,
                cascade_iterations1=cascade_iterations1,
                cascade_iterations2=cascade_iterations2,
                device=device,
            )
            self.critic_meta = MAPSCritic(
                cfg,
                cent_obs_shape,
                cascade_iterations1=cascade_iterations1,
                cascade_iterations2=cascade_iterations2,
                device=device,
            )

        self.optimizers = self._build_optimizers(cfg)

    def _build_optimizers(self, cfg: DictConfig) -> _PolicyOptimizers:
        name = str(cfg.optimizer.name).upper()
        actor_lr = float(cfg.optimizer.actor_lr)
        critic_lr = float(cfg.optimizer.critic_lr)
        eps = float(cfg.optimizer.opti_eps)
        wd = float(cfg.optimizer.weight_decay)
        amsgrad = bool(cfg.optimizer.get("amsgrad", False))

        if name == "ADAM":
            kwargs_actor = dict(lr=actor_lr, eps=eps, weight_decay=wd, amsgrad=amsgrad)
            kwargs_critic = dict(lr=critic_lr, eps=eps, weight_decay=wd, amsgrad=amsgrad)
            actor_opt = torch.optim.Adam(self.actor.parameters(), **kwargs_actor)
            critic_opt = torch.optim.Adam(self.critic.parameters(), **kwargs_critic)
            actor_meta_opt = (
                torch.optim.Adam(self.actor_meta.parameters(), **kwargs_actor)
                if self.actor_meta is not None
                else None
            )
            critic_meta_opt = (
                torch.optim.Adam(self.critic_meta.parameters(), **kwargs_critic)
                if self.critic_meta is not None
                else None
            )
        elif name == "RANGERVA":
            if not _RANGERVA_AVAILABLE:
                log.warning(
                    "RangerVA requested but torch-optimizer not installed; falling back to Adam. "
                    "Install via `uv pip install --python .venv-marl/bin/python torch-optimizer`."
                )
                return self._build_optimizers_fallback_adam(cfg)
            RangerVA = _torch_optim.RangerVA
            kwargs_actor = dict(lr=actor_lr, eps=eps, weight_decay=wd)
            kwargs_critic = dict(lr=critic_lr, eps=eps, weight_decay=wd)
            actor_opt = RangerVA(self.actor.parameters(), **kwargs_actor)
            critic_opt = RangerVA(self.critic.parameters(), **kwargs_critic)
            actor_meta_opt = (
                RangerVA(self.actor_meta.parameters(), **kwargs_actor)
                if self.actor_meta is not None
                else None
            )
            critic_meta_opt = (
                RangerVA(self.critic_meta.parameters(), **kwargs_critic)
                if self.critic_meta is not None
                else None
            )
        else:
            raise ValueError(
                f"Unsupported optimizer {name!r}. Port supports only 'Adam' or 'RangerVA' (E.5 scope)."
            )

        return _PolicyOptimizers(
            actor=actor_opt, critic=critic_opt, actor_meta=actor_meta_opt, critic_meta=critic_meta_opt
        )

    def _build_optimizers_fallback_adam(self, cfg: DictConfig) -> _PolicyOptimizers:
        """Fallback to Adam when RangerVA was requested but is unavailable."""
        cfg_fallback = cfg.copy()
        cfg_fallback.optimizer.name = "Adam"
        return self._build_optimizers(cfg_fallback)

    # ---- Convenience wrappers (match student's R_MAPPOPolicy API surface) ----

    def total_params(self) -> dict[str, int]:
        """Returns parameter counts for logging (student L45-47)."""
        def count(m: nn.Module | None) -> int:
            if m is None:
                return 0
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "actor": count(self.actor),
            "critic": count(self.critic),
            "actor_meta": count(self.actor_meta),
            "critic_meta": count(self.critic_meta),
        }
