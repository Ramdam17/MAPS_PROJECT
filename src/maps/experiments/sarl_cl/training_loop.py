"""SARL+CL training orchestrator — continual-learning training loop.

Ports the CL half of ``SARL_CL/examples_cl/maps.py`` (the core of
``dqn()`` — lines 1028-2050). Structurally mirrors
:mod:`maps.experiments.sarl.training_loop` but adds:

1. Optional teacher networks (``teacher_first_net`` / ``teacher_second_net``)
   frozen for the whole run; drive the distillation + feature losses inside
   :func:`sarl_cl_update_step`.
2. Optional :class:`AdaptiveQNetwork` backbone for cross-game transfer
   (handles variable ``in_channels`` via zero-padding).
3. ``load_partial_state_dict`` helper for loading checkpoints whose
   parameter shapes don't match exactly (e.g. loading a Breakout-trained
   policy into an Adaptive variant with more channels). Paper lines 1261-1278.
4. Per-network :class:`DynamicLossWeighter` instances maintained across
   the whole curriculum.

What we deliberately omit vs the paper
--------------------------------------
The paper's dqn() includes per-game checkpoint bookkeeping (per-game
optimizer state, per-game replay buffer) that is mostly a data-munging
concern for multi-stage curriculum scripts. We keep the NUMERICAL core
(teachers, losses, weights) but save a simpler single-checkpoint schema.
Sprint 06 can extend to the full multi-game schema if needed for
reproducing the exact paper curriculum runs.

Parity boundary
---------------
Inner update (``sarl_cl_update_step``) is structurally faithful to the
paper's ``train()`` function. Outer loop mirrors the paper's ``dqn()``
at the level of: episode structure, target-net sync cadence, validation
cadence, LR decay cadence. Not bit-exact across RNG consumption paths
(see the SARL training_loop note on RNG divergence for details).

References
----------
- SARL_CL/examples_cl/maps.py:1028-2050 (source).
- Vargas et al. (2025), MAPS TMLR submission §4.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from maps.experiments.sarl.data import SarlReplayBuffer, Transition, get_state, target_wager
from maps.experiments.sarl.evaluate import ValidationSummary, aggregate_validation
from maps.experiments.sarl.rollout import epsilon_greedy_action
from maps.experiments.sarl_cl.loss_weighting import DynamicLossWeighter
from maps.experiments.sarl_cl.model import (
    AdaptiveQNetwork,
    SarlCLQNetwork,
    SarlCLSecondOrderNetwork,
)
from maps.experiments.sarl_cl.trainer import (
    LossMixingWeights,
    SarlCLUpdateOutput,
    sarl_cl_update_step,
)

log = logging.getLogger(__name__)

# ── Paper constants (source: SARL_CL/examples_cl/maps.py:92-107) ────────────
# These match the standard SARL defaults — CL does not change optimization
# hyperparameters, only the loss composition.

# Sprint-08 D.9+D.12 (2026-04-20): aligned paper-faithful, same as standard
# SARL. See sarl/training_loop.py + deviations.md for full rationale.
BATCH_SIZE = 128  # paper Table 11 (paper-faithful, unchanged)
REPLAY_BUFFER_SIZE = 100_000
REPLAY_START_SIZE = 5_000
TRAINING_FREQ = 1
TARGET_NETWORK_UPDATE_FREQ = 500  # paper CL uses 500 (not 1000); line 1121
MIN_SQUARED_GRAD = 0.01
STEP_SIZE_1 = 0.0003  # paper Table 11 (paper-faithful, unchanged)
STEP_SIZE_2 = 0.0002  # paper Table 11 (was 0.00005)
ADAM_BETAS: tuple[float, float] = (0.95, 0.95)  # paper Table 11 (was PyTorch default)
SCHEDULER_STEP = 0.999
SCHEDULER_PERIOD = 1  # paper Table 11 (was 1000); see _build_optimizers warning


class MinAtarLike(Protocol):
    """Minimal env interface (same as evaluate.MinAtarLike)."""

    def reset(self) -> Any: ...
    def state(self) -> np.ndarray: ...
    def act(self, action: torch.Tensor) -> tuple[float, bool]: ...
    def num_actions(self) -> int: ...
    def state_shape(self) -> tuple[int, int, int]: ...


# ── Configuration dataclass ─────────────────────────────────────────────────


@dataclass
class SarlCLTrainingConfig:
    """Hyperparameters for :func:`run_training_cl`.

    Paper settings 1-6 map to ``(meta, cascade_1, cascade_2)`` identically
    to the standard SARL config; see
    :data:`maps.experiments.sarl_cl.training_loop._SETTING_TABLE`.
    """

    # Environment.
    game: str = "space_invaders"
    seed: int = 42

    # Paper setting knobs.
    meta: bool = False
    cascade_iterations_1: int = 1
    cascade_iterations_2: int = 1

    # CL-specific toggles.
    curriculum: bool = False  # if True, expects a teacher checkpoint via teacher_load_path
    adaptive_backbone: bool = False  # use AdaptiveQNetwork (variable in_channels)
    max_input_channels: int = 7  # upper bound across curriculum games (Freeway=7)
    teacher_load_path: Path | None = None  # path to a previous-task checkpoint

    # Loss mixing (paper uses 1,1,1 — see trainer.LossMixingWeights).
    weight_task: float = 1.0
    weight_distillation: float = 1.0
    weight_feature: float = 1.0

    # DQN hyperparameters.
    # num_frames = 500_000 per paper Table 11 (D.12); override to 5_000_000
    # for legacy reproduction. See D-sarl-num-frames.
    num_frames: int = 500_000
    batch_size: int = BATCH_SIZE
    replay_buffer_size: int = REPLAY_BUFFER_SIZE
    replay_start_size: int = REPLAY_START_SIZE
    training_freq: int = TRAINING_FREQ
    target_update_freq: int = TARGET_NETWORK_UPDATE_FREQ
    step_size_1: float = STEP_SIZE_1
    step_size_2: float = STEP_SIZE_2
    scheduler_period: int = SCHEDULER_PERIOD
    scheduler_gamma: float = SCHEDULER_STEP
    adam_betas: tuple[float, float] = ADAM_BETAS
    # Paper Table 11 α=45 (→ 0.45). Aligned 2026-04-20 (D.9).
    alpha: float = 45.0
    # Paper Table 11 γ=0.999 (aligned 2026-04-20, D.7). See deviations.md
    # D-sarl-gamma. Override to 0.99 for student-baseline reproduction.
    gamma: float = 0.999

    # Validation cadence.
    validation_every_episodes: int = 50
    validation_iterations: int = 3

    # Runtime.
    device: str = "cpu"
    output_dir: Path | None = None


@dataclass
class CLTrainingMetrics:
    """Per-episode + per-validation metrics.

    CL-specific fields are the component-loss tracks (task / distillation /
    feature) collected from :class:`SarlCLUpdateOutput.components_first` and
    ``components_second``. They're ``None`` entries when the run has no
    teacher (first task in a curriculum).
    """

    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_frames: list[int] = field(default_factory=list)
    episode_losses_first: list[float | None] = field(default_factory=list)
    episode_losses_second: list[float | None] = field(default_factory=list)
    # CL component breakdowns (task / distillation / feature) per network.
    episode_components_first_task: list[float | None] = field(default_factory=list)
    episode_components_first_distillation: list[float | None] = field(default_factory=list)
    episode_components_first_feature: list[float | None] = field(default_factory=list)
    episode_components_second_task: list[float | None] = field(default_factory=list)
    episode_components_second_distillation: list[float | None] = field(default_factory=list)
    episode_components_second_feature: list[float | None] = field(default_factory=list)
    validation_frames: list[int] = field(default_factory=list)
    validation_summaries: list[ValidationSummary] = field(default_factory=list)
    total_updates: int = 0
    total_frames: int = 0
    wall_time_seconds: float = 0.0


# ── Helpers ─────────────────────────────────────────────────────────────────


def load_partial_state_dict(
    model: torch.nn.Module, state_dict: dict[str, torch.Tensor]
) -> torch.nn.Module:
    """Load parameters from ``state_dict`` into ``model`` only where shapes match.

    Port of SARL_CL/examples_cl/maps.py:1265-1278. Used when loading a
    previous-task checkpoint into a (potentially differently-shaped)
    new-task network. Parameters whose shape disagrees are left at the
    fresh init — a deliberate design choice in the paper for cross-game
    transfer with the Adaptive backbone.

    Operates on a deep copy of ``model`` so the caller's original is
    never mutated. Returns the (new) loaded model.
    """
    model = copy.deepcopy(model)
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name].copy_(param)
            log.debug("partial load: copied %s (shape %s)", name, tuple(param.shape))
        else:
            log.info(
                "partial load: skipped %s (shape mismatch: want %s, got %s)",
                name,
                tuple(model_state[name].shape) if name in model_state else "absent",
                tuple(param.shape),
            )
    model.load_state_dict(model_state)
    return model


def _build_first_order(
    in_channels: int,
    num_actions: int,
    cfg: SarlCLTrainingConfig,
) -> SarlCLQNetwork | AdaptiveQNetwork:
    """Build FO policy net — Adaptive variant if ``cfg.adaptive_backbone``."""
    if cfg.adaptive_backbone:
        return AdaptiveQNetwork(cfg.max_input_channels, num_actions).to(cfg.device)
    return SarlCLQNetwork(in_channels, num_actions).to(cfg.device)


def _build_networks(
    in_channels: int,
    num_actions: int,
    cfg: SarlCLTrainingConfig,
) -> tuple[
    SarlCLQNetwork | AdaptiveQNetwork,
    SarlCLQNetwork | AdaptiveQNetwork,
    SarlCLSecondOrderNetwork | None,
    SarlCLQNetwork | AdaptiveQNetwork | None,
    SarlCLSecondOrderNetwork | None,
]:
    """Policy, target, SO (optional), FO-teacher (optional), SO-teacher (optional)."""
    # Local re-seed for deterministic network init. Upstream callers MUST have
    # already called ``maps.utils.seeding.set_all_seeds(cfg.seed)`` to cover
    # Python random / NumPy / CUDA RNGs; this ``torch.manual_seed`` alone only
    # re-seeds the PyTorch CPU generator.
    torch.manual_seed(cfg.seed)
    policy = _build_first_order(in_channels, num_actions, cfg)
    target = _build_first_order(in_channels, num_actions, cfg)
    target.load_state_dict(policy.state_dict())
    target.eval()
    second = SarlCLSecondOrderNetwork(in_channels).to(cfg.device) if cfg.meta else None

    teacher_first: SarlCLQNetwork | AdaptiveQNetwork | None = None
    teacher_second: SarlCLSecondOrderNetwork | None = None
    if cfg.curriculum and cfg.teacher_load_path is not None:
        ckpt = torch.load(cfg.teacher_load_path, map_location=cfg.device, weights_only=False)
        teacher_first = _build_first_order(in_channels, num_actions, cfg)
        teacher_first = load_partial_state_dict(teacher_first, ckpt["policy_net_state_dict"])
        teacher_first.eval()
        for p in teacher_first.parameters():
            p.requires_grad_(False)

        if cfg.meta and "second_net_state_dict" in ckpt:
            teacher_second = SarlCLSecondOrderNetwork(in_channels).to(cfg.device)
            teacher_second = load_partial_state_dict(teacher_second, ckpt["second_net_state_dict"])
            teacher_second.eval()
            for p in teacher_second.parameters():
                p.requires_grad_(False)

    return policy, target, second, teacher_first, teacher_second


def _build_optimizers(
    policy: torch.nn.Module,
    second: torch.nn.Module | None,
    cfg: SarlCLTrainingConfig,
) -> tuple[optim.Optimizer, optim.Optimizer | None, Any, Any | None]:
    # Sprint-08 D.9: same paper-faithful-but-suspect step_size=1 warning as
    # standard SARL. See sarl/training_loop._build_optimizers for rationale.
    if cfg.scheduler_period == 1:
        log.warning(
            "scheduler_period=1 (paper Table 11 as-written): LR decays every "
            "update; with γ=%.3f, effective LR ≈ 0 after ~%d updates. May be "
            "a paper typo — override with `-o scheduler.step_size=1000` to "
            "reproduce student. See deviations.md D-sarl-sched-step.",
            cfg.scheduler_gamma,
            int(-5.0 / (cfg.scheduler_gamma - 1.0)) if cfg.scheduler_gamma < 1 else 0,
        )
    opt1 = optim.Adam(
        policy.parameters(),
        lr=cfg.step_size_1,
        eps=MIN_SQUARED_GRAD,
        betas=cfg.adam_betas,
    )
    sch1 = StepLR(opt1, step_size=cfg.scheduler_period, gamma=cfg.scheduler_gamma)
    opt2: optim.Optimizer | None = None
    sch2: Any | None = None
    if cfg.meta:
        assert second is not None
        opt2 = optim.Adam(
            second.parameters(),
            lr=cfg.step_size_2,
            eps=MIN_SQUARED_GRAD,
            betas=cfg.adam_betas,
        )
        sch2 = StepLR(opt2, step_size=cfg.scheduler_period, gamma=cfg.scheduler_gamma)
    return opt1, opt2, sch1, sch2


# ── Main entry point ────────────────────────────────────────────────────────


def run_training_cl(
    env: MinAtarLike,
    cfg: SarlCLTrainingConfig,
) -> tuple[
    SarlCLQNetwork | AdaptiveQNetwork,
    SarlCLSecondOrderNetwork | None,
    CLTrainingMetrics,
]:
    """Train a SARL+CL agent for ``cfg.num_frames`` frames.

    Returns the trained first-order net, optional second-order net, and
    a :class:`CLTrainingMetrics` snapshot. If a teacher checkpoint is
    supplied via ``cfg.teacher_load_path``, the run uses the full 3-term
    CL loss; otherwise it degenerates to the standard SARL path
    (still through the CL networks — ``SarlCLQNetwork`` etc. — because
    the architectures differ even without a teacher).
    """
    device = torch.device(cfg.device)
    state_shape = env.state_shape()
    num_actions = env.num_actions()
    in_channels = state_shape[-1]

    log.info(
        "SARL+CL training starting: game=%s seed=%d meta=%s cascade=(%d,%d) "
        "frames=%d curriculum=%s adaptive=%s teacher=%s",
        cfg.game,
        cfg.seed,
        cfg.meta,
        cfg.cascade_iterations_1,
        cfg.cascade_iterations_2,
        cfg.num_frames,
        cfg.curriculum,
        cfg.adaptive_backbone,
        cfg.teacher_load_path,
    )

    policy_net, target_net, second_order_net, teacher_first, teacher_second = _build_networks(
        in_channels, num_actions, cfg
    )
    optimizer, optimizer2, scheduler1, scheduler2 = _build_optimizers(
        policy_net, second_order_net, cfg
    )

    # One weighter per network — persists across the curriculum (paper's
    # global ``loss_weighter`` / ``loss_weighter_second``).
    loss_weighter = DynamicLossWeighter() if teacher_first is not None else None
    loss_weighter_second = (
        DynamicLossWeighter() if (teacher_first is not None and cfg.meta) else None
    )

    mixing = LossMixingWeights(
        task=cfg.weight_task,
        distillation=cfg.weight_distillation,
        feature=cfg.weight_feature,
    )

    buffer = SarlReplayBuffer(cfg.replay_buffer_size)
    metrics = CLTrainingMetrics()

    t = 0
    episode_idx = 0
    policy_update_counter = 0
    wall_start = time.time()

    while t < cfg.num_frames:
        # ── Episode setup ──────────────────────────────────────────────────
        env.reset()
        state = get_state(env.state(), device=device)
        episode_return = 0.0
        episode_length = 0
        episode_loss_1_sum = 0.0
        episode_loss_2_sum = 0.0
        episode_comp1_task = 0.0
        episode_comp1_distill = 0.0
        episode_comp1_feat = 0.0
        episode_comp2_task = 0.0
        episode_comp2_distill = 0.0
        episode_comp2_feat = 0.0
        episode_update_count = 0
        episode_cl_update_count = 0  # separate counter for component breakdowns
        done = False

        policy_net.train()
        if second_order_net is not None:
            second_order_net.train()

        # ── Episode loop ───────────────────────────────────────────────────
        while not done and t < cfg.num_frames:
            sel = epsilon_greedy_action(
                state,
                policy_net,
                t=t,
                replay_start_size=cfg.replay_start_size,
                num_actions=num_actions,
                cascade_iterations_1=cfg.cascade_iterations_1,
                device=device,
            )
            reward, done = env.act(sel.action)
            reward_t = torch.tensor([[reward]], device=device, dtype=torch.float32)
            done_t = torch.tensor([[1 if done else 0]], device=device, dtype=torch.int64)
            next_state = get_state(env.state(), device=device)
            buffer.add(state, next_state, sel.action, reward_t, done_t)
            state = next_state
            episode_return += float(reward)
            episode_length += 1

            if (
                t > cfg.replay_start_size
                and len(buffer) >= cfg.batch_size
                and t % cfg.training_freq == 0
            ):
                sample: list[Transition] = buffer.sample(cfg.batch_size)
                out: SarlCLUpdateOutput = sarl_cl_update_step(
                    sample=sample,
                    policy_net=policy_net,
                    target_net=target_net,
                    second_order_net=second_order_net,
                    teacher_first_net=teacher_first,
                    teacher_second_net=teacher_second,
                    optimizer=optimizer,
                    optimizer2=optimizer2,
                    scheduler1=scheduler1,
                    scheduler2=scheduler2,
                    loss_weighter=loss_weighter,
                    loss_weighter_second=loss_weighter_second,
                    mixing=mixing,
                    meta=cfg.meta,
                    alpha=cfg.alpha,
                    gamma=cfg.gamma,
                    cascade_iterations_1=cfg.cascade_iterations_1,
                    cascade_iterations_2=cfg.cascade_iterations_2,
                    target_wager_fn=target_wager,
                    train=True,
                    device=device,
                )
                episode_loss_1_sum += float(out.loss.item())
                if out.loss_second is not None:
                    episode_loss_2_sum += float(out.loss_second.item())
                if out.components_first is not None:
                    episode_comp1_task += out.components_first.task
                    episode_comp1_distill += out.components_first.distillation
                    episode_comp1_feat += out.components_first.feature
                    if out.components_second is not None:
                        episode_comp2_task += out.components_second.task
                        episode_comp2_distill += out.components_second.distillation
                        episode_comp2_feat += out.components_second.feature
                    episode_cl_update_count += 1

                episode_update_count += 1
                policy_update_counter += 1
                metrics.total_updates += 1

                if policy_update_counter % cfg.target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    log.debug(
                        "target_net synced at update %d (frame %d)",
                        policy_update_counter,
                        t,
                    )

            t += 1

        # ── Episode bookkeeping ────────────────────────────────────────────
        episode_idx += 1
        metrics.episode_returns.append(episode_return)
        metrics.episode_lengths.append(episode_length)
        metrics.episode_frames.append(t)

        mean_loss_1 = episode_loss_1_sum / episode_update_count if episode_update_count else None
        metrics.episode_losses_first.append(mean_loss_1)

        if cfg.meta:
            mean_loss_2 = (
                episode_loss_2_sum / episode_update_count if episode_update_count else None
            )
            metrics.episode_losses_second.append(mean_loss_2)
        else:
            metrics.episode_losses_second.append(None)

        if episode_cl_update_count > 0:
            n = episode_cl_update_count
            metrics.episode_components_first_task.append(episode_comp1_task / n)
            metrics.episode_components_first_distillation.append(episode_comp1_distill / n)
            metrics.episode_components_first_feature.append(episode_comp1_feat / n)
            if cfg.meta:
                metrics.episode_components_second_task.append(episode_comp2_task / n)
                metrics.episode_components_second_distillation.append(episode_comp2_distill / n)
                metrics.episode_components_second_feature.append(episode_comp2_feat / n)
            else:
                metrics.episode_components_second_task.append(None)
                metrics.episode_components_second_distillation.append(None)
                metrics.episode_components_second_feature.append(None)
        else:
            for lst in (
                metrics.episode_components_first_task,
                metrics.episode_components_first_distillation,
                metrics.episode_components_first_feature,
                metrics.episode_components_second_task,
                metrics.episode_components_second_distillation,
                metrics.episode_components_second_feature,
            ):
                lst.append(None)

        if episode_idx % 100 == 0 or episode_idx == 1:
            log.info(
                "ep=%d frames=%d G=%.2f len=%d loss1=%s loss2=%s",
                episode_idx,
                t,
                episode_return,
                episode_length,
                f"{mean_loss_1:.4f}" if mean_loss_1 is not None else "n/a",
                f"{metrics.episode_losses_second[-1]:.4f}"
                if metrics.episode_losses_second[-1] is not None
                else "n/a",
            )

        # ── Validation ─────────────────────────────────────────────────────
        if episode_idx % cfg.validation_every_episodes == 0:
            summary = aggregate_validation(
                env,
                policy_net,
                cfg.cascade_iterations_1,
                n_episodes=cfg.validation_iterations,
                second_order_net=second_order_net,
                cascade_iterations_2=cfg.cascade_iterations_2 if cfg.meta else None,
                device=device,
            )
            metrics.validation_frames.append(t)
            metrics.validation_summaries.append(summary)
            log.info(
                "validation @ frame %d: mean_return=%.2f ± %.2f (n=%d) bet_ratio=%s",
                t,
                summary.mean_return,
                summary.std_return,
                summary.n_episodes,
                f"{summary.mean_bet_ratio:.3f}" if summary.mean_bet_ratio is not None else "n/a",
            )

    metrics.total_frames = t
    metrics.wall_time_seconds = time.time() - wall_start
    log.info(
        "SARL+CL training complete: %d frames, %d episodes, %d updates in %.1fs",
        t,
        episode_idx,
        metrics.total_updates,
        metrics.wall_time_seconds,
    )

    if cfg.output_dir is not None:
        _persist_outputs(policy_net, second_order_net, metrics, cfg)

    return policy_net, second_order_net, metrics


def _persist_outputs(
    policy_net: SarlCLQNetwork | AdaptiveQNetwork,
    second_order_net: SarlCLSecondOrderNetwork | None,
    metrics: CLTrainingMetrics,
    cfg: SarlCLTrainingConfig,
) -> None:
    """Dump final weights + metrics. Schema kept compatible with teacher loading.

    Saves a subset of the paper's checkpoint keys — enough to serve as a
    teacher for the next curriculum stage via ``cfg.teacher_load_path``.
    """
    assert cfg.output_dir is not None
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "policy_net_state_dict": policy_net.state_dict(),
        "second_net_state_dict": second_order_net.state_dict() if second_order_net else None,
        "game": cfg.game,
        "seed": cfg.seed,
        "meta": cfg.meta,
        "cascade_iterations_1": cfg.cascade_iterations_1,
        "cascade_iterations_2": cfg.cascade_iterations_2,
        "num_frames": cfg.num_frames,
    }
    torch.save(checkpoint, out / "checkpoint.pt")

    # Plain-JSON metrics for quick inspection.
    import json

    summaries = [
        {
            "mean_return": s.mean_return,
            "std_return": s.std_return,
            "mean_steps": s.mean_steps,
            "n_episodes": s.n_episodes,
            "mean_bet_ratio": s.mean_bet_ratio,
        }
        for s in metrics.validation_summaries
    ]
    payload = {
        "game": cfg.game,
        "seed": cfg.seed,
        "meta": cfg.meta,
        "cascade_iterations_1": cfg.cascade_iterations_1,
        "cascade_iterations_2": cfg.cascade_iterations_2,
        "num_frames": cfg.num_frames,
        "curriculum": cfg.curriculum,
        "adaptive_backbone": cfg.adaptive_backbone,
        "teacher_load_path": str(cfg.teacher_load_path) if cfg.teacher_load_path else None,
        "episode_returns": metrics.episode_returns,
        "episode_lengths": metrics.episode_lengths,
        "episode_frames": metrics.episode_frames,
        "episode_losses_first": metrics.episode_losses_first,
        "episode_losses_second": metrics.episode_losses_second,
        "episode_components_first_task": metrics.episode_components_first_task,
        "episode_components_first_distillation": metrics.episode_components_first_distillation,
        "episode_components_first_feature": metrics.episode_components_first_feature,
        "episode_components_second_task": metrics.episode_components_second_task,
        "episode_components_second_distillation": metrics.episode_components_second_distillation,
        "episode_components_second_feature": metrics.episode_components_second_feature,
        "validation_frames": metrics.validation_frames,
        "validation_summaries": summaries,
        "total_updates": metrics.total_updates,
        "total_frames": metrics.total_frames,
        "wall_time_seconds": metrics.wall_time_seconds,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    log.info("CL checkpoint + metrics persisted to %s", out)


# ── Paper setting → config mapping ──────────────────────────────────────────


_SETTING_TABLE = {
    1: (False, 1, 1),
    2: (False, 50, 1),
    3: (True, 1, 1),
    4: (True, 50, 1),
    5: (True, 1, 50),
    6: (True, 50, 50),
}


def setting_to_config_cl(
    setting: int, base: SarlCLTrainingConfig | None = None
) -> SarlCLTrainingConfig:
    """Translate paper setting 1-6 into (meta, cascade_1, cascade_2) for CL.

    Preserves all other fields of ``base`` — so curriculum/adaptive/teacher
    flags set by the caller survive the translation.
    """
    if setting not in _SETTING_TABLE:
        raise ValueError(f"setting must be 1-6, got {setting}")
    meta, c1, c2 = _SETTING_TABLE[setting]
    cfg = base if base is not None else SarlCLTrainingConfig()
    return SarlCLTrainingConfig(
        **{
            **cfg.__dict__,
            "meta": meta,
            "cascade_iterations_1": c1,
            "cascade_iterations_2": c2,
        }
    )
