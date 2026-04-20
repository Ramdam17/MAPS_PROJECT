"""SARL training orchestrator — ties replay, rollout, update, and validation.

Ports the training-loop half of ``external/MinAtar/examples/maps.py`` (the
core of ``dqn()`` at lines 1095-2145, standard-DQN path only — continuous
learning lives in ``maps.experiments.sarl_cl`` per Sprint 04b §4.6).

Scope
-----
What this module does:

* Initialize networks, optimizer(s), scheduler(s), and replay buffer.
* Run the outer training loop: episode → env-step → buffer-add → ε-greedy
  action → update via ``sarl_update_step`` (when buffer warm) → target-net
  sync every ``TARGET_NETWORK_UPDATE_FREQ`` frames.
* Every ``checkpoint_every_episodes`` episodes, run validation rollouts.
* Optionally persist metrics + state dicts.

What this module does **not** do:

* Continuous learning / teacher networks / distillation loss — see
  ``sarl_cl.training_loop``.
* Resuming from a paper-format checkpoint — the paper's checkpoint schema
  has ~20 keys including curriculum flags; we use a smaller schema here
  and leave legacy-checkpoint import for Sprint 06 if needed.
* Plotting / z-score computation — those live in ``scripts/`` / notebooks.

Parity boundary
---------------
The **inner update** (``sarl_update_step``) is Tier-3 bit-exact against the
paper (see ``tests/parity/sarl/test_tier3_update.py``). The **outer loop**
is structurally faithful but **not** bit-exact against the paper because:

* ε-greedy RNG consumption depends on how many times the greedy branch is
  taken, which depends on the trained weights — a tiny numerical drift
  anywhere upstream would reshuffle draws.
* Episode boundaries depend on env.act() which depends on its own RNG.
* Target-net sync cadence depends on the exact counter value when crossing
  thresholds.

Sprint 07 will reproduce paper z-scores across 30 seeds (which is
statistically robust even without bit-exact RNG) — this module is the
engine that Sprint 07 drives.

References
----------
- Mnih et al. (2015). Human-level control through deep reinforcement learning.
  Nature 518:529-533. — outer DQN loop structure.
- Vargas et al. (2025), MAPS TMLR submission §3.
"""

from __future__ import annotations

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
from maps.experiments.sarl.model import SarlQNetwork, SarlSecondOrderNetwork
from maps.experiments.sarl.rollout import epsilon_greedy_action
from maps.experiments.sarl.trainer import sarl_update_step

log = logging.getLogger(__name__)

# ── Paper constants (source: external/MinAtar/examples/maps.py:92-107) ──────

BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 100_000
REPLAY_START_SIZE = 5_000
TRAINING_FREQ = 1  # update every N frames once warmup is done
TARGET_NETWORK_UPDATE_FREQ = 1_000  # sync cadence, in policy-update steps
MIN_SQUARED_GRAD = 0.01  # Adam eps
STEP_SIZE_1 = 0.0003  # policy-net learning rate
STEP_SIZE_2 = 0.00005  # second-order learning rate
SCHEDULER_STEP = 0.999  # StepLR gamma
SCHEDULER_PERIOD = 1_000  # StepLR step_size


class MinAtarLike(Protocol):
    """Minimal env interface (same contract as evaluate.MinAtarLike)."""

    def reset(self) -> Any: ...
    def state(self) -> np.ndarray: ...
    def act(self, action: torch.Tensor) -> tuple[float, bool]: ...
    def num_actions(self) -> int: ...
    def state_shape(self) -> tuple[int, int, int]: ...


# ── Configuration dataclass (maps 1:1 to CLI args and config.yaml) ──────────


@dataclass
class SarlTrainingConfig:
    """Hyperparameters for :func:`run_training`.

    Paper settings 1-6 translate to the ``meta`` / ``cascade_iterations_*``
    fields as follows (see ``maps.py:2683-2710``):

    ======= =========== =============== ===============
    setting meta        cascade_iter_1  cascade_iter_2
    ======= =========== =============== ===============
    1       False        1               1              (vanilla DQN)
    2       False       50               1              (cascade on FO only)
    3       True         1               1              (meta on, cascade off)
    4       True        50               1              (meta + cascade on FO)
    5       True         1              50              (meta + cascade on SO)
    6       True        50              50              (meta + cascade on both)
    ======= =========== =============== ===============
    """

    # Environment.
    game: str = "space_invaders"
    seed: int = 42

    # Paper setting (1-6) — concrete knobs below are what actually drive training.
    meta: bool = False
    cascade_iterations_1: int = 1
    cascade_iterations_2: int = 1

    # DQN hyperparameters.
    num_frames: int = 5_000_000
    batch_size: int = BATCH_SIZE
    replay_buffer_size: int = REPLAY_BUFFER_SIZE
    replay_start_size: int = REPLAY_START_SIZE
    training_freq: int = TRAINING_FREQ
    target_update_freq: int = TARGET_NETWORK_UPDATE_FREQ
    step_size_1: float = STEP_SIZE_1
    step_size_2: float = STEP_SIZE_2
    scheduler_period: int = SCHEDULER_PERIOD
    scheduler_gamma: float = SCHEDULER_STEP

    # EMA coefficient for target_wager (paper passes percent then /100).
    alpha: float = 1.0

    # Validation cadence.
    validation_every_episodes: int = 50
    validation_iterations: int = 3

    # Runtime.
    device: str = "cpu"
    output_dir: Path | None = None  # if set, dump metrics JSON + final weights


@dataclass
class TrainingMetrics:
    """Captured during training for later inspection / plotting."""

    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_frames: list[int] = field(default_factory=list)  # cumulative t at end of ep
    episode_losses_first: list[float | None] = field(default_factory=list)
    episode_losses_second: list[float | None] = field(default_factory=list)
    validation_frames: list[int] = field(default_factory=list)
    validation_summaries: list[ValidationSummary] = field(default_factory=list)
    total_updates: int = 0
    total_frames: int = 0
    wall_time_seconds: float = 0.0


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_networks(
    in_channels: int, num_actions: int, cfg: SarlTrainingConfig
) -> tuple[SarlQNetwork, SarlQNetwork, SarlSecondOrderNetwork | None]:
    """Construct policy / target / optional second-order nets.

    Init order matters for init-RNG reproducibility — keep policy→target→second.
    """
    # Local re-seed to keep network init deterministic even if caller reused the
    # RNG between calls. The *correct* entry-point seeding (Python/NumPy/CUDA)
    # must be done upstream via ``maps.utils.seeding.set_all_seeds(cfg.seed)``
    # — the CLI (``scripts/run_sarl.py``) does this; don't call
    # ``_build_networks`` without that prelude.
    torch.manual_seed(cfg.seed)
    policy = SarlQNetwork(in_channels, num_actions).to(cfg.device)
    target = SarlQNetwork(in_channels, num_actions).to(cfg.device)
    target.load_state_dict(policy.state_dict())
    target.eval()  # target net never trains
    second = SarlSecondOrderNetwork(in_channels).to(cfg.device) if cfg.meta else None
    return policy, target, second


def _build_optimizers(
    policy: torch.nn.Module,
    second: torch.nn.Module | None,
    cfg: SarlTrainingConfig,
) -> tuple[optim.Optimizer, optim.Optimizer | None, Any, Any | None]:
    """Adam + StepLR per paper convention."""
    opt1 = optim.Adam(policy.parameters(), lr=cfg.step_size_1, eps=MIN_SQUARED_GRAD)
    sch1 = StepLR(opt1, step_size=cfg.scheduler_period, gamma=cfg.scheduler_gamma)
    opt2: optim.Optimizer | None = None
    sch2: Any | None = None
    if cfg.meta:
        assert second is not None
        opt2 = optim.Adam(second.parameters(), lr=cfg.step_size_2, eps=MIN_SQUARED_GRAD)
        sch2 = StepLR(opt2, step_size=cfg.scheduler_period, gamma=cfg.scheduler_gamma)
    return opt1, opt2, sch1, sch2


# ── Main entry point ────────────────────────────────────────────────────────


def run_training(
    env: MinAtarLike,
    cfg: SarlTrainingConfig,
) -> tuple[SarlQNetwork, SarlSecondOrderNetwork | None, TrainingMetrics]:
    """Train a SARL agent for ``cfg.num_frames`` frames. Returns trained nets + metrics.

    The env is consumed in-place (reset between episodes). Pass a fresh env
    for each call.
    """
    device = torch.device(cfg.device)
    state_shape = env.state_shape()
    num_actions = env.num_actions()
    in_channels = state_shape[-1]  # MinAtar is HWC → last axis is channels

    log.info(
        "SARL training starting: game=%s seed=%d meta=%s cascade=(%d,%d) frames=%d",
        cfg.game,
        cfg.seed,
        cfg.meta,
        cfg.cascade_iterations_1,
        cfg.cascade_iterations_2,
        cfg.num_frames,
    )

    policy_net, target_net, second_order_net = _build_networks(in_channels, num_actions, cfg)
    optimizer, optimizer2, scheduler1, scheduler2 = _build_optimizers(
        policy_net, second_order_net, cfg
    )
    buffer = SarlReplayBuffer(cfg.replay_buffer_size)
    metrics = TrainingMetrics()

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
        episode_update_count = 0
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

            # Update once buffer is warm AND we've hit a training-freq frame.
            if (
                t > cfg.replay_start_size
                and len(buffer) >= cfg.batch_size
                and t % cfg.training_freq == 0
            ):
                sample: list[Transition] = buffer.sample(cfg.batch_size)
                out = sarl_update_step(
                    sample=sample,
                    policy_net=policy_net,
                    target_net=target_net,
                    second_order_net=second_order_net,
                    optimizer=optimizer,
                    optimizer2=optimizer2,
                    scheduler1=scheduler1,
                    scheduler2=scheduler2,
                    meta=cfg.meta,
                    alpha=cfg.alpha,
                    cascade_iterations_1=cfg.cascade_iterations_1,
                    cascade_iterations_2=cfg.cascade_iterations_2,
                    target_wager_fn=target_wager,
                    device=device,
                )
                episode_loss_1_sum += float(out.loss.item())
                if out.loss_second is not None:
                    episode_loss_2_sum += float(out.loss_second.item())
                episode_update_count += 1
                policy_update_counter += 1
                metrics.total_updates += 1

                # Target-net sync on the update-counter clock (not frame clock).
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
        "SARL training complete: %d frames, %d episodes, %d updates in %.1fs",
        t,
        episode_idx,
        metrics.total_updates,
        metrics.wall_time_seconds,
    )

    if cfg.output_dir is not None:
        _persist_outputs(policy_net, second_order_net, metrics, cfg)

    return policy_net, second_order_net, metrics


def _persist_outputs(
    policy_net: SarlQNetwork,
    second_order_net: SarlSecondOrderNetwork | None,
    metrics: TrainingMetrics,
    cfg: SarlTrainingConfig,
) -> None:
    """Dump final weights + metrics to ``cfg.output_dir``.

    Minimal schema — Sprint 07 will extend if needed.
    """
    assert cfg.output_dir is not None
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(policy_net.state_dict(), out / "policy_net.pt")
    if second_order_net is not None:
        torch.save(second_order_net.state_dict(), out / "second_order_net.pt")

    # Metrics as JSON — ValidationSummary is a dataclass so we serialize manually.
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
        "episode_returns": metrics.episode_returns,
        "episode_lengths": metrics.episode_lengths,
        "episode_frames": metrics.episode_frames,
        "episode_losses_first": metrics.episode_losses_first,
        "episode_losses_second": metrics.episode_losses_second,
        "validation_frames": metrics.validation_frames,
        "validation_summaries": summaries,
        "total_updates": metrics.total_updates,
        "total_frames": metrics.total_frames,
        "wall_time_seconds": metrics.wall_time_seconds,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    log.info("metrics persisted to %s", out)


# ── Paper setting → config mapping ──────────────────────────────────────────


_SETTING_TABLE = {
    # setting: (meta, cascade_1, cascade_2)
    1: (False, 1, 1),
    2: (False, 50, 1),
    3: (True, 1, 1),
    4: (True, 50, 1),
    5: (True, 1, 50),
    6: (True, 50, 50),
}


def setting_to_config(setting: int, base: SarlTrainingConfig | None = None) -> SarlTrainingConfig:
    """Translate paper's integer 1-6 setting into (meta, cascade_1, cascade_2).

    Parameters
    ----------
    setting : int (1-6)
        Paper setting — see ``maps.py:2683-2710`` for canonical mapping.
    base : SarlTrainingConfig, optional
        Seed config whose other fields are preserved. When None, a fresh
        config with defaults is returned (game=space_invaders, seed=42, …).
    """
    if setting not in _SETTING_TABLE:
        raise ValueError(f"setting must be 1-6, got {setting}")
    meta, c1, c2 = _SETTING_TABLE[setting]
    cfg = base if base is not None else SarlTrainingConfig()
    return SarlTrainingConfig(
        **{
            **cfg.__dict__,
            "meta": meta,
            "cascade_iterations_1": c1,
            "cascade_iterations_2": c2,
        }
    )
