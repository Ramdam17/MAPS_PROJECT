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
import os
import random
import time
from dataclasses import asdict, dataclass, field
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

# Sprint-08 D.9+D.12 (2026-04-20): 3 constants aligned paper-faithful
# (STEP_SIZE_2, ADAM_BETAS, SCHEDULER_PERIOD). BATCH_SIZE and STEP_SIZE_1
# were ALREADY paper-faithful pre-D.9; D.9 mistakenly changed them based
# on misreading and D.12 restores the correct paper-Table-11 values.
# Override via CLI (`-o training.num_frames=5000000 ...`) for student/legacy
# reproduction. See docs/reproduction/deviations.md D-sarl-{num-frames,
# lr-2nd,adam-beta1/2,sched-step}.
BATCH_SIZE = 128  # paper Table 11 row 1 (student 128 too) — paper-faithful
REPLAY_BUFFER_SIZE = 100_000
REPLAY_START_SIZE = 5_000
TRAINING_FREQ = 1  # update every N frames once warmup is done
TARGET_NETWORK_UPDATE_FREQ = 1_000  # sync cadence, in policy-update steps
MIN_SQUARED_GRAD = 0.01  # Adam eps
STEP_SIZE_1 = 0.0003  # policy-net learning rate — paper Table 11 row 9 (paper-faithful)
STEP_SIZE_2 = 0.0002  # second-order learning rate — paper Table 11 (was 0.00005 — D-sarl-lr-2nd)
ADAM_BETAS: tuple[float, float] = (0.95, 0.95)  # paper Table 11 (student omitted → PyTorch default 0.9/0.999)
SCHEDULER_STEP = 0.999  # StepLR gamma (paper silent; student value kept)
SCHEDULER_PERIOD = 1  # StepLR step_size — paper Table 11 (suspected typo, student used 1000); see _build_optimizers


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
    # num_frames = 500_000 per paper Table 11 (D.12); override to 5_000_000
    # for legacy / student-empirical reproduction. See D-sarl-num-frames.
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
    # Paper Table 11 betas=(0.95, 0.95); student omits → PyTorch default
    # (0.9, 0.999). Aligned paper-faithful 2026-04-20 (D.9). See D-sarl-adam-beta1/2.
    adam_betas: tuple[float, float] = ADAM_BETAS
    # Paper Table 11 γ=0.999 (aligned 2026-04-20, D.7). Override to 0.99 for
    # student-baseline reproduction. See deviations.md D-sarl-gamma.
    gamma: float = 0.999

    # EMA coefficient for target_wager (paper passes percent then /100).
    # Paper Table 11 α=45 (→ 0.45 post-div). Aligned 2026-04-20 (D.2; D.9
    # aligns dataclass default with config yaml). See D-sarl-alpha-ema.
    alpha: float = 45.0

    # Sprint-08 D.22b: paper eq.4 first-order loss family. 'cae' (default,
    # paper-faithful-via-student-code) or 'simclr' (paper-prose; not ported —
    # raises NotImplementedError at setup if selected). See
    # docs/reports/sprint-08-d22b-simclr-decision.md.
    first_order_loss_kind: str = "cae"

    # Validation cadence.
    validation_every_episodes: int = 50
    validation_iterations: int = 3

    # Checkpoint cadence (Sprint-08 D.13). Checkpoint every N policy updates
    # to `output_dir / checkpoint.pt`. Set to 0 to disable intra-training
    # checkpoints (the final one at end-of-training is still written if
    # output_dir is set). Resume semantics: see `resume_from` + run_training.
    checkpoint_every_updates: int = 10_000

    # When set, run_training loads the checkpoint at this path before the
    # training loop (instead of starting fresh). The cfg at this point MUST
    # be compatible with the cfg snapshot inside the checkpoint; mismatches
    # on game/seed/meta/cascade/num_frames raise ValueError.
    resume_from: Path | None = None

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
    # Sprint-08 D.4: cascade_effective_iters_1 is always 1 because
    # SarlQNetwork.forward has no dropout → cascade is no-op on that path.
    # cascade_effective_iters_2 matches the config value when meta=True (2nd-order
    # path has dropout, cascade averages 50 masks) else None (2nd-order disabled).
    # See deviations.md D-sarl-cascade-noop + docs/reviews/cascade.md §(d).
    cascade_effective_iters_1: int = 1
    cascade_effective_iters_2: int | None = None


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
    # Sprint-08 D.9: the paper-faithful scheduler step_size=1 decays the LR at
    # every update → on a 5M-frame run with γ=0.999 the LR collapses to ~0
    # after ~5k updates. This is what the paper prescribes literally, but the
    # behaviour is suspicious enough to warrant a loud warning so the
    # reviewer can verify / override. Student used step_size=1000 which is
    # far more moderate; see sarl.yaml scheduler block for overrides.
    if cfg.scheduler_period == 1:
        log.warning(
            "scheduler_period=1 (paper Table 11 as-written): LR decays every "
            "update; with γ=%.3f, effective LR ≈ 0 after ~%d updates. This "
            "may be a paper typo — override with `-o scheduler.step_size=1000` "
            "to reproduce the student baseline. See deviations.md D-sarl-sched-step.",
            cfg.scheduler_gamma,
            # Estimate of when LR drops below 1% of initial.
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


# ── Checkpoint / resume (Sprint-08 D.13) ────────────────────────────────────

#: Sprint-08 D.22b — allowed values for the first-order-loss toggle.
#: Only 'cae' is implemented; 'simclr' is a reservation guard-rail.
_FIRST_ORDER_LOSS_KINDS: frozenset[str] = frozenset({"cae", "simclr"})


def _check_first_order_loss_kind(kind: str) -> None:
    """Validate the D.22b toggle before training starts.

    Raises
    ------
    NotImplementedError
        If ``kind == "simclr"``. Paper §2.2 prose describes SimCLR but no
        existing implementation (paper author code or our port) realises it;
        D.22b decision is to keep CAE until Phase F results motivate a flip.
        See docs/reports/sprint-08-d22b-simclr-decision.md.
    ValueError
        If ``kind`` is not one of ``_FIRST_ORDER_LOSS_KINDS``.
    """
    if kind == "simclr":
        raise NotImplementedError(
            "first_order_loss.kind='simclr' is a paper-prose-faithful variant "
            "that has not been ported. The paper's own code and our port both "
            "use 'cae'. See docs/reports/sprint-08-d22b-simclr-decision.md for "
            "the decision log + revisit conditions, and flip this toggle back "
            "to 'cae' or implement SimCLR before rerunning."
        )
    if kind not in _FIRST_ORDER_LOSS_KINDS:
        raise ValueError(
            f"first_order_loss.kind must be one of {sorted(_FIRST_ORDER_LOSS_KINDS)}, "
            f"got {kind!r}."
        )


#: Format version for the checkpoint payload. Bump when the schema changes
#: in a way the `_restore_from_checkpoint` loader cannot silently handle.
_CHECKPOINT_FORMAT_VERSION = 1

#: Fields from `SarlTrainingConfig` that must match between the checkpoint
#: snapshot and the caller's cfg at resume time. Other fields (e.g.
#: `output_dir`, `device`, `validation_every_episodes`) may legitimately
#: differ and are not compared.
_CHECKPOINT_CFG_GUARDS: tuple[str, ...] = (
    "game",
    "seed",
    "meta",
    "cascade_iterations_1",
    "cascade_iterations_2",
    "num_frames",
)


def _persist_checkpoint(
    checkpoint_path: Path,
    *,
    t: int,
    episode_idx: int,
    policy_update_counter: int,
    policy_net: torch.nn.Module,
    target_net: torch.nn.Module,
    second_order_net: torch.nn.Module | None,
    optimizer: optim.Optimizer,
    optimizer2: optim.Optimizer | None,
    scheduler1: Any,
    scheduler2: Any | None,
    buffer: SarlReplayBuffer,
    metrics: TrainingMetrics,
    cfg: SarlTrainingConfig,
) -> None:
    """Atomically persist the full SARL training state to ``checkpoint_path``.

    Writes first to ``checkpoint_path.with_suffix('.pt.tmp')`` then performs
    ``os.replace`` (POSIX-atomic rename) so an interrupted write cannot leave
    a half-valid file at the target path.

    The payload captures every piece of state needed for bit-exact resume:
    step counters, all model weights, both optimizers' state, both schedulers'
    state, the replay buffer contents, the metrics collected so far, and the
    RNG states of ``torch`` / Python ``random`` / ``numpy.random`` (legacy
    singleton — matches `maps.utils.seeding.set_all_seeds`).

    Buffer serialisation uses the list of `Transition` namedtuples directly;
    it is picklable because each `Transition` field is a CPU tensor or scalar.
    For a 100k-buffer of (10×10×C) MinAtar transitions this adds ~300-500 MB
    to the file size — acceptable for the resume cadence (every 10k updates).
    """
    payload: dict[str, Any] = {
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        # Counters.
        "t": t,
        "episode_idx": episode_idx,
        "policy_update_counter": policy_update_counter,
        # Models.
        "policy_state_dict": policy_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "second_order_state_dict": (
            second_order_net.state_dict() if second_order_net is not None else None
        ),
        # Optim + sched.
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer2_state_dict": optimizer2.state_dict() if optimizer2 is not None else None,
        "scheduler1_state_dict": scheduler1.state_dict(),
        "scheduler2_state_dict": scheduler2.state_dict() if scheduler2 is not None else None,
        # Buffer (list of Transitions + cyclic write head).
        "buffer_buffer": buffer.buffer,
        "buffer_location": buffer.location,
        "buffer_size": buffer.buffer_size,
        # Metrics — convert dataclass to dict so load doesn't require the class
        # to be identical across versions (ValidationSummary is nested; torch
        # saves the tuple of them without issue).
        "metrics": metrics,
        # Cfg snapshot (for compatibility checks at resume time).
        "cfg_snapshot": asdict(cfg),
        # RNG states.
        "rng_torch": torch.get_rng_state(),
        "rng_torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "rng_python": random.getstate(),
        "rng_numpy_legacy": np.random.get_state(),
    }

    # Atomic write: tmp → rename. Parent dir must exist; caller ensures via
    # `paths.ensure_dirs()` typically, but we mkdir defensively as a fallback.
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, checkpoint_path)
    log.info(
        "checkpoint persisted: %s (t=%d, episode=%d, updates=%d)",
        checkpoint_path,
        t,
        episode_idx,
        policy_update_counter,
    )


def _restore_from_checkpoint(
    checkpoint_path: Path,
    cfg: SarlTrainingConfig,
    policy_net: torch.nn.Module,
    target_net: torch.nn.Module,
    second_order_net: torch.nn.Module | None,
    optimizer: optim.Optimizer,
    optimizer2: optim.Optimizer | None,
    scheduler1: Any,
    scheduler2: Any | None,
) -> tuple[int, int, int, SarlReplayBuffer, TrainingMetrics]:
    """Load a checkpoint and restore all state in place.

    Returns ``(t, episode_idx, policy_update_counter, buffer, metrics)``.
    The caller's ``policy_net`` / ``target_net`` / ``second_order_net`` /
    ``optimizer`` / ``optimizer2`` / ``scheduler1`` / ``scheduler2`` are all
    mutated in place via their respective ``load_state_dict`` methods.

    Validates that the checkpoint's ``cfg_snapshot`` agrees with the caller's
    ``cfg`` on the fields listed in ``_CHECKPOINT_CFG_GUARDS``. A mismatch
    raises ``ValueError`` with the offending fields listed — the caller should
    not silently resume a run with different paper settings / seed.

    Also restores the three RNG streams (torch, Python random, numpy legacy).
    CUDA RNG is restored when both the checkpoint has it AND CUDA is available
    (if not available the cached state is ignored with a debug log).
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)

    version = payload.get("format_version")
    if version != _CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format_version={version!r} is incompatible with this "
            f"runtime's expected {_CHECKPOINT_FORMAT_VERSION}. Refusing to resume."
        )

    # Cfg compatibility guardrail.
    snapshot = payload["cfg_snapshot"]
    current = asdict(cfg)
    mismatches = {
        k: (snapshot.get(k), current.get(k))
        for k in _CHECKPOINT_CFG_GUARDS
        if snapshot.get(k) != current.get(k)
    }
    if mismatches:
        raise ValueError(
            "checkpoint cfg mismatch on guarded fields — refusing to resume. "
            f"Differences (checkpoint → caller): {mismatches}"
        )

    # Models.
    policy_net.load_state_dict(payload["policy_state_dict"])
    target_net.load_state_dict(payload["target_state_dict"])
    if second_order_net is not None:
        so_state = payload["second_order_state_dict"]
        if so_state is None:
            raise ValueError(
                "checkpoint has no second-order state but caller expects meta=True"
            )
        second_order_net.load_state_dict(so_state)
    elif payload["second_order_state_dict"] is not None:
        log.warning(
            "checkpoint has second-order state but caller is meta=False; discarded"
        )

    # Optim + sched.
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler1.load_state_dict(payload["scheduler1_state_dict"])
    if optimizer2 is not None:
        opt2_state = payload["optimizer2_state_dict"]
        if opt2_state is None:
            raise ValueError("checkpoint has no optimizer2 but caller expects meta=True")
        optimizer2.load_state_dict(opt2_state)
    if scheduler2 is not None:
        sch2_state = payload["scheduler2_state_dict"]
        if sch2_state is None:
            raise ValueError("checkpoint has no scheduler2 but caller expects meta=True")
        scheduler2.load_state_dict(sch2_state)

    # Buffer.
    buffer = SarlReplayBuffer(payload["buffer_size"])
    buffer.buffer = payload["buffer_buffer"]
    buffer.location = payload["buffer_location"]

    # RNG.
    torch.set_rng_state(payload["rng_torch"])
    if payload.get("rng_torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["rng_torch_cuda"])
    elif payload.get("rng_torch_cuda") is not None:
        log.debug("checkpoint has CUDA RNG state but CUDA unavailable here; ignored")
    random.setstate(payload["rng_python"])
    np.random.set_state(payload["rng_numpy_legacy"])

    metrics = payload["metrics"]
    log.info(
        "checkpoint restored: %s (resuming at t=%d, episode=%d, updates=%d)",
        checkpoint_path,
        payload["t"],
        payload["episode_idx"],
        payload["policy_update_counter"],
    )
    return (
        int(payload["t"]),
        int(payload["episode_idx"]),
        int(payload["policy_update_counter"]),
        buffer,
        metrics,
    )


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

    # Sprint-08 D.22b: fail-fast if the D-002 toggle isn't the implemented path.
    _check_first_order_loss_kind(cfg.first_order_loss_kind)

    log.info(
        "SARL training starting: game=%s seed=%d meta=%s cascade=(%d,%d) frames=%d",
        cfg.game,
        cfg.seed,
        cfg.meta,
        cfg.cascade_iterations_1,
        cfg.cascade_iterations_2,
        cfg.num_frames,
    )

    # Sprint-08 D.4 (Option A, paper-faithful): the SARL 1st-order forward
    # (SarlQNetwork) has no dropout → cascade is a mathematical no-op on that
    # path regardless of cascade_iterations_1. We keep the paper-prescribed
    # value (50 for settings 2/4/6) for parity; post-reproduction we may add
    # dropout or a shortcut. Warn once so the behaviour is discoverable from
    # run logs. See docs/reviews/cascade.md §(d) + deviations.md
    # D-sarl-cascade-noop. Effective iters are also written to metrics.json.
    if cfg.cascade_iterations_1 > 1:
        log.warning(
            "cascade_iterations_1=%d but SarlQNetwork.forward is deterministic "
            "(no dropout): cascade is a no-op on the 1st-order path — kept for "
            "paper parity (effective=1). See deviations.md D-sarl-cascade-noop.",
            cfg.cascade_iterations_1,
        )

    policy_net, target_net, second_order_net = _build_networks(in_channels, num_actions, cfg)
    optimizer, optimizer2, scheduler1, scheduler2 = _build_optimizers(
        policy_net, second_order_net, cfg
    )

    # Resume from checkpoint if requested (Sprint-08 D.13). The restore runs
    # AFTER network + optimizer construction because we need their objects to
    # call `.load_state_dict` on them in place. Buffer + metrics are
    # reconstructed from the checkpoint payload.
    if cfg.resume_from is not None:
        t, episode_idx, policy_update_counter, buffer, metrics = _restore_from_checkpoint(
            cfg.resume_from,
            cfg,
            policy_net,
            target_net,
            second_order_net,
            optimizer,
            optimizer2,
            scheduler1,
            scheduler2,
        )
        log.info("resumed from %s at t=%d", cfg.resume_from, t)
    else:
        buffer = SarlReplayBuffer(cfg.replay_buffer_size)
        metrics = TrainingMetrics()
        # Record the effective cascade iteration counts (D.4). 1st-order forward
        # has no dropout → effective=1 regardless of config. 2nd-order has
        # dropout p=0.1 → config value is the true effective count when meta
        # is enabled.
        metrics.cascade_effective_iters_1 = 1
        metrics.cascade_effective_iters_2 = cfg.cascade_iterations_2 if cfg.meta else None
        t = 0
        episode_idx = 0
        policy_update_counter = 0

    wall_start = time.time()
    checkpoint_path: Path | None = (
        cfg.output_dir / "checkpoint.pt" if cfg.output_dir is not None else None
    )

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
                    gamma=cfg.gamma,
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

                # Intra-training checkpoint (Sprint-08 D.13). Fires only when
                # output_dir is set AND the user enabled it with
                # checkpoint_every_updates > 0.
                if (
                    checkpoint_path is not None
                    and cfg.checkpoint_every_updates > 0
                    and policy_update_counter % cfg.checkpoint_every_updates == 0
                ):
                    _persist_checkpoint(
                        checkpoint_path,
                        t=t,
                        episode_idx=episode_idx,
                        policy_update_counter=policy_update_counter,
                        policy_net=policy_net,
                        target_net=target_net,
                        second_order_net=second_order_net,
                        optimizer=optimizer,
                        optimizer2=optimizer2,
                        scheduler1=scheduler1,
                        scheduler2=scheduler2,
                        buffer=buffer,
                        metrics=metrics,
                        cfg=cfg,
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

    # Final checkpoint — always write when output_dir is set, regardless of
    # checkpoint_every_updates (so the user gets a resume-able end-of-training
    # state even if intra-training checkpoints were disabled).
    if checkpoint_path is not None:
        _persist_checkpoint(
            checkpoint_path,
            t=t,
            episode_idx=episode_idx,
            policy_update_counter=policy_update_counter,
            policy_net=policy_net,
            target_net=target_net,
            second_order_net=second_order_net,
            optimizer=optimizer,
            optimizer2=optimizer2,
            scheduler1=scheduler1,
            scheduler2=scheduler2,
            buffer=buffer,
            metrics=metrics,
            cfg=cfg,
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
        "cascade_effective_iters_1": metrics.cascade_effective_iters_1,
        "cascade_effective_iters_2": metrics.cascade_effective_iters_2,
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
