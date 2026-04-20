"""Tier 3 parity tests — SARL+CL update-step equivalence.

Asserts that ``maps.experiments.sarl_cl.trainer.sarl_cl_update_step``
reproduces ``_reference_sarl_cl.reference_dqn_update_step_cl`` (verbatim-
structure transcription of the paper's CL ``train()``) **bit-exactly** on
one full update, starting from identical weights and fed an identical sample.

Covers two branches:

1. **Non-CL (degenerate)**: ``teacher_first_net=None``. Single-term loss
   (CAE on FO, BCE-with-logits on SO).
2. **CL**: teacher present. Three-term loss per network (task + distillation
   + feature), normalised by a shared ``DynamicLossWeighter`` instance
   (same object handed to both ref and port so their running-max states
   stay perfectly synced after the update call).

Both branches run with ``meta=True`` so we exercise the SO path, and with
``cascade_iterations = 1`` (cascade off) to keep the tests fast. Cascade-on
parity is implicitly covered by Tier 1.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from maps.experiments.sarl.data import SarlReplayBuffer, Transition
from maps.experiments.sarl_cl.loss_weighting import DynamicLossWeighter
from maps.experiments.sarl_cl.model import (
    SarlCLQNetwork,
    SarlCLSecondOrderNetwork,
)
from maps.experiments.sarl_cl.trainer import LossMixingWeights, sarl_cl_update_step
from tests.parity.sarl_cl._reference_sarl_cl import (
    MIN_SQUARED_GRAD,
    QNetwork as RefQNetwork,
    SecondOrderNetwork as RefSecondOrderNetwork,
    WEIGHT1,
    WEIGHT2,
    WEIGHT3,
    reference_dqn_update_step_cl,
    scheduler_step,
    step_size1,
    step_size2,
    target_wager,
    transition as ref_transition,
)


# ─── Configuration ──────────────────────────────────────────────────────────

IN_CHANNELS = 4
NUM_ACTIONS = 6
BATCH_SIZE = 32
SEED = 2026
ATOL = 1e-6

CASCADE_OFF = 1


# ─── Helpers ────────────────────────────────────────────────────────────────


def _copy_state_dict(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """Copy every parameter from src into dst by name. Keys must match exactly."""
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    assert set(src_state.keys()) == set(dst_state.keys()), (
        f"state_dict keys differ: src={set(src_state.keys())} dst={set(dst_state.keys())}"
    )
    dst.load_state_dict(src_state)


def _build_networks_ref(
    seed: int,
) -> tuple[RefQNetwork, RefQNetwork, RefSecondOrderNetwork]:
    torch.manual_seed(seed)
    policy = RefQNetwork(IN_CHANNELS, NUM_ACTIONS)
    target = RefQNetwork(IN_CHANNELS, NUM_ACTIONS)
    target.load_state_dict(policy.state_dict())
    second = RefSecondOrderNetwork(IN_CHANNELS)
    return policy, target, second


def _build_networks_ours(
    ref_policy: RefQNetwork,
    ref_target: RefQNetwork,
    ref_second: RefSecondOrderNetwork,
) -> tuple[SarlCLQNetwork, SarlCLQNetwork, SarlCLSecondOrderNetwork]:
    policy = SarlCLQNetwork(IN_CHANNELS, NUM_ACTIONS)
    target = SarlCLQNetwork(IN_CHANNELS, NUM_ACTIONS)
    second = SarlCLSecondOrderNetwork(IN_CHANNELS)
    _copy_state_dict(ref_policy, policy)
    _copy_state_dict(ref_target, target)
    _copy_state_dict(ref_second, second)
    return policy, target, second


def _build_optimizers(
    policy: torch.nn.Module,
    second: torch.nn.Module,
) -> tuple[optim.Optimizer, optim.Optimizer, Any, Any]:
    opt1 = optim.Adam(policy.parameters(), lr=step_size1, eps=MIN_SQUARED_GRAD)
    sch1 = StepLR(opt1, step_size=1000, gamma=scheduler_step)
    opt2 = optim.Adam(second.parameters(), lr=step_size2, eps=MIN_SQUARED_GRAD)
    sch2 = StepLR(opt2, step_size=1000, gamma=scheduler_step)
    return opt1, opt2, sch1, sch2


def _make_sample_ref(batch_size: int, seed: int) -> list[Any]:
    """Build a reproducible batch of transitions using the reference transition."""
    random.seed(seed)
    torch.manual_seed(seed)
    samples = []
    for _ in range(batch_size):
        state = torch.randn(1, IN_CHANNELS, 10, 10)
        next_state = torch.randn(1, IN_CHANNELS, 10, 10)
        action = torch.tensor([[random.randrange(NUM_ACTIONS)]], dtype=torch.int64)
        reward = torch.tensor([[random.random()]], dtype=torch.float32)
        is_terminal = torch.tensor([[random.random() < 0.1]], dtype=torch.int64)
        samples.append(ref_transition(state, next_state, action, reward, is_terminal))
    return samples


def _sample_as_port(ref_sample: list[Any]) -> list[Transition]:
    """Convert reference-transition samples to the port's Transition type.

    Same tensor contents; only the namedtuple class differs.
    """
    return [
        Transition(s.state, s.next_state, s.action, s.reward, s.is_terminal)
        for s in ref_sample
    ]


# ─── Tests: non-CL branch ───────────────────────────────────────────────────


@pytest.mark.parametrize("meta", [False, True])
def test_non_cl_branch_matches_reference(meta: bool) -> None:
    """Degenerate path (no teacher): port must match reference bit-exactly."""
    # Build two separate network instances — one for ref, one for port.
    ref_policy, ref_target, ref_second = _build_networks_ref(SEED)
    our_policy, our_target, our_second = _build_networks_ours(
        ref_policy, ref_target, ref_second
    )

    ref_opt1, ref_opt2, ref_sch1, ref_sch2 = _build_optimizers(ref_policy, ref_second)
    our_opt1, our_opt2, our_sch1, our_sch2 = _build_optimizers(our_policy, our_second)

    # Identical sample fed to both.
    ref_sample = _make_sample_ref(BATCH_SIZE, SEED)
    our_sample = _sample_as_port(ref_sample)

    # Keep dropout off so the two paths are deterministic (SO forward has
    # dropout in both when .train() — we use .eval() on SO for parity).
    ref_second.eval()
    our_second.eval()

    # ref path
    reference_dqn_update_step_cl(
        ref_sample,
        ref_policy, ref_target, ref_second if meta else None,
        ref_opt1, ref_opt2 if meta else None,
        ref_sch1, ref_sch2 if meta else None,
        meta=meta, alpha=45, cascade_iterations_1=CASCADE_OFF, cascade_iterations_2=CASCADE_OFF,
    )

    # port path
    mixing = LossMixingWeights(task=WEIGHT1, distillation=WEIGHT2, feature=WEIGHT3)
    sarl_cl_update_step(
        sample=our_sample,
        policy_net=our_policy, target_net=our_target,
        second_order_net=our_second if meta else None,
        teacher_first_net=None, teacher_second_net=None,
        optimizer=our_opt1, optimizer2=our_opt2 if meta else None,
        scheduler1=our_sch1, scheduler2=our_sch2 if meta else None,
        loss_weighter=None, loss_weighter_second=None,
        mixing=mixing,
        meta=meta, alpha=45,
        cascade_iterations_1=CASCADE_OFF, cascade_iterations_2=CASCADE_OFF,
        target_wager_fn=target_wager,
    )

    # Compare all parameters — weights must match after one update.
    for (k_ref, v_ref), (k_ours, v_ours) in zip(
        ref_policy.state_dict().items(), our_policy.state_dict().items(), strict=True
    ):
        assert k_ref == k_ours
        assert torch.allclose(v_ref, v_ours, atol=ATOL), (
            f"policy_net param {k_ref} drifted (meta={meta})"
        )
    if meta:
        for (k_ref, v_ref), (k_ours, v_ours) in zip(
            ref_second.state_dict().items(), our_second.state_dict().items(), strict=True
        ):
            assert k_ref == k_ours
            assert torch.allclose(v_ref, v_ours, atol=ATOL), (
                f"second_order_net param {k_ref} drifted"
            )


# ─── Tests: CL branch (3-term loss with shared teacher + weighters) ────────


def test_cl_branch_matches_reference() -> None:
    """CL path: shared DynamicLossWeighter + identical teachers → bit-exact."""
    # Build 3 networks + teachers (teachers = frozen clones of initial policy).
    ref_policy, ref_target, ref_second = _build_networks_ref(SEED)
    ref_teacher_first = RefQNetwork(IN_CHANNELS, NUM_ACTIONS)
    ref_teacher_first.load_state_dict(ref_policy.state_dict())
    ref_teacher_first.eval()
    for p in ref_teacher_first.parameters():
        p.requires_grad_(False)

    ref_teacher_second = RefSecondOrderNetwork(IN_CHANNELS)
    ref_teacher_second.load_state_dict(ref_second.state_dict())
    ref_teacher_second.eval()
    for p in ref_teacher_second.parameters():
        p.requires_grad_(False)

    our_policy, our_target, our_second = _build_networks_ours(
        ref_policy, ref_target, ref_second
    )
    our_teacher_first = SarlCLQNetwork(IN_CHANNELS, NUM_ACTIONS)
    _copy_state_dict(ref_teacher_first, our_teacher_first)
    our_teacher_first.eval()
    for p in our_teacher_first.parameters():
        p.requires_grad_(False)

    our_teacher_second = SarlCLSecondOrderNetwork(IN_CHANNELS)
    _copy_state_dict(ref_teacher_second, our_teacher_second)
    our_teacher_second.eval()
    for p in our_teacher_second.parameters():
        p.requires_grad_(False)

    # Shared DynamicLossWeighter instances — passed to BOTH update steps so
    # their historical_max state stays perfectly synced regardless of which
    # branch updates it first.
    shared_weighter = DynamicLossWeighter()
    shared_weighter_second = DynamicLossWeighter()

    ref_opt1, ref_opt2, ref_sch1, ref_sch2 = _build_optimizers(ref_policy, ref_second)
    our_opt1, our_opt2, our_sch1, our_sch2 = _build_optimizers(our_policy, our_second)

    ref_sample = _make_sample_ref(BATCH_SIZE, SEED)
    our_sample = _sample_as_port(ref_sample)

    # Put SO in eval to kill dropout variance.
    ref_second.eval()
    our_second.eval()

    # --- ref path ---
    reference_dqn_update_step_cl(
        ref_sample,
        ref_policy, ref_target, ref_second,
        ref_opt1, ref_opt2, ref_sch1, ref_sch2,
        meta=True, alpha=45,
        cascade_iterations_1=CASCADE_OFF, cascade_iterations_2=CASCADE_OFF,
        teacher_first_net=ref_teacher_first,
        teacher_second_net=ref_teacher_second,
        loss_weighter=shared_weighter,
        loss_weighter_second=shared_weighter_second,
    )

    # Snapshot weighter state post-ref so we can restore before the port run.
    saved_w_hist = dict(shared_weighter.historical_max)
    saved_w_hist_prev = dict(shared_weighter.historical_max_prev)
    saved_w_moving = dict(shared_weighter.moving_avgs)
    saved_w_steps = shared_weighter.steps
    saved_w2_hist = dict(shared_weighter_second.historical_max)
    saved_w2_hist_prev = dict(shared_weighter_second.historical_max_prev)
    saved_w2_moving = dict(shared_weighter_second.moving_avgs)
    saved_w2_steps = shared_weighter_second.steps

    # Reset the weighters to their pre-ref state for the port run (so both
    # sides see identical running-max at the divisor).
    shared_weighter.historical_max = {k: float("-inf") for k in shared_weighter.keys}
    shared_weighter.historical_max_prev = {k: float("-inf") for k in shared_weighter.keys}
    shared_weighter.moving_avgs = {k: 1.0 for k in shared_weighter.keys}
    shared_weighter.steps = 0
    shared_weighter_second.historical_max = {k: float("-inf") for k in shared_weighter_second.keys}
    shared_weighter_second.historical_max_prev = {
        k: float("-inf") for k in shared_weighter_second.keys
    }
    shared_weighter_second.moving_avgs = {k: 1.0 for k in shared_weighter_second.keys}
    shared_weighter_second.steps = 0

    # --- port path ---
    mixing = LossMixingWeights(task=WEIGHT1, distillation=WEIGHT2, feature=WEIGHT3)
    sarl_cl_update_step(
        sample=our_sample,
        policy_net=our_policy, target_net=our_target, second_order_net=our_second,
        teacher_first_net=our_teacher_first, teacher_second_net=our_teacher_second,
        optimizer=our_opt1, optimizer2=our_opt2,
        scheduler1=our_sch1, scheduler2=our_sch2,
        loss_weighter=shared_weighter,
        loss_weighter_second=shared_weighter_second,
        mixing=mixing,
        meta=True, alpha=45,
        cascade_iterations_1=CASCADE_OFF, cascade_iterations_2=CASCADE_OFF,
        target_wager_fn=target_wager,
    )

    # Weighter state after port run must match the ref run snapshot exactly.
    assert shared_weighter.historical_max == saved_w_hist
    assert shared_weighter_second.historical_max == saved_w2_hist
    assert shared_weighter.steps == saved_w_steps == 1
    assert shared_weighter_second.steps == saved_w2_steps == 1

    # Weights must match after one CL update.
    for (k_ref, v_ref), (k_ours, v_ours) in zip(
        ref_policy.state_dict().items(), our_policy.state_dict().items(), strict=True
    ):
        assert k_ref == k_ours
        assert torch.allclose(v_ref, v_ours, atol=ATOL), (
            f"policy_net param {k_ref} drifted on CL branch"
        )
    for (k_ref, v_ref), (k_ours, v_ours) in zip(
        ref_second.state_dict().items(), our_second.state_dict().items(), strict=True
    ):
        assert k_ref == k_ours
        assert torch.allclose(v_ref, v_ours, atol=ATOL), (
            f"second_order_net param {k_ref} drifted on CL branch"
        )
