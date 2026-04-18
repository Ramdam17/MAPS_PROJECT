"""Unit tests for :func:`maps.experiments.sarl_cl.trainer.sarl_cl_update_step`.

Covers the four branches induced by ``(cl_enabled, meta)``:

* vanilla FO only ``(cl_enabled=False, meta=False)``
* vanilla FO + SO ``(cl_enabled=False, meta=True)``
* CL FO only ``(cl_enabled=True,  meta=False)``
* CL FO + SO ``(cl_enabled=True,  meta=True)``

Plus:

* ``train=False`` — no weight update, no scheduler step.
* Component-loss decomposition is reported when (and only when) a teacher
  is present.
* Backward-order preservation — the paper's SO-before-FO sequence is
  implicit in the function body; we sanity-check that weights of BOTH
  networks move together on a meta+CL step.
* Scheduler advance — ``scheduler1.last_epoch`` increments when ``train=True``
  but not when ``train=False``.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from maps.experiments.sarl.data import SarlReplayBuffer, Transition, target_wager
from maps.experiments.sarl_cl.loss_weighting import DynamicLossWeighter
from maps.experiments.sarl_cl.model import SarlCLQNetwork, SarlCLSecondOrderNetwork
from maps.experiments.sarl_cl.trainer import (
    LossMixingWeights,
    SarlCLComponentLosses,
    SarlCLUpdateOutput,
    sarl_cl_update_step,
)

IN_CHANNELS = 4
NUM_ACTIONS = 6
BATCH_SIZE = 16
SEED = 2026


# ── Fixtures ───────────────────────────────────────────────────────────────


def _build_nets(
    seed: int = SEED,
) -> tuple[SarlCLQNetwork, SarlCLQNetwork, SarlCLSecondOrderNetwork]:
    """Build policy, target, second with identical RNG init."""
    torch.manual_seed(seed)
    policy = SarlCLQNetwork(IN_CHANNELS, NUM_ACTIONS)
    target = SarlCLQNetwork(IN_CHANNELS, NUM_ACTIONS)
    target.load_state_dict(policy.state_dict())
    second = SarlCLSecondOrderNetwork(IN_CHANNELS)
    return policy, target, second


def _build_teachers(
    policy_ref: SarlCLQNetwork,
    second_ref: SarlCLSecondOrderNetwork,
) -> tuple[SarlCLQNetwork, SarlCLSecondOrderNetwork]:
    """Build frozen teachers with slightly offset weights from ``policy_ref``.

    The teachers must have the *same architecture* (so weight_regularization
    is well-defined) but *different values* (so the anchor contributes a
    non-zero loss).
    """
    teacher_first = SarlCLQNetwork(IN_CHANNELS, NUM_ACTIONS)
    teacher_first.load_state_dict(policy_ref.state_dict())
    with torch.no_grad():
        for p in teacher_first.parameters():
            p.add_(0.1 * torch.randn_like(p))
        p.requires_grad_(False)
    for p in teacher_first.parameters():
        p.requires_grad_(False)
    teacher_first.eval()

    teacher_second = SarlCLSecondOrderNetwork(IN_CHANNELS)
    teacher_second.load_state_dict(second_ref.state_dict())
    with torch.no_grad():
        for p in teacher_second.parameters():
            p.add_(0.1 * torch.randn_like(p))
    for p in teacher_second.parameters():
        p.requires_grad_(False)
    teacher_second.eval()

    return teacher_first, teacher_second


def _build_optimizers(
    policy: torch.nn.Module, second: torch.nn.Module
) -> tuple[optim.Optimizer, optim.Optimizer, Any, Any]:
    opt1 = optim.Adam(policy.parameters(), lr=3e-4, eps=0.01)
    opt2 = optim.Adam(second.parameters(), lr=5e-5, eps=0.01)
    sch1 = StepLR(opt1, step_size=1000, gamma=0.999)
    sch2 = StepLR(opt2, step_size=1000, gamma=0.999)
    return opt1, opt2, sch1, sch2


def _build_sample(n: int = BATCH_SIZE, seed: int = 0) -> list[Transition]:
    """Synthetic batch of ``Transition`` tuples, matching paper shapes."""
    rng = random.Random(seed)
    buf = SarlReplayBuffer(buffer_size=n)
    for _ in range(n):
        state = torch.rand(1, IN_CHANNELS, 10, 10)
        next_state = torch.rand(1, IN_CHANNELS, 10, 10)
        action = torch.tensor([[rng.randrange(NUM_ACTIONS)]], dtype=torch.int64)
        reward = torch.tensor([[rng.uniform(-1.0, 1.0)]], dtype=torch.float32)
        done = torch.tensor([[1 if rng.random() < 0.1 else 0]], dtype=torch.int64)
        buf.add(state, next_state, action, reward, done)
    return list(buf.buffer)


def _snapshot(net: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in net.named_parameters()}


def _any_param_moved(before: dict[str, torch.Tensor], net: torch.nn.Module) -> bool:
    for name, p in net.named_parameters():
        if not torch.allclose(before[name], p.detach(), atol=1e-10):
            return True
    return False


# ── vanilla FO only (no meta, no teacher) ─────────────────────────────────


def test_vanilla_fo_only_updates_policy_only() -> None:
    policy, target, second = _build_nets()
    opt1, _, sch1, _ = _build_optimizers(policy, second)
    sample = _build_sample()

    before_policy = _snapshot(policy)
    before_second = _snapshot(second)

    out = sarl_cl_update_step(
        sample=sample,
        policy_net=policy,
        target_net=target,
        second_order_net=None,
        teacher_first_net=None,
        teacher_second_net=None,
        optimizer=opt1,
        optimizer2=None,
        scheduler1=sch1,
        scheduler2=None,
        loss_weighter=None,
        loss_weighter_second=None,
        mixing=LossMixingWeights(),
        meta=False,
        alpha=1.0,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        target_wager_fn=target_wager,
        train=True,
    )

    assert isinstance(out, SarlCLUpdateOutput)
    assert out.loss_second is None
    assert out.components_first is None
    assert out.components_second is None
    assert torch.isfinite(out.loss).all()

    assert _any_param_moved(before_policy, policy)
    # SO net was never touched.
    assert not _any_param_moved(before_second, second)


# ── vanilla meta (no teacher, both nets train) ─────────────────────────────


def test_vanilla_meta_updates_both_networks() -> None:
    policy, target, second = _build_nets()
    opt1, opt2, sch1, sch2 = _build_optimizers(policy, second)
    sample = _build_sample()

    before_policy = _snapshot(policy)
    before_second = _snapshot(second)

    out = sarl_cl_update_step(
        sample=sample,
        policy_net=policy,
        target_net=target,
        second_order_net=second,
        teacher_first_net=None,
        teacher_second_net=None,
        optimizer=opt1,
        optimizer2=opt2,
        scheduler1=sch1,
        scheduler2=sch2,
        loss_weighter=None,
        loss_weighter_second=None,
        mixing=LossMixingWeights(),
        meta=True,
        alpha=1.0,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        target_wager_fn=target_wager,
        train=True,
    )

    assert out.loss_second is not None
    assert torch.isfinite(out.loss_second).all()
    # Shapes: wager is (B, 2), q_values (B, 1) after gather.
    assert out.wager_logits.shape == (BATCH_SIZE, 2)
    assert out.q_values.shape == (BATCH_SIZE, 1)

    assert _any_param_moved(before_policy, policy)
    assert _any_param_moved(before_second, second)
    # No teacher → no component breakdown.
    assert out.components_first is None
    assert out.components_second is None


# ── CL branch (FO only) ────────────────────────────────────────────────────


def test_cl_fo_only_populates_component_losses() -> None:
    policy, target, second = _build_nets()
    teacher_first, _ = _build_teachers(policy, second)

    opt1, _, sch1, _ = _build_optimizers(policy, second)
    weighter = DynamicLossWeighter()
    sample = _build_sample()

    before_policy = _snapshot(policy)

    out = sarl_cl_update_step(
        sample=sample,
        policy_net=policy,
        target_net=target,
        second_order_net=None,
        teacher_first_net=teacher_first,
        teacher_second_net=None,
        optimizer=opt1,
        optimizer2=None,
        scheduler1=sch1,
        scheduler2=None,
        loss_weighter=weighter,
        loss_weighter_second=None,
        mixing=LossMixingWeights(),
        meta=False,
        alpha=1.0,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        target_wager_fn=target_wager,
        train=True,
    )

    assert out.components_first is not None
    assert out.components_second is None
    c = out.components_first
    assert isinstance(c, SarlCLComponentLosses)
    # All three terms are finite, non-negative scalars.
    for v in (c.task, c.distillation, c.feature):
        assert isinstance(v, float)
        assert v >= 0.0
        assert torch.isfinite(torch.tensor(v))
    assert _any_param_moved(before_policy, policy)

    # The weighter now has non-default historical_max for all three keys.
    for key in ("task", "distillation", "feature"):
        assert weighter.historical_max[key] > 0.0


# ── CL branch (FO + SO) ────────────────────────────────────────────────────


def test_cl_meta_populates_second_order_components() -> None:
    policy, target, second = _build_nets()
    teacher_first, teacher_second = _build_teachers(policy, second)

    opt1, opt2, sch1, sch2 = _build_optimizers(policy, second)
    w1 = DynamicLossWeighter()
    w2 = DynamicLossWeighter()
    sample = _build_sample()

    before_policy = _snapshot(policy)
    before_second = _snapshot(second)

    out = sarl_cl_update_step(
        sample=sample,
        policy_net=policy,
        target_net=target,
        second_order_net=second,
        teacher_first_net=teacher_first,
        teacher_second_net=teacher_second,
        optimizer=opt1,
        optimizer2=opt2,
        scheduler1=sch1,
        scheduler2=sch2,
        loss_weighter=w1,
        loss_weighter_second=w2,
        mixing=LossMixingWeights(),
        meta=True,
        alpha=1.0,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        target_wager_fn=target_wager,
        train=True,
    )

    assert out.components_first is not None
    assert out.components_second is not None
    for c in (out.components_first, out.components_second):
        for v in (c.task, c.distillation, c.feature):
            assert v >= 0.0
            assert torch.isfinite(torch.tensor(v))

    # Both nets moved (the paper's SO-before-FO order ensures this).
    assert _any_param_moved(before_policy, policy)
    assert _any_param_moved(before_second, second)


# ── train=False — forward only ─────────────────────────────────────────────


def test_train_false_leaves_weights_and_scheduler_untouched() -> None:
    policy, target, second = _build_nets()
    teacher_first, teacher_second = _build_teachers(policy, second)

    opt1, opt2, sch1, sch2 = _build_optimizers(policy, second)
    sch1_last = sch1.last_epoch
    sch2_last = sch2.last_epoch
    w1 = DynamicLossWeighter()
    w2 = DynamicLossWeighter()
    sample = _build_sample()

    before_policy = _snapshot(policy)
    before_second = _snapshot(second)

    out = sarl_cl_update_step(
        sample=sample,
        policy_net=policy,
        target_net=target,
        second_order_net=second,
        teacher_first_net=teacher_first,
        teacher_second_net=teacher_second,
        optimizer=opt1,
        optimizer2=opt2,
        scheduler1=sch1,
        scheduler2=sch2,
        loss_weighter=w1,
        loss_weighter_second=w2,
        mixing=LossMixingWeights(),
        meta=True,
        alpha=1.0,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        target_wager_fn=target_wager,
        train=False,
    )

    # Loss computed forward-only → finite.
    assert torch.isfinite(out.loss).all()
    assert torch.isfinite(out.loss_second).all()
    # No weight update, no scheduler advance.
    assert not _any_param_moved(before_policy, policy)
    assert not _any_param_moved(before_second, second)
    assert sch1.last_epoch == sch1_last
    assert sch2.last_epoch == sch2_last


# ── scheduler advances when train=True ─────────────────────────────────────


def test_scheduler_advances_only_on_training_updates() -> None:
    policy, target, second = _build_nets()
    opt1, opt2, sch1, sch2 = _build_optimizers(policy, second)
    sample = _build_sample()

    sch1_before = sch1.last_epoch
    sch2_before = sch2.last_epoch

    sarl_cl_update_step(
        sample=sample,
        policy_net=policy,
        target_net=target,
        second_order_net=second,
        teacher_first_net=None,
        teacher_second_net=None,
        optimizer=opt1,
        optimizer2=opt2,
        scheduler1=sch1,
        scheduler2=sch2,
        loss_weighter=None,
        loss_weighter_second=None,
        mixing=LossMixingWeights(),
        meta=True,
        alpha=1.0,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        target_wager_fn=target_wager,
        train=True,
    )

    assert sch1.last_epoch == sch1_before + 1
    assert sch2.last_epoch == sch2_before + 1


# ── cascade > 1 path ───────────────────────────────────────────────────────


def test_cascade_iterations_path_is_reachable() -> None:
    """Setting 6 (cascade_1=cascade_2=50) must run without errors."""
    policy, target, second = _build_nets()
    opt1, opt2, sch1, sch2 = _build_optimizers(policy, second)
    sample = _build_sample()

    out = sarl_cl_update_step(
        sample=sample,
        policy_net=policy,
        target_net=target,
        second_order_net=second,
        teacher_first_net=None,
        teacher_second_net=None,
        optimizer=opt1,
        optimizer2=opt2,
        scheduler1=sch1,
        scheduler2=sch2,
        loss_weighter=None,
        loss_weighter_second=None,
        mixing=LossMixingWeights(),
        meta=True,
        alpha=1.0,
        cascade_iterations_1=50,
        cascade_iterations_2=50,
        target_wager_fn=target_wager,
        train=True,
    )

    assert torch.isfinite(out.loss).all()
    assert torch.isfinite(out.loss_second).all()


# ── Mixing weights affect the total loss ───────────────────────────────────


def test_cl_mixing_weights_zero_distillation_removes_that_component() -> None:
    """Setting a mixing weight to zero must suppress that loss term.

    With distillation weight=0, the total loss should equal
    ``task * weighted_task + feature * weighted_feature``.
    """
    policy, target, second = _build_nets()
    teacher_first, _ = _build_teachers(policy, second)
    opt1, _, sch1, _ = _build_optimizers(policy, second)
    w1 = DynamicLossWeighter()
    sample = _build_sample()

    out = sarl_cl_update_step(
        sample=sample,
        policy_net=policy,
        target_net=target,
        second_order_net=None,
        teacher_first_net=teacher_first,
        teacher_second_net=None,
        optimizer=opt1,
        optimizer2=None,
        scheduler1=sch1,
        scheduler2=None,
        loss_weighter=w1,
        loss_weighter_second=None,
        mixing=LossMixingWeights(task=1.0, distillation=0.0, feature=1.0),
        meta=False,
        alpha=1.0,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        target_wager_fn=target_wager,
        train=False,
    )

    c = out.components_first
    # Reconstructed loss with distillation=0 must match out.loss.
    reconstructed = c.task + c.feature
    assert out.loss.item() == pytest.approx(reconstructed, rel=1e-5)


# ── CL precondition: meta=True requires teacher_second ─────────────────────


def test_cl_meta_without_teacher_second_raises() -> None:
    policy, target, second = _build_nets()
    teacher_first, _ = _build_teachers(policy, second)
    opt1, opt2, sch1, sch2 = _build_optimizers(policy, second)
    w1 = DynamicLossWeighter()
    w2 = DynamicLossWeighter()
    sample = _build_sample()

    with pytest.raises(AssertionError, match="teacher_second_net"):
        sarl_cl_update_step(
            sample=sample,
            policy_net=policy,
            target_net=target,
            second_order_net=second,
            teacher_first_net=teacher_first,
            teacher_second_net=None,  # ← missing
            optimizer=opt1,
            optimizer2=opt2,
            scheduler1=sch1,
            scheduler2=sch2,
            loss_weighter=w1,
            loss_weighter_second=w2,
            mixing=LossMixingWeights(),
            meta=True,
            alpha=1.0,
            cascade_iterations_1=1,
            cascade_iterations_2=1,
            target_wager_fn=target_wager,
            train=True,
        )


def test_meta_true_without_second_order_net_raises() -> None:
    policy, target, _ = _build_nets()
    opt1, _, sch1, _ = _build_optimizers(policy, torch.nn.Linear(1, 1))
    sample = _build_sample()

    with pytest.raises(AssertionError, match="second_order_net"):
        sarl_cl_update_step(
            sample=sample,
            policy_net=policy,
            target_net=target,
            second_order_net=None,
            teacher_first_net=None,
            teacher_second_net=None,
            optimizer=opt1,
            optimizer2=None,
            scheduler1=sch1,
            scheduler2=None,
            loss_weighter=None,
            loss_weighter_second=None,
            mixing=LossMixingWeights(),
            meta=True,
            alpha=1.0,
            cascade_iterations_1=1,
            cascade_iterations_2=1,
            target_wager_fn=target_wager,
            train=True,
        )
