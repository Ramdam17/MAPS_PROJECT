"""Tier 3 parity tests — SARL DQN update-step equivalence.

Asserts that ``maps.experiments.sarl.trainer.sarl_update_step`` reproduces
``_reference_sarl.reference_dqn_update_step`` (verbatim-structure transcription
of the paper's ``train()``) **bit-exactly** on one full update, starting from
identical weights and fed an identical sample.

Why Tier 3 matters
------------------
Tier 1 proved forward-pass parity (same logits given same weights + input).
Tier 2 proved replay-buffer parity (same transitions given same seed).
Tier 3 closes the loop: the DQN *update rule* — losses, backward pass order,
optimizer step, scheduler — produces the same weight update as the paper.

If Tier 3 is green, then a full training run under identical seeds would
produce identical trajectories; any later deviation in reproduction (Sprint 07)
would come from something outside this function (e.g. environment stepping,
exploration policy) rather than from the learning rule itself.

Test strategy
-------------
1. **Identical init**: build both sets of networks with the same seed, copy
   weights ref → ours to eliminate any residual construction-order drift.
2. **Identical inputs**: the same ``sample`` (list of transitions), same RNG
   state just before each update (so dropout masks match).
3. **Per-parameter comparison**: after one update call, compare
   ``named_parameters()`` and ``.grad`` tensor-by-tensor at ``atol=1e-6``.
4. **Parametrize over both branches** (``meta=False``, ``meta=True``) and
   both cascade regimes (``iterations=1`` "cascade off", ``iterations=50``
   "cascade on") so settings 1-6 from the paper are all covered.

Notes on cross-gradient preservation
------------------------------------
The meta branch calls ``loss_second.backward(retain_graph=True)`` BEFORE
``loss.backward()``. ``loss_second`` flows through ``comparison_1`` which is
a function of ``policy_net.fc_hidden``, so ``policy_net.grad`` receives
contributions from BOTH losses. Tier 3 tests this implicitly: if the order
flipped, post-step weights in ``policy_net`` would diverge and the
``test_policy_net_weights_match`` assertion would catch it.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from maps.experiments.sarl.data import SarlReplayBuffer, Transition
from maps.experiments.sarl.model import SarlQNetwork, SarlSecondOrderNetwork
from maps.experiments.sarl.trainer import sarl_update_step
from tests.parity.sarl._reference_sarl import (
    MIN_SQUARED_GRAD,
    reference_dqn_update_step,
    scheduler_step,
    step_size1,
    step_size2,
    target_wager,
)
from tests.parity.sarl._reference_sarl import (
    QNetwork as RefQNetwork,
)
from tests.parity.sarl._reference_sarl import (
    SecondOrderNetwork as RefSecondOrderNetwork,
)
from tests.parity.sarl._reference_sarl import (
    transition as ref_transition,
)

# ─── Configuration ───────────────────────────────────────────────────────────

IN_CHANNELS = 4  # MinAtar Space Invaders has 4 channels
NUM_ACTIONS = 6
BATCH_SIZE = 32  # smaller than paper's 128 to keep tests fast
SEED = 2026
ATOL = 1e-6

# Paper uses cascade_iterations=1 when cascade is OFF, =50 when ON.
CASCADE_OFF = 1
CASCADE_ON = 50


# ─── Fixtures & helpers ──────────────────────────────────────────────────────


#: Keys allowed in dst but not in src (paper-faithful parameters that the
#: frozen student reference does not carry). D.7 adds `b_recon` to
#: SarlQNetwork per paper eq.12; it is zero-initialised so numerical parity
#: at init is preserved even though the key set diverges.
_DST_ONLY_ALLOWED_KEYS: frozenset[str] = frozenset({"b_recon"})


def _copy_state_dict(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """Copy every parameter from src into dst by name.

    Uses ``strict=False`` because ref vs ours may differ in buffer metadata
    (e.g. dropout reports ``training`` state); PARAMETER names must match
    EXCEPT for paper-faithful extras on our side (see
    ``_DST_ONLY_ALLOWED_KEYS``), which we tolerate and leave at their
    zero-initialised value so numerical parity at init holds.
    """
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    src_keys = set(src_state.keys())
    dst_keys = set(dst_state.keys())
    # Keys missing on the src side would mean we lost something critical.
    missing_in_dst = src_keys - dst_keys
    assert not missing_in_dst, (
        f"dst is missing keys present in src: {missing_in_dst}"
    )
    # Keys extra on the dst side are only allowed if explicitly whitelisted.
    extras = dst_keys - src_keys - _DST_ONLY_ALLOWED_KEYS
    assert not extras, (
        f"dst has unexpected extra keys not whitelisted: {extras}"
    )
    dst.load_state_dict(src_state, strict=False)


def _build_networks_ref(seed: int) -> tuple[RefQNetwork, RefQNetwork, RefSecondOrderNetwork]:
    """Build reference (paper) networks with a deterministic init."""
    torch.manual_seed(seed)
    policy = RefQNetwork(IN_CHANNELS, NUM_ACTIONS)
    target = RefQNetwork(IN_CHANNELS, NUM_ACTIONS)
    target.load_state_dict(policy.state_dict())  # target = policy at init (paper convention)
    second = RefSecondOrderNetwork(IN_CHANNELS)
    return policy, target, second


def _build_networks_ours(
    ref_policy: RefQNetwork,
    ref_target: RefQNetwork,
    ref_second: RefSecondOrderNetwork,
) -> tuple[SarlQNetwork, SarlQNetwork, SarlSecondOrderNetwork]:
    """Build our networks and sync weights from the reference."""
    policy = SarlQNetwork(IN_CHANNELS, NUM_ACTIONS)
    target = SarlQNetwork(IN_CHANNELS, NUM_ACTIONS)
    second = SarlSecondOrderNetwork(IN_CHANNELS)
    _copy_state_dict(ref_policy, policy)
    _copy_state_dict(ref_target, target)
    _copy_state_dict(ref_second, second)
    return policy, target, second


def _build_sample(n: int = BATCH_SIZE, seed: int = 0) -> list[Transition]:
    """Produce a deterministic batch of synthetic transitions.

    The batch is NOT drawn from a buffer — Tier 2 already certified that
    path. Here we build transitions directly, matching the shapes expected
    by ``train()``: state / next_state as (1, C, 10, 10), action as (1, 1)
    int64, reward as (1, 1) float, is_terminal as (1, 1) int.
    """
    rng = random.Random(seed)
    buf = SarlReplayBuffer(buffer_size=n)
    for _ in range(n):
        state = torch.rand(1, IN_CHANNELS, 10, 10)
        next_state = torch.rand(1, IN_CHANNELS, 10, 10)
        action = torch.tensor([[rng.randrange(NUM_ACTIONS)]], dtype=torch.int64)
        reward = torch.tensor([[rng.uniform(-1.0, 1.0)]], dtype=torch.float32)
        # ~1 in 10 terminal, matching typical MinAtar episode rhythm.
        done = torch.tensor([[1 if rng.random() < 0.1 else 0]], dtype=torch.int64)
        buf.add(state, next_state, action, reward, done)
    return list(buf.buffer)


def _build_optimizers(
    policy: torch.nn.Module, second: torch.nn.Module
) -> tuple[optim.Optimizer, optim.Optimizer, Any, Any]:
    """Build Adam optimizers + StepLR schedulers per the paper."""
    opt1 = optim.Adam(policy.parameters(), lr=step_size1, eps=MIN_SQUARED_GRAD)
    opt2 = optim.Adam(second.parameters(), lr=step_size2, eps=MIN_SQUARED_GRAD)
    sch1 = StepLR(opt1, step_size=1000, gamma=scheduler_step)
    sch2 = StepLR(opt2, step_size=1000, gamma=scheduler_step)
    return opt1, opt2, sch1, sch2


def _assert_named_params_match(
    ref: torch.nn.Module, ours: torch.nn.Module, *, tag: str, atol: float = ATOL
) -> None:
    """Compare ``named_parameters()`` element-wise at ``atol``."""
    ref_params = dict(ref.named_parameters())
    our_params = dict(ours.named_parameters())
    assert ref_params.keys() == our_params.keys(), (
        f"[{tag}] param names differ: ref={ref_params.keys()} ours={our_params.keys()}"
    )
    for name, ref_p in ref_params.items():
        our_p = our_params[name]
        assert ref_p.shape == our_p.shape, f"[{tag}] {name} shape differs"
        max_abs = (ref_p - our_p).abs().max().item()
        assert torch.allclose(ref_p, our_p, atol=atol), (
            f"[{tag}] {name} post-step weights differ: max |Δ| = {max_abs:.3e}"
        )


def _assert_named_grads_match(
    ref: torch.nn.Module, ours: torch.nn.Module, *, tag: str, atol: float = ATOL
) -> None:
    """Compare ``.grad`` of each named parameter at ``atol``.

    ``optimizer.step()`` does NOT zero gradients, so post-update grads are
    still accessible. Missing grads (``None`` on both sides) are allowed as
    long as both sides agree.
    """
    ref_params = dict(ref.named_parameters())
    our_params = dict(ours.named_parameters())
    for name, ref_p in ref_params.items():
        our_p = our_params[name]
        # Both None → both skipped (e.g. if grad was never populated).
        if ref_p.grad is None and our_p.grad is None:
            continue
        assert ref_p.grad is not None, f"[{tag}] {name}: ref has no grad, ours does"
        assert our_p.grad is not None, f"[{tag}] {name}: ours has no grad, ref does"
        max_abs = (ref_p.grad - our_p.grad).abs().max().item()
        assert torch.allclose(ref_p.grad, our_p.grad, atol=atol), (
            f"[{tag}] {name} gradients differ: max |Δ| = {max_abs:.3e}"
        )


# ─── Sanity: the reference itself is self-consistent ────────────────────────


def test_reference_update_runs_without_error() -> None:
    """Smoke test: the de-globalized reference update step runs end-to-end."""
    ref_policy, ref_target, ref_second = _build_networks_ref(SEED)
    sample = _build_sample()
    opt1, opt2, sch1, sch2 = _build_optimizers(ref_policy, ref_second)

    torch.manual_seed(SEED)
    loss, loss_second, _ = reference_dqn_update_step(
        sample=sample,
        policy_net=ref_policy,
        target_net=ref_target,
        second_order_net=ref_second,
        optimizer=opt1,
        optimizer2=opt2,
        scheduler1=sch1,
        scheduler2=sch2,
        meta=True,
        alpha=1,
        cascade_iterations_1=CASCADE_OFF,
        cascade_iterations_2=CASCADE_OFF,
    )
    assert torch.isfinite(loss).all()
    assert torch.isfinite(loss_second).all()


# ─── Parity: non-meta branch (paper settings 1/2/3) ─────────────────────────


@pytest.mark.parametrize("cascade_iters", [CASCADE_OFF, CASCADE_ON])
def test_non_meta_update_matches_reference(cascade_iters: int) -> None:
    """``meta=False`` branch: single-loss DQN update is bit-exact."""
    # Reference side.
    ref_policy, ref_target, ref_second = _build_networks_ref(SEED)
    ref_opt1, ref_opt2, ref_sch1, ref_sch2 = _build_optimizers(ref_policy, ref_second)

    # Ours side — mirror weights from ref so we start in sync.
    our_policy, our_target, our_second = _build_networks_ours(ref_policy, ref_target, ref_second)
    our_opt1, our_opt2, our_sch1, our_sch2 = _build_optimizers(our_policy, our_second)

    sample = _build_sample()

    # Both updates see the same dropout mask (no dropout in non-meta path,
    # but we reseed anyway as a discipline).
    torch.manual_seed(SEED)
    ref_loss, ref_loss_second, _ = reference_dqn_update_step(
        sample=sample,
        policy_net=ref_policy,
        target_net=ref_target,
        second_order_net=ref_second,
        optimizer=ref_opt1,
        optimizer2=ref_opt2,
        scheduler1=ref_sch1,
        scheduler2=ref_sch2,
        meta=False,
        alpha=1,
        cascade_iterations_1=cascade_iters,
        cascade_iterations_2=CASCADE_OFF,
    )
    assert ref_loss_second is None, "non-meta path should not return loss_second"

    torch.manual_seed(SEED)
    our_out = sarl_update_step(
        sample=sample,
        policy_net=our_policy,
        target_net=our_target,
        second_order_net=our_second,
        optimizer=our_opt1,
        optimizer2=our_opt2,
        scheduler1=our_sch1,
        scheduler2=our_sch2,
        meta=False,
        alpha=1,
        cascade_iterations_1=cascade_iters,
        cascade_iterations_2=CASCADE_OFF,
        target_wager_fn=target_wager,
    )

    # Losses scalar-match.
    assert torch.allclose(ref_loss, our_out.loss, atol=ATOL), (
        f"loss differs: ref={ref_loss.item():.6g} ours={our_out.loss.item():.6g}"
    )

    # Gradient + post-step weight parity on the first-order network.
    _assert_named_grads_match(ref_policy, our_policy, tag="policy")
    _assert_named_params_match(ref_policy, our_policy, tag="policy")

    # Target net MUST be untouched by the update.
    _assert_named_params_match(ref_target, our_target, tag="target-unchanged")


# ─── Parity: meta branch (paper settings 4/5/6) ─────────────────────────────


@pytest.mark.parametrize(
    "cascade_1,cascade_2",
    [
        (CASCADE_OFF, CASCADE_OFF),  # setting 4 (meta on, cascade off)
        (CASCADE_ON, CASCADE_OFF),  # setting 5 (meta on, cascade on FO only)
        (CASCADE_OFF, CASCADE_ON),  # setting 5 alt
        (CASCADE_ON, CASCADE_ON),  # setting 6 (meta on, cascade on both)
    ],
    ids=["setting-4-casc-off", "setting-5a", "setting-5b", "setting-6-casc-on"],
)
def test_meta_update_matches_reference(cascade_1: int, cascade_2: int) -> None:
    """``meta=True`` branch: two-loss cross-gradient update is bit-exact."""
    ref_policy, ref_target, ref_second = _build_networks_ref(SEED)
    ref_opt1, ref_opt2, ref_sch1, ref_sch2 = _build_optimizers(ref_policy, ref_second)

    our_policy, our_target, our_second = _build_networks_ours(ref_policy, ref_target, ref_second)
    our_opt1, our_opt2, our_sch1, our_sch2 = _build_optimizers(our_policy, our_second)

    sample = _build_sample()

    # Re-seed immediately before each update so dropout in SecondOrderNetwork
    # samples the same mask in both runs.
    torch.manual_seed(SEED)
    ref_loss, ref_loss_second, _ = reference_dqn_update_step(
        sample=sample,
        policy_net=ref_policy,
        target_net=ref_target,
        second_order_net=ref_second,
        optimizer=ref_opt1,
        optimizer2=ref_opt2,
        scheduler1=ref_sch1,
        scheduler2=ref_sch2,
        meta=True,
        alpha=1,
        cascade_iterations_1=cascade_1,
        cascade_iterations_2=cascade_2,
    )

    torch.manual_seed(SEED)
    our_out = sarl_update_step(
        sample=sample,
        policy_net=our_policy,
        target_net=our_target,
        second_order_net=our_second,
        optimizer=our_opt1,
        optimizer2=our_opt2,
        scheduler1=our_sch1,
        scheduler2=our_sch2,
        meta=True,
        alpha=1,
        cascade_iterations_1=cascade_1,
        cascade_iterations_2=cascade_2,
        target_wager_fn=target_wager,
    )

    # Both losses must match.
    assert torch.allclose(ref_loss, our_out.loss, atol=ATOL), (
        f"[casc={cascade_1},{cascade_2}] loss differs: "
        f"ref={ref_loss.item():.6g} ours={our_out.loss.item():.6g}"
    )
    assert torch.allclose(ref_loss_second, our_out.loss_second, atol=ATOL), (
        f"[casc={cascade_1},{cascade_2}] loss_second differs: "
        f"ref={ref_loss_second.item():.6g} ours={our_out.loss_second.item():.6g}"
    )

    # Policy_net receives grads from BOTH losses (cross-gradient). Matching
    # here is the hard check: if statement order flipped, policy grads diverge.
    _assert_named_grads_match(ref_policy, our_policy, tag="policy-meta")
    _assert_named_grads_match(ref_second, our_second, tag="second-order")

    # Post-step weights match on both networks.
    _assert_named_params_match(ref_policy, our_policy, tag="policy-meta")
    _assert_named_params_match(ref_second, our_second, tag="second-order")

    # Target net untouched.
    _assert_named_params_match(ref_target, our_target, tag="target-unchanged")


# ─── Guardrails for statement-order regressions ─────────────────────────────


def test_meta_update_changes_both_networks() -> None:
    """Sanity: after a meta update, BOTH policy and second must move.

    Prevents a dead-code regression where meta branch silently skips a step.
    """
    ref_policy_before, _, ref_second_before = _build_networks_ref(SEED)

    our_policy, our_target, our_second = _build_networks_ours(
        ref_policy_before, _build_networks_ref(SEED)[1], ref_second_before
    )
    opt1, opt2, sch1, sch2 = _build_optimizers(our_policy, our_second)

    sample = _build_sample()
    torch.manual_seed(SEED)
    sarl_update_step(
        sample=sample,
        policy_net=our_policy,
        target_net=our_target,
        second_order_net=our_second,
        optimizer=opt1,
        optimizer2=opt2,
        scheduler1=sch1,
        scheduler2=sch2,
        meta=True,
        alpha=1,
        cascade_iterations_1=CASCADE_OFF,
        cascade_iterations_2=CASCADE_OFF,
        target_wager_fn=target_wager,
    )

    # Compare to the pre-update snapshot. Some parameter must have moved.
    # Skip dst-only keys (paper-faithful extras like `b_recon`) that the
    # reference snapshot does not carry — comparing them would KeyError.
    ref_policy_keys = set(ref_policy_before.state_dict().keys())
    ref_second_keys = set(ref_second_before.state_dict().keys())
    policy_moved = any(
        not torch.equal(ref_policy_before.state_dict()[k], our_policy.state_dict()[k])
        for k in our_policy.state_dict()
        if k in ref_policy_keys
    )
    second_moved = any(
        not torch.equal(ref_second_before.state_dict()[k], our_second.state_dict()[k])
        for k in our_second.state_dict()
        if k in ref_second_keys
    )
    assert policy_moved, "policy_net parameters did not change after update"
    assert second_moved, "second_order_net parameters did not change after update"


def test_transition_accepted_by_reference() -> None:
    """Guardrail: passing our ``Transition`` into the reference function works.

    The reference does ``transition(*zip(*sample))`` — it repackages with its
    own namedtuple class. Field ORDER matters, not the class name. If we
    reordered Transition fields, the reference would silently misinterpret
    them. This catches that.
    """
    assert Transition._fields == ref_transition._fields
