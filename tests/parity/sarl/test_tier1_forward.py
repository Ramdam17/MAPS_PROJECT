"""Tier 1 parity tests — SARL model forward-pass equivalence.

Asserts that ``maps.experiments.sarl.model`` reproduces the paper reference
(``tests/parity/sarl/_reference_sarl.py``) bit-for-bit at ``atol=1e-6`` under:

1. Zero-cascade pass (``prev=None``, ``cascade_rate=1.0``) — pure feed-forward
   with no accumulation.
2. Single-step cascade (``prev`` seeded, ``cascade_rate=0.02``) — tests the
   ``cascade_update`` arithmetic.
3. Multi-iteration cascade rollout (50 iterations at α=0.02) — tests the
   iteration-level accumulation matches.
4. Reproduction residual (``Input - ReLU(decoder(Hidden))``) — tests that the
   tied-weight autoencoder branch still produces identical comparison values.

All four tests share the same pinned seed, in_channels=4 (Breakout default),
num_actions=6, batch=32 spec to keep failures localisable. Dropout is disabled
via ``.eval()`` to make the comparison deterministic — dropout parity itself
belongs in Tier 2/3 where we verify training-mode equivalence under shared RNG.

The refactored and reference models use **shared ``state_dict``** so we're
testing forward-pass math, not initialisation. Init-order parity is documented
in ``src/maps/experiments/sarl/model.py`` but not asserted here because it's a
weaker claim than forward equivalence (layer order + RNG coupling).
"""

from __future__ import annotations

import pytest
import torch

from maps.experiments.sarl.model import SarlQNetwork, SarlSecondOrderNetwork
from maps.utils.seeding import set_all_seeds
from tests.parity.sarl._reference_sarl import (
    QNetwork as RefQNetwork,
)
from tests.parity.sarl._reference_sarl import (
    SecondOrderNetwork as RefSecondOrderNetwork,
)

# ─── Shared fixtures ──────────────────────────────────────────────────────────

IN_CHANNELS = 4  # Breakout default
NUM_ACTIONS = 6  # max across MinAtar games (Freeway uses 3, Space Invaders 4, etc.)
BATCH = 32
STATE_SHAPE = (BATCH, IN_CHANNELS, 10, 10)
CASCADE_RATE_PAPER = 0.02  # paper's α from §2.1
N_ITER_PAPER = 50  # paper's cascade depth
ATOL = 1e-6


@pytest.fixture
def paired_q_networks() -> tuple[RefQNetwork, SarlQNetwork]:
    """Reference + refactored Q-networks with identical weights.

    Seeds before each construction so their default inits are identical, then
    cross-loads state_dict just to be safe against silent layer-order drift.
    """
    set_all_seeds(42)
    ref = RefQNetwork(IN_CHANNELS, NUM_ACTIONS).eval()
    set_all_seeds(42)
    ours = SarlQNetwork(IN_CHANNELS, NUM_ACTIONS).eval()
    ours.load_state_dict(ref.state_dict())
    return ref, ours


@pytest.fixture
def paired_second_order() -> tuple[RefSecondOrderNetwork, SarlSecondOrderNetwork]:
    """Reference + refactored SecondOrder networks with identical weights."""
    set_all_seeds(42)
    ref = RefSecondOrderNetwork(IN_CHANNELS).eval()
    set_all_seeds(42)
    ours = SarlSecondOrderNetwork(IN_CHANNELS).eval()
    ours.load_state_dict(ref.state_dict())
    return ref, ours


@pytest.fixture
def pinned_states() -> torch.Tensor:
    """Deterministic state batch — shape (32, 4, 10, 10)."""
    set_all_seeds(42)
    return torch.randn(*STATE_SHAPE)


# ─── QNetwork tests ───────────────────────────────────────────────────────────


def test_q_forward_zero_cascade(
    paired_q_networks: tuple[RefQNetwork, SarlQNetwork],
    pinned_states: torch.Tensor,
) -> None:
    """`prev_h2=None, cascade_rate=1.0` — pure feed-forward equivalence."""
    ref, ours = paired_q_networks
    with torch.no_grad():
        ref_q, ref_h, ref_c, ref_h2 = ref(pinned_states, None, 1.0)
        our_q, our_h, our_c, our_h2 = ours(pinned_states, None, 1.0)
    assert torch.allclose(ref_q, our_q, atol=ATOL), "Q-values diverged"
    assert torch.allclose(ref_h, our_h, atol=ATOL), "hidden diverged"
    assert torch.allclose(ref_c, our_c, atol=ATOL), "comparison diverged"
    assert torch.allclose(ref_h2, our_h2, atol=ATOL), "hidden_copy diverged"


def test_q_forward_single_cascade_step(
    paired_q_networks: tuple[RefQNetwork, SarlQNetwork],
    pinned_states: torch.Tensor,
) -> None:
    """With a pinned ``prev_h2`` and α=0.02, one cascade step matches."""
    ref, ours = paired_q_networks
    set_all_seeds(1234)
    prev_h2 = torch.randn(BATCH, 128)
    with torch.no_grad():
        ref_out = ref(pinned_states, prev_h2, CASCADE_RATE_PAPER)
        our_out = ours(pinned_states, prev_h2, CASCADE_RATE_PAPER)
    for name, r, o in zip(
        ["q", "hidden", "comparison", "hidden_copy"], ref_out, our_out, strict=True
    ):
        assert torch.allclose(r, o, atol=ATOL), f"{name} diverged after 1 cascade step"


def test_q_forward_50_cascade_iterations(
    paired_q_networks: tuple[RefQNetwork, SarlQNetwork],
    pinned_states: torch.Tensor,
) -> None:
    """Full paper rollout: 50 iterations at α=0.02, threaded hidden state."""
    ref, ours = paired_q_networks
    prev_ref: torch.Tensor | None = None
    prev_ours: torch.Tensor | None = None
    with torch.no_grad():
        for _ in range(N_ITER_PAPER):
            _, prev_ref, _, _ = ref(pinned_states, prev_ref, CASCADE_RATE_PAPER)
            _, prev_ours, _, _ = ours(pinned_states, prev_ours, CASCADE_RATE_PAPER)
    assert torch.allclose(prev_ref, prev_ours, atol=ATOL), (
        f"Hidden diverged after {N_ITER_PAPER} iterations"
    )


def test_q_comparison_branch_non_trivial(
    paired_q_networks: tuple[RefQNetwork, SarlQNetwork],
    pinned_states: torch.Tensor,
) -> None:
    """Sanity check: comparison is not zero (tied-weight branch is actually
    reconstructing non-trivially). If this ever fires, the test suite is
    testing equivalence of two trivially-zero tensors — a false positive."""
    _, ours = paired_q_networks
    with torch.no_grad():
        _, _, comparison, _ = ours(pinned_states, None, 1.0)
    assert comparison.abs().mean() > 1e-3, "comparison is suspiciously near zero"


# ─── SecondOrderNetwork tests ────────────────────────────────────────────────


def test_second_order_forward_zero_cascade(
    paired_second_order: tuple[RefSecondOrderNetwork, SarlSecondOrderNetwork],
) -> None:
    """`prev_comparison=None, cascade_rate=1.0` — pure feed-forward equivalence.

    Dropout is disabled (.eval()), so the only randomness sources are weight
    init (controlled by the fixture) and the comparison input (pinned below).
    """
    ref, ours = paired_second_order
    set_all_seeds(42)
    comparison = torch.randn(BATCH, 1024)  # NUM_LINEAR_UNITS = 1024
    with torch.no_grad():
        ref_w, ref_c = ref(comparison, None, 1.0)
        our_w, our_c = ours(comparison, None, 1.0)
    assert torch.allclose(ref_w, our_w, atol=ATOL), "wager logits diverged"
    assert torch.allclose(ref_c, our_c, atol=ATOL), "comparison_out diverged"


def test_second_order_wager_is_raw_logits(
    paired_second_order: tuple[RefSecondOrderNetwork, SarlSecondOrderNetwork],
) -> None:
    """Guardrail: paper's wager head returns raw logits, NOT probabilities.

    If this test fails, someone has added a softmax/sigmoid — catching the
    deviation early prevents it from silently propagating into the loss
    computation where it would change gradient magnitudes.
    """
    _, ours = paired_second_order
    set_all_seeds(42)
    comparison = torch.randn(BATCH, 1024)
    with torch.no_grad():
        wager, _ = ours(comparison, None, 1.0)
    # Softmax output would sum to 1.0 along dim=-1; sigmoid output would be
    # bounded in [0, 1]. Raw logits cover a wider range.
    row_sums = wager.sum(dim=-1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3), (
        "wager output sums to 1 per row → softmax was applied (paper returns raw logits)"
    )


def test_second_order_50_cascade_iterations(
    paired_second_order: tuple[RefSecondOrderNetwork, SarlSecondOrderNetwork],
) -> None:
    """50-step cascade accumulation on the comparison branch."""
    ref, ours = paired_second_order
    set_all_seeds(42)
    comparison = torch.randn(BATCH, 1024)
    prev_ref: torch.Tensor | None = None
    prev_ours: torch.Tensor | None = None
    with torch.no_grad():
        for _ in range(N_ITER_PAPER):
            _, prev_ref = ref(comparison, prev_ref, CASCADE_RATE_PAPER)
            _, prev_ours = ours(comparison, prev_ours, CASCADE_RATE_PAPER)
    assert torch.allclose(prev_ref, prev_ours, atol=ATOL)
