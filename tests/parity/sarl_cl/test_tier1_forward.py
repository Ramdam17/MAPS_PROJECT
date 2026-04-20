"""Tier 1 parity tests — SARL+CL forward-pass equivalence.

Asserts that ``maps.experiments.sarl_cl.model`` (SarlCLQNetwork,
SarlCLSecondOrderNetwork, AdaptiveQNetwork) reproduces the paper reference
(``tests/parity/sarl_cl._reference_sarl_cl``) bit-for-bit at ``atol=1e-6``
after ``load_state_dict`` synchronisation.

Scope mirrors ``tests/parity/sarl/test_tier1_forward.py`` but for the CL
networks. Dropout is disabled via ``.eval()`` so stochasticity does not
mask genuine math drift.
"""

from __future__ import annotations

import pytest
import torch

from maps.experiments.sarl_cl.model import (
    AdaptiveQNetwork,
    SarlCLQNetwork,
    SarlCLSecondOrderNetwork,
)
from maps.utils.seeding import set_all_seeds
from tests.parity.sarl_cl._reference_sarl_cl import (
    AdaptiveQNetwork as RefAdaptiveQNetwork,
)
from tests.parity.sarl_cl._reference_sarl_cl import (
    QNetwork as RefQNetwork,
)
from tests.parity.sarl_cl._reference_sarl_cl import (
    SecondOrderNetwork as RefSecondOrderNetwork,
)

# ─── Shared fixtures ────────────────────────────────────────────────────────

IN_CHANNELS = 4
NUM_ACTIONS = 6
MAX_INPUT_CHANNELS = 10  # paper T.11 CL row 20 (D.20-aligned)
BATCH = 32
STATE_SHAPE = (BATCH, IN_CHANNELS, 10, 10)
CASCADE_RATE_PAPER = 0.02
N_ITER_PAPER = 50
ATOL = 1e-6


@pytest.fixture
def paired_q_network() -> tuple[RefQNetwork, SarlCLQNetwork]:
    """Reference + refactored SarlCLQNetwork with identical weights."""
    set_all_seeds(42)
    ref = RefQNetwork(IN_CHANNELS, NUM_ACTIONS).eval()
    set_all_seeds(42)
    ours = SarlCLQNetwork(IN_CHANNELS, NUM_ACTIONS).eval()
    ours.load_state_dict(ref.state_dict())
    return ref, ours


@pytest.fixture
def paired_second_order() -> tuple[RefSecondOrderNetwork, SarlCLSecondOrderNetwork]:
    """Reference + refactored CL SecondOrderNetwork with identical weights."""
    set_all_seeds(42)
    ref = RefSecondOrderNetwork(IN_CHANNELS).eval()
    set_all_seeds(42)
    ours = SarlCLSecondOrderNetwork(IN_CHANNELS).eval()
    ours.load_state_dict(ref.state_dict())
    return ref, ours


@pytest.fixture
def paired_adaptive_q() -> tuple[RefAdaptiveQNetwork, AdaptiveQNetwork]:
    """Reference + refactored AdaptiveQNetwork with identical weights."""
    set_all_seeds(42)
    ref = RefAdaptiveQNetwork(MAX_INPUT_CHANNELS, NUM_ACTIONS).eval()
    set_all_seeds(42)
    ours = AdaptiveQNetwork(MAX_INPUT_CHANNELS, NUM_ACTIONS).eval()
    ours.load_state_dict(ref.state_dict())
    return ref, ours


@pytest.fixture
def pinned_states() -> torch.Tensor:
    set_all_seeds(42)
    return torch.randn(*STATE_SHAPE)


@pytest.fixture
def pinned_states_adaptive() -> torch.Tensor:
    """State with max_input_channels — Adaptive fixture uses 10 channels."""
    set_all_seeds(42)
    return torch.randn(BATCH, MAX_INPUT_CHANNELS, 10, 10)


@pytest.fixture
def pinned_states_adaptive_truncated() -> torch.Tensor:
    """State with fewer channels than max — exercises the zero-pad path."""
    set_all_seeds(42)
    return torch.randn(BATCH, 4, 10, 10)


# ─── SarlCLQNetwork tests ──────────────────────────────────────────────────


def test_sarlcl_q_forward_zero_cascade(
    paired_q_network: tuple[RefQNetwork, SarlCLQNetwork],
    pinned_states: torch.Tensor,
) -> None:
    ref, ours = paired_q_network
    q_ref, h_ref, c_ref, o_ref = ref(pinned_states, None, 1.0)
    q_ours, h_ours, c_ours, o_ours = ours(pinned_states, None, 1.0)
    assert torch.allclose(q_ref, q_ours, atol=ATOL)
    assert torch.allclose(h_ref, h_ours, atol=ATOL)
    assert torch.allclose(c_ref, c_ours, atol=ATOL)
    assert torch.allclose(o_ref, o_ours, atol=ATOL)


def test_sarlcl_q_forward_single_step_cascade(
    paired_q_network: tuple[RefQNetwork, SarlCLQNetwork],
    pinned_states: torch.Tensor,
) -> None:
    ref, ours = paired_q_network
    prev = torch.randn(BATCH, 1024) * 0.1  # matches Output dim (1024)
    q_ref, _, _, o_ref = ref(pinned_states, prev, CASCADE_RATE_PAPER)
    q_ours, _, _, o_ours = ours(pinned_states, prev, CASCADE_RATE_PAPER)
    assert torch.allclose(q_ref, q_ours, atol=ATOL)
    assert torch.allclose(o_ref, o_ours, atol=ATOL)


def test_sarlcl_q_forward_multi_iter_cascade(
    paired_q_network: tuple[RefQNetwork, SarlCLQNetwork],
    pinned_states: torch.Tensor,
) -> None:
    """Paper's 50-iter cascade rollout at α=0.02 must match."""
    ref, ours = paired_q_network
    prev_ref: torch.Tensor | None = None
    prev_ours: torch.Tensor | None = None
    for _ in range(N_ITER_PAPER):
        _, _, _, prev_ref = ref(pinned_states, prev_ref, CASCADE_RATE_PAPER)
        _, _, _, prev_ours = ours(pinned_states, prev_ours, CASCADE_RATE_PAPER)
    assert prev_ref is not None and prev_ours is not None
    assert torch.allclose(prev_ref, prev_ours, atol=ATOL)


def test_sarlcl_q_reconstruction_residual(
    paired_q_network: tuple[RefQNetwork, SarlCLQNetwork],
    pinned_states: torch.Tensor,
) -> None:
    """Comparison = Input - Output must agree with the reference."""
    ref, ours = paired_q_network
    _, _, c_ref, _ = ref(pinned_states, None, 1.0)
    _, _, c_ours, _ = ours(pinned_states, None, 1.0)
    assert torch.allclose(c_ref, c_ours, atol=ATOL)


# ─── SarlCLSecondOrderNetwork tests ────────────────────────────────────────


def test_sarlcl_second_order_forward_zero_cascade(
    paired_second_order: tuple[RefSecondOrderNetwork, SarlCLSecondOrderNetwork],
) -> None:
    ref, ours = paired_second_order
    set_all_seeds(42)
    comparison = torch.randn(BATCH, 1024)
    w_ref, c_ref = ref(comparison, None, 1.0)
    w_ours, c_ours = ours(comparison, None, 1.0)
    assert torch.allclose(w_ref, w_ours, atol=ATOL)
    assert torch.allclose(c_ref, c_ours, atol=ATOL)


def test_sarlcl_second_order_multi_iter_cascade(
    paired_second_order: tuple[RefSecondOrderNetwork, SarlCLSecondOrderNetwork],
) -> None:
    ref, ours = paired_second_order
    set_all_seeds(42)
    comparison = torch.randn(BATCH, 1024)
    prev_ref: torch.Tensor | None = None
    prev_ours: torch.Tensor | None = None
    for _ in range(N_ITER_PAPER):
        _, prev_ref = ref(comparison, prev_ref, CASCADE_RATE_PAPER)
        _, prev_ours = ours(comparison, prev_ours, CASCADE_RATE_PAPER)
    assert prev_ref is not None and prev_ours is not None
    assert torch.allclose(prev_ref, prev_ours, atol=ATOL)


# ─── AdaptiveQNetwork tests ────────────────────────────────────────────────


def test_adaptive_q_forward_full_channels(
    paired_adaptive_q: tuple[RefAdaptiveQNetwork, AdaptiveQNetwork],
    pinned_states_adaptive: torch.Tensor,
) -> None:
    ref, ours = paired_adaptive_q
    q_ref, h_ref, c_ref, o_ref = ref(pinned_states_adaptive, None, 1.0)
    q_ours, h_ours, c_ours, o_ours = ours(pinned_states_adaptive, None, 1.0)
    assert torch.allclose(q_ref, q_ours, atol=ATOL)
    assert torch.allclose(h_ref, h_ours, atol=ATOL)
    assert torch.allclose(c_ref, c_ours, atol=ATOL)
    assert torch.allclose(o_ref, o_ours, atol=ATOL)


def test_adaptive_q_forward_channel_zero_padding(
    paired_adaptive_q: tuple[RefAdaptiveQNetwork, AdaptiveQNetwork],
    pinned_states_adaptive_truncated: torch.Tensor,
) -> None:
    """Inputs with fewer channels than max must be zero-padded identically."""
    ref, ours = paired_adaptive_q
    q_ref, _, _, _ = ref(pinned_states_adaptive_truncated, None, 1.0)
    q_ours, _, _, _ = ours(pinned_states_adaptive_truncated, None, 1.0)
    assert torch.allclose(q_ref, q_ours, atol=ATOL)


def test_adaptive_q_multi_iter_cascade(
    paired_adaptive_q: tuple[RefAdaptiveQNetwork, AdaptiveQNetwork],
    pinned_states_adaptive: torch.Tensor,
) -> None:
    ref, ours = paired_adaptive_q
    prev_ref: torch.Tensor | None = None
    prev_ours: torch.Tensor | None = None
    for _ in range(N_ITER_PAPER):
        _, _, _, prev_ref = ref(pinned_states_adaptive, prev_ref, CASCADE_RATE_PAPER)
        _, _, _, prev_ours = ours(pinned_states_adaptive, prev_ours, CASCADE_RATE_PAPER)
    assert prev_ref is not None and prev_ours is not None
    assert torch.allclose(prev_ref, prev_ours, atol=ATOL)
