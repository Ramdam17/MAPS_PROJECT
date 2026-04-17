"""Parity tests: new maps.components must match the frozen Blindsight reference.

If these break, the new API has diverged from Juan's reference code. That is a
bug in `src/maps/` (not in the reference) — the reference snapshot is
intentionally frozen.

The two networks use randomness in their init (uniform weights) and forward
(dropout). We set the seed, instantiate both, copy weights from reference to
new, then compare outputs under `.eval()` (dropout disabled) to get
deterministic comparisons.
"""

from __future__ import annotations

import pytest
import torch

from maps.components import SecondOrderNetwork
from maps.networks import FirstOrderMLP
from tests.parity._reference_blindsight import (
    ReferenceFirstOrderNetwork,
    ReferenceSecondOrderNetwork,
)

SEED = 42
BATCH = 8
INPUT_DIM = 100
HIDDEN = 40
ALPHA = 0.02


@pytest.fixture(autouse=True)
def _reset_seed():
    torch.manual_seed(SEED)


def _copy_first_order_weights(new: FirstOrderMLP, ref: ReferenceFirstOrderNetwork) -> None:
    with torch.no_grad():
        new.fc1.weight.copy_(ref.fc1.weight)
        new.fc2.weight.copy_(ref.fc2.weight)


def _copy_second_order_weights(new: SecondOrderNetwork, ref: ReferenceSecondOrderNetwork) -> None:
    with torch.no_grad():
        new.wagering_head.wager.weight.copy_(ref.wager.weight)
        new.wagering_head.wager.bias.copy_(ref.wager.bias)


def test_first_order_matches_reference_single_pass():
    ref = ReferenceFirstOrderNetwork(hidden_units=HIDDEN, data_factor=1, use_gelu=False).eval()
    new = FirstOrderMLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN).eval()
    _copy_first_order_weights(new, ref)

    x = torch.randn(BATCH, INPUT_DIM)
    h1_ref, h2_ref = ref(x, prev_h1=None, prev_h2=None, cascade_rate=1.0)
    h1_new, h2_new = new(x, prev_h1=None, prev_h2=None, cascade_rate=1.0)

    assert torch.allclose(h1_ref, h1_new, atol=1e-6), "encoder output diverges"
    assert torch.allclose(h2_ref, h2_new, atol=1e-6), "decoder output diverges"


def test_first_order_matches_reference_cascade_loop():
    """50-iteration cascade with α=0.02 must agree step-for-step."""
    ref = ReferenceFirstOrderNetwork(hidden_units=HIDDEN, data_factor=1, use_gelu=False).eval()
    new = FirstOrderMLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN).eval()
    _copy_first_order_weights(new, ref)

    x = torch.randn(BATCH, INPUT_DIM)
    h2_ref = None
    h2_new = None
    for _ in range(50):
        _, h2_ref = ref(x, prev_h1=None, prev_h2=h2_ref, cascade_rate=ALPHA)
        _, h2_new = new(x, prev_h1=None, prev_h2=h2_new, cascade_rate=ALPHA)

    assert torch.allclose(h2_ref, h2_new, atol=1e-6)


def test_second_order_matches_reference_single_pass():
    ref = ReferenceSecondOrderNetwork(use_gelu=False, hidden_2nd=100).eval()
    new = SecondOrderNetwork(input_dim=INPUT_DIM, n_wager_units=1).eval()
    _copy_second_order_weights(new, ref)

    first_input = torch.randn(BATCH, INPUT_DIM)
    first_output = torch.randn(BATCH, INPUT_DIM)

    wager_ref, comp_ref = ref(first_input, first_output, prev_comparison=None, cascade_rate=1.0)
    wager_new, comp_new = new(first_input, first_output, prev_comparison=None, cascade_rate=1.0)

    assert torch.allclose(comp_ref, comp_new, atol=1e-6)
    assert torch.allclose(wager_ref, wager_new, atol=1e-6)


def test_second_order_matches_reference_cascade_loop():
    ref = ReferenceSecondOrderNetwork(use_gelu=False, hidden_2nd=100).eval()
    new = SecondOrderNetwork(input_dim=INPUT_DIM, n_wager_units=1).eval()
    _copy_second_order_weights(new, ref)

    first_input = torch.randn(BATCH, INPUT_DIM)
    first_output = torch.randn(BATCH, INPUT_DIM)

    comp_ref = None
    comp_new = None
    for _ in range(50):
        _, comp_ref = ref(first_input, first_output, prev_comparison=comp_ref, cascade_rate=ALPHA)
        _, comp_new = new(first_input, first_output, prev_comparison=comp_new, cascade_rate=ALPHA)

    assert torch.allclose(comp_ref, comp_new, atol=1e-6)
