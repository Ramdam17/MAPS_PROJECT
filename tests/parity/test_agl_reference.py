"""Parity tests: FirstOrderMLP with chunked-sigmoid decoder must match AGL reference."""

from __future__ import annotations

import pytest
import torch

from maps.networks import FirstOrderMLP, make_chunked_sigmoid
from tests.parity._reference_agl import BITS_PER_LETTER, ReferenceAGLFirstOrderNetwork

SEED = 42
BATCH = 8
INPUT_DIM = 48
HIDDEN = 40
ALPHA = 0.02


@pytest.fixture(autouse=True)
def _reset_seed():
    torch.manual_seed(SEED)


def _copy_weights(new: FirstOrderMLP, ref: ReferenceAGLFirstOrderNetwork) -> None:
    with torch.no_grad():
        new.fc1.weight.copy_(ref.fc1.weight)
        new.fc2.weight.copy_(ref.fc2.weight)


def test_agl_first_order_matches_reference_single_pass():
    ref = ReferenceAGLFirstOrderNetwork(hidden_units=HIDDEN, data_factor=1, use_gelu=False).eval()
    new = FirstOrderMLP(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN,
        decoder_activation=make_chunked_sigmoid(BITS_PER_LETTER),
    ).eval()
    _copy_weights(new, ref)

    x = torch.randn(BATCH, INPUT_DIM)
    h1_ref, h2_ref = ref(x, prev_h1=None, prev_h2=None, cascade_rate=1.0)
    h1_new, h2_new = new(x, prev_h1=None, prev_h2=None, cascade_rate=1.0)

    assert torch.allclose(h1_ref, h1_new, atol=1e-6)
    assert torch.allclose(h2_ref, h2_new, atol=1e-6)


def test_agl_first_order_matches_reference_cascade_loop():
    ref = ReferenceAGLFirstOrderNetwork(hidden_units=HIDDEN, data_factor=1, use_gelu=False).eval()
    new = FirstOrderMLP(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN,
        decoder_activation=make_chunked_sigmoid(BITS_PER_LETTER),
    ).eval()
    _copy_weights(new, ref)

    x = torch.randn(BATCH, INPUT_DIM)
    h2_ref = None
    h2_new = None
    for _ in range(50):
        _, h2_ref = ref(x, prev_h1=None, prev_h2=h2_ref, cascade_rate=ALPHA)
        _, h2_new = new(x, prev_h1=None, prev_h2=h2_new, cascade_rate=ALPHA)

    assert torch.allclose(h2_ref, h2_new, atol=1e-6)
