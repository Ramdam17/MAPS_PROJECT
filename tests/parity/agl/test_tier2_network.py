"""Tier-2 parity: AGL first-order network (chunked-sigmoid + cascade).

Sprint 13.G / D13.4. The shared ``FirstOrderMLP`` + cascade maths are already
parity-tested against paper_reference in Sprint 11/12; here we pin the
**AGL-specific wiring**: the ``make_chunked_sigmoid(6)`` decoder, cascade
determinism, and that ``cascade_rate=1`` (no cascade) collapses to a single
forward.
"""

from __future__ import annotations

import torch

from maps.networks.first_order_mlp import FirstOrderMLP, make_chunked_sigmoid
from maps.utils.seeding import set_all_seeds

INPUT_DIM = 48
HIDDEN_DIM = 40
BITS = 6


def _make_net():
    return FirstOrderMLP(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        decoder_activation=make_chunked_sigmoid(BITS),
        encoder_dropout=0.1,
        weight_init_range=(-1.0, 1.0),
    )


def test_forward_is_deterministic_in_eval():
    set_all_seeds(42)
    net = _make_net().eval()
    x = torch.rand(16, INPUT_DIM)
    _, h2_a = net(x, None, None, 1.0)
    _, h2_b = net(x, None, None, 1.0)
    assert torch.equal(h2_a, h2_b)


def test_chunked_sigmoid_equals_elementwise_sigmoid():
    """Chunked sigmoid is element-wise (chunking is semantic only): == torch.sigmoid,
    all outputs in [0, 1]."""
    chunked = make_chunked_sigmoid(BITS)
    x = torch.randn(4, INPUT_DIM)
    out = chunked(x)
    assert torch.allclose(out, torch.sigmoid(x), atol=1e-7)
    assert (out >= 0).all() and (out <= 1).all()


def test_cascade_rate_one_equals_single_forward():
    """rate=1 for 50 iterations must collapse to a single forward (no accumulation)."""
    set_all_seeds(7)
    net = _make_net().eval()  # eval → dropout off → deterministic across iterations
    x = torch.rand(8, INPUT_DIM)
    _, single = net(x, None, None, 1.0)

    set_all_seeds(7)
    net2 = _make_net().eval()
    h1 = h2 = None
    for _ in range(50):
        h1, h2 = net2(x, h1, h2, 1.0)
    assert torch.allclose(single, h2, atol=1e-6)


def test_cascade_backward_populates_grads():
    set_all_seeds(1)
    net = _make_net()
    x = torch.rand(8, INPUT_DIM)
    h1 = h2 = None
    for _ in range(50):
        h1, h2 = net(x, h1, h2, 0.02)
    h2.sum().backward()
    assert net.fc1.weight.grad is not None
    assert torch.isfinite(net.fc1.weight.grad).all()
