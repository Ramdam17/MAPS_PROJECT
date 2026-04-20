"""Unit tests for ComparatorMatrix, WageringHead, SecondOrderNetwork."""

from __future__ import annotations

import pytest
import torch

from maps.components import ComparatorMatrix, SecondOrderNetwork, WageringHead


def test_comparator_is_elementwise_diff():
    x = torch.randn(4, 10)
    y = torch.randn(4, 10)
    out = ComparatorMatrix()(x, y)
    assert out.shape == x.shape
    assert torch.allclose(out, x - y)


def test_wagering_head_single_unit_shape_and_range():
    head = WageringHead(input_dim=100, n_wager_units=1)
    x = torch.randn(4, 100)
    out = head(x)
    assert out.shape == (4, 1)
    assert ((out >= 0) & (out <= 1)).all()


def test_wagering_head_two_unit_raw_logits():
    """Paper eq.3: 2-unit wager returns raw logits (no softmax, no sigmoid).

    Eq.5 applies `binary_cross_entropy_with_logits` per unit → internal sigmoid
    per unit, NOT a softmax over the 2 units. See docs/reviews/second_order.md
    §C.4 + deviations.md D-001 for the rationale.
    """
    torch.manual_seed(0)
    head = WageringHead(input_dim=100, n_wager_units=2)
    x = torch.randn(4, 100)
    out = head(x)
    assert out.shape == (4, 2)
    # Raw logits are unconstrained — must NOT sum to 1 generically.
    assert not torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-3)
    # And can be outside [0, 1] — at least one value should be < 0 or > 1
    # across the batch (uniform init + random input makes this near-certain).
    assert (out < 0).any() or (out > 1).any()


def test_wagering_head_rejects_other_unit_counts():
    with pytest.raises(ValueError, match="n_wager_units"):
        WageringHead(input_dim=100, n_wager_units=3)


def test_second_order_network_shapes():
    net = SecondOrderNetwork(input_dim=100, n_wager_units=1).eval()
    x = torch.randn(4, 100)
    y = torch.randn(4, 100)
    wager, comp = net(x, y, prev_comparison=None, cascade_rate=1.0)
    assert wager.shape == (4, 1)
    assert comp.shape == (4, 100)


def test_second_order_cascade_reduces_to_passthrough_at_alpha_one():
    """With α=1, the cascade is a no-op: state = current comparator output."""
    torch.manual_seed(0)
    net = SecondOrderNetwork(input_dim=100, n_wager_units=1).eval()
    x = torch.randn(4, 100)
    y = torch.randn(4, 100)

    _, comp1 = net(x, y, prev_comparison=None, cascade_rate=1.0)
    garbage = torch.randn_like(comp1)
    _, comp2 = net(x, y, prev_comparison=garbage, cascade_rate=1.0)
    # With α=1, `prev_comparison` is overwritten: comp2 should equal comp1 exactly.
    assert torch.allclose(comp1, comp2, atol=1e-6)


def test_second_order_dropout_inactive_in_eval():
    """In `.eval()`, the dropout is off → outputs are deterministic across calls."""
    torch.manual_seed(0)
    net = SecondOrderNetwork(input_dim=100).eval()
    x = torch.randn(4, 100)
    y = torch.randn(4, 100)
    out1, _ = net(x, y, prev_comparison=None, cascade_rate=1.0)
    out2, _ = net(x, y, prev_comparison=None, cascade_rate=1.0)
    assert torch.allclose(out1, out2)


def test_second_order_gradient_flows_to_comparator_input():
    net = SecondOrderNetwork(input_dim=100).eval()
    x = torch.randn(4, 100, requires_grad=True)
    y = torch.randn(4, 100)
    wager, _ = net(x, y, prev_comparison=None, cascade_rate=1.0)
    wager.sum().backward()
    assert x.grad is not None
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
