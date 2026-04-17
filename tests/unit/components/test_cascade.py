"""Unit tests for cascade_update invariants."""

from __future__ import annotations

import pytest
import torch

from maps.components.cascade import cascade_update, n_iterations_from_alpha


def test_alpha_one_returns_new_activation():
    new = torch.randn(4, 10)
    prev = torch.randn(4, 10)
    assert torch.allclose(cascade_update(new, prev, alpha=1.0), new)


def test_prev_none_bootstrap():
    new = torch.randn(4, 10)
    assert torch.allclose(cascade_update(new, None, alpha=0.02), new)


def test_cascade_converges_to_new_after_many_iterations():
    """With constant `new`, the cascade state should asymptote to `new`."""
    new = torch.ones(2, 3) * 5.0
    state = None
    for _ in range(500):  # >> 1/α
        state = cascade_update(new, state, alpha=0.02)
    assert torch.allclose(state, new, atol=1e-4)


def test_alpha_zero_raises():
    with pytest.raises(ValueError, match="alpha must lie"):
        cascade_update(torch.zeros(2), torch.zeros(2), alpha=0.0)


def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError):
        cascade_update(torch.zeros(2), torch.zeros(2), alpha=1.5)
    with pytest.raises(ValueError):
        cascade_update(torch.zeros(2), torch.zeros(2), alpha=-0.1)


def test_gradient_flows():
    """Cascade update must be autograd-compatible for training."""
    new = torch.randn(4, 8, requires_grad=True)
    prev = torch.randn(4, 8)
    out = cascade_update(new, prev, alpha=0.5)
    out.sum().backward()
    assert new.grad is not None
    assert not torch.allclose(new.grad, torch.zeros_like(new.grad))


def test_n_iterations_from_alpha():
    assert n_iterations_from_alpha(0.02) == 50
    assert n_iterations_from_alpha(0.1) == 10
    assert n_iterations_from_alpha(1.0) == 1
