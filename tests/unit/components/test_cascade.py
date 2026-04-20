"""Unit tests for cascade_update invariants."""

from __future__ import annotations

import pytest
import torch

from maps.components.cascade import cascade_update, n_iterations_from_alpha


def test_alpha_one_returns_new_activation():
    new = torch.randn(4, 10)
    prev = torch.randn(4, 10)
    assert torch.allclose(cascade_update(new, prev, cascade_rate=1.0), new)


def test_prev_none_bootstrap():
    new = torch.randn(4, 10)
    assert torch.allclose(cascade_update(new, None, cascade_rate=0.02), new)


def test_cascade_converges_to_new_after_many_iterations():
    """With constant `new`, the cascade state should asymptote to `new`."""
    new = torch.ones(2, 3) * 5.0
    state = None
    for _ in range(500):  # >> 1/α
        state = cascade_update(new, state, cascade_rate=0.02)
    assert torch.allclose(state, new, atol=1e-4)


def test_alpha_zero_raises():
    with pytest.raises(ValueError, match="cascade_rate must lie"):
        cascade_update(torch.zeros(2), torch.zeros(2), cascade_rate=0.0)


def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError):
        cascade_update(torch.zeros(2), torch.zeros(2), cascade_rate=1.5)
    with pytest.raises(ValueError):
        cascade_update(torch.zeros(2), torch.zeros(2), cascade_rate=-0.1)


def test_gradient_flows():
    """Cascade update must be autograd-compatible for training."""
    new = torch.randn(4, 8, requires_grad=True)
    prev = torch.randn(4, 8)
    out = cascade_update(new, prev, cascade_rate=0.5)
    out.sum().backward()
    assert new.grad is not None
    assert not torch.allclose(new.grad, torch.zeros_like(new.grad))


def test_n_iterations_from_alpha():
    assert n_iterations_from_alpha(0.02) == 50
    assert n_iterations_from_alpha(0.1) == 10
    assert n_iterations_from_alpha(1.0) == 1


def test_cascade_noop_when_new_equals_prev():
    """Single-step no-op property: when `new == prev`, cascade returns `prev`
    within 1 float32 ULP.

    Captures the finding from docs/reviews/cascade.md §(d): algebraically
    `α·x + (1-α)·x == x` for any α. In float32 the result is within
    ~2·ε ≈ 2.4e-7 of `x` due to two rounded multiplications + an add,
    hence `atol=1e-6` (safe margin).
    """
    x = torch.randn(4, 128)
    for alpha in [0.02, 0.1, 0.45, 0.5, 0.999, 1.0]:
        out = cascade_update(x, x, cascade_rate=alpha)
        assert torch.allclose(out, x, atol=1e-6), (
            f"α={alpha}: expected ≈ x when new==prev, max_diff={((out - x).abs().max().item())}"
        )


def test_cascade_deterministic_caller_loop_is_1_iter_equivalent():
    """Simulate a caller loop where the forward returns a constant `h_raw`
    (deterministic forward path, as SarlQNetwork without dropout). The cascade
    state after 50 iterations must equal `h_raw` exactly (atol=1e-6) — so the
    50-iter loop is a mathematical no-op vs a single iteration.

    This test documents the behavior flagged in docs/reviews/cascade.md §(d)
    as a regression gate: if the cascade state diverges from `h_raw` on a
    deterministic forward, something fundamental has shifted and we need to
    rediscuss.
    """
    h_raw = torch.randn(8, 128) * 3.0  # arbitrary shape + scale

    # Simulate caller loop: prev starts None (bootstrap), `new` is always h_raw.
    state = None
    for _ in range(50):
        state = cascade_update(h_raw, state, cascade_rate=0.02)

    assert state is not None
    assert torch.allclose(state, h_raw, atol=1e-6), (
        f"Deterministic 50-iter loop should give h_raw exactly; "
        f"max_diff={((state - h_raw).abs().max().item())}"
    )
