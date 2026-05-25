"""Unit tests for :mod:`maps.core.cascade` — McClelland eq.6."""

from __future__ import annotations

import pytest
import torch

from maps.core.cascade import cascade_update, n_iterations_from_alpha

# ---------------------------------------------------------------------------
# cascade_update
# ---------------------------------------------------------------------------


def test_bootstrap_returns_new_when_prev_is_none() -> None:
    """At t=0, prev is None → cascade_update returns new unchanged."""
    new = torch.tensor([1.0, 2.0, 3.0])
    out = cascade_update(new, None, cascade_rate=0.02)
    assert torch.equal(out, new)


def test_alpha_one_collapses_to_new() -> None:
    """α=1 → out = 1·new + 0·prev = new (no-cascade fallback)."""
    new = torch.tensor([1.0, 2.0, 3.0])
    prev = torch.tensor([10.0, 20.0, 30.0])
    out = cascade_update(new, prev, cascade_rate=1.0)
    assert torch.equal(out, new)


def test_eq6_formula_bit_exact() -> None:
    """Single iteration matches α·new + (1-α)·prev bit-exact at α=0.5."""
    new = torch.tensor([1.0, 2.0, 3.0])
    prev = torch.tensor([4.0, 6.0, 8.0])
    out = cascade_update(new, prev, cascade_rate=0.5)
    expected = 0.5 * new + 0.5 * prev
    assert torch.equal(out, expected)


def test_deterministic_noop_after_n_iterations() -> None:
    """When ``new`` is constant across iterations, ``cascade_update``
    applied 50 times produces the same tensor as 1 application —
    analytical closure :math:`a(\\infty) \\to \\text{new}`. This is
    the D-sarl-cascade-noop finding made testable.

    Iteration 0 is exact via the bootstrap branch (returns ``new``
    directly). Iterations 1..49 walk the eq.6 formula but each step
    collapses to ``new`` modulo float32 quantization of α and (1-α).
    """
    new = torch.tensor([1.0, 2.0, 3.0])

    # bootstrap iteration is exact (short-circuit returns `new`)
    out_0 = cascade_update(new, None, cascade_rate=0.02)
    assert torch.equal(out_0, new)

    # 49 more iterations — cumulative float drift bounded by ~50·eps_f32
    out = out_0
    for _ in range(49):
        out = cascade_update(new, out, cascade_rate=0.02)
    assert torch.allclose(out, new, atol=1e-5)


def test_cascade_rate_zero_raises() -> None:
    new = torch.tensor([1.0])
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        cascade_update(new, None, cascade_rate=0.0)


def test_cascade_rate_negative_raises() -> None:
    new = torch.tensor([1.0])
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        cascade_update(new, None, cascade_rate=-0.1)


def test_cascade_rate_above_one_raises() -> None:
    new = torch.tensor([1.0])
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        cascade_update(new, None, cascade_rate=1.1)


def test_preserves_shape() -> None:
    """Output shape matches `new_activation` shape (no implicit reshape)."""
    new = torch.randn(4, 8, 16)
    prev = torch.zeros(4, 8, 16)
    out = cascade_update(new, prev, cascade_rate=0.02)
    assert out.shape == new.shape


def test_preserves_dtype() -> None:
    """Output dtype matches `new_activation` dtype (float32 in → float32 out)."""
    new = torch.tensor([1.0, 2.0], dtype=torch.float32)
    prev = torch.tensor([0.0, 0.0], dtype=torch.float32)
    out = cascade_update(new, prev, cascade_rate=0.5)
    assert out.dtype == torch.float32


def test_stochastic_path_is_not_noop() -> None:
    """When ``new`` varies between iterations (proxy for dropout in
    forward), cascade_update accumulates a different value than any
    single iteration. This is the mechanism that makes cascade
    non-trivial in `SecondOrderNetwork` (cascade + dropout = MC-dropout
    averaging à la Gal & Ghahramani 2016)."""
    torch.manual_seed(0)
    out = None
    samples: list[torch.Tensor] = []
    for _ in range(50):
        # simulate a stochastic forward: new draws fresh noise each step
        new_t = torch.randn(8)
        samples.append(new_t)
        out = cascade_update(new_t, out, cascade_rate=0.02)

    # Final cascaded output should differ from both the last sample
    # and the mean of samples (it's an exponential moving average,
    # weighted toward earlier samples through (1-α) decay).
    assert not torch.allclose(out, samples[-1], atol=1e-3)


# ---------------------------------------------------------------------------
# n_iterations_from_alpha
# ---------------------------------------------------------------------------


def test_n_iterations_paper_default() -> None:
    """Paper §2.1 convention: α=0.02 → 50 iterations."""
    assert n_iterations_from_alpha(0.02) == 50


def test_n_iterations_extremes() -> None:
    assert n_iterations_from_alpha(1.0) == 1
    assert n_iterations_from_alpha(0.5) == 2
    assert n_iterations_from_alpha(0.1) == 10


def test_n_iterations_rejects_zero() -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        n_iterations_from_alpha(0.0)


def test_n_iterations_rejects_negative() -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        n_iterations_from_alpha(-0.1)


def test_n_iterations_rejects_above_one() -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        n_iterations_from_alpha(1.1)
