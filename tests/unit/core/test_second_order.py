"""Unit tests for :mod:`maps.core.second_order`."""

from __future__ import annotations

import pytest
import torch

from maps.core.second_order import (
    ComparatorMatrix,
    SecondOrderNetwork,
    WageringHead,
)

# ---------------------------------------------------------------------------
# ComparatorMatrix
# ---------------------------------------------------------------------------


def test_comparator_eq1_bit_exact() -> None:
    """ComparatorMatrix(x, y) == x - y for arbitrary shapes."""
    cm = ComparatorMatrix()
    for shape in [(4, 10), (1, 1), (8, 100), (16, 48)]:
        x = torch.randn(*shape)
        y = torch.randn(*shape)
        assert torch.equal(cm(x, y), x - y)


def test_comparator_shape_mismatch_raises() -> None:
    cm = ComparatorMatrix()
    x = torch.randn(4, 10)
    y = torch.randn(4, 8)  # different last dim
    with pytest.raises(ValueError, match="same shape"):
        cm(x, y)


def test_comparator_has_no_parameters() -> None:
    """Stateless (nn.Module for API symmetry) — no learnable params."""
    cm = ComparatorMatrix()
    assert sum(p.numel() for p in cm.parameters()) == 0


# ---------------------------------------------------------------------------
# WageringHead
# ---------------------------------------------------------------------------


def test_wager_head_n1_returns_sigmoid_probs() -> None:
    """n_wager_units=1 → sigmoid-activated, values in [0, 1]."""
    head = WageringHead(input_dim=16, n_wager_units=1)
    x = torch.randn(8, 16) * 10  # large logits
    out = head(x)
    assert out.shape == (8, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_wager_head_n2_returns_raw_logits() -> None:
    """n_wager_units=2 → raw logits (no softmax, can be negative)."""
    head = WageringHead(input_dim=16, n_wager_units=2)
    # Use large negative inputs to force at least one negative logit
    # despite the non-negative readout weights (init uniform[0, 0.1]).
    x = torch.full((8, 16), -5.0)
    out = head(x)
    assert out.shape == (8, 2)
    # Some logits must be negative — guards against an accidental
    # softmax/sigmoid wrapper around the 2-unit path.
    assert (out < 0).any()


def test_wager_head_hidden_layer_adds_parameters() -> None:
    """hidden_dim=10 inserts Linear→ReLU→Linear (Pasquali D.25)."""
    no_hidden = WageringHead(input_dim=16, n_wager_units=1, hidden_dim=None)
    with_hidden = WageringHead(input_dim=16, n_wager_units=1, hidden_dim=10)

    n_no = sum(p.numel() for p in no_hidden.parameters())
    n_yes = sum(p.numel() for p in with_hidden.parameters())

    # no_hidden: Linear(16 → 1) = 16*1 + 1 = 17 params
    assert n_no == 17
    # with_hidden: Linear(16 → 10) + Linear(10 → 1)
    #   = (16*10 + 10) + (10*1 + 1) = 170 + 11 = 181 params
    assert n_yes == 181


def test_wager_head_hidden_uses_relu_activation() -> None:
    """Hidden layer uses ReLU: negative pre-activations get clamped to 0
    and the readout sees only the bias term."""
    head = WageringHead(input_dim=4, n_wager_units=1, hidden_dim=4)
    # Force hidden pre-activations strongly negative → ReLU → all zeros.
    with torch.no_grad():
        head.hidden.weight.fill_(-1.0)
        head.hidden.bias.fill_(-1.0)

    x = torch.ones(1, 4)  # hidden pre-act = -5 < 0 → ReLU → 0
    out = head(x)
    # With hidden output == 0, readout returns sigmoid(readout.bias).
    expected = torch.sigmoid(head.readout.bias)
    assert torch.allclose(out, expected, atol=1e-6)


def test_wager_head_readout_init_uniform_0_to_0_1() -> None:
    """Readout weight is init uniform[0, 0.1] (student-parity)."""
    head = WageringHead(input_dim=500, n_wager_units=1)
    w = head.readout.weight.flatten()
    assert (w >= 0.0).all()
    assert (w <= 0.1).all()
    # uniform(0, 0.1): true mean=0.05, true std=0.1/sqrt(12)≈0.0289.
    # 500 samples → tight bounds.
    assert 0.04 < w.mean().item() < 0.06
    assert 0.025 < w.std().item() < 0.033


def test_wager_head_invalid_n_wager_units_raises() -> None:
    with pytest.raises(ValueError, match="n_wager_units"):
        WageringHead(input_dim=10, n_wager_units=0)


def test_wager_head_invalid_hidden_dim_raises() -> None:
    with pytest.raises(ValueError, match="hidden_dim"):
        WageringHead(input_dim=10, hidden_dim=0)
    with pytest.raises(ValueError, match="hidden_dim"):
        WageringHead(input_dim=10, hidden_dim=-5)


# ---------------------------------------------------------------------------
# SecondOrderNetwork
# ---------------------------------------------------------------------------


def test_so_forward_returns_tuple_with_correct_shapes() -> None:
    so = SecondOrderNetwork(input_dim=100, n_wager_units=1)
    so.eval()
    fi = torch.randn(8, 100)
    fo = torch.randn(8, 100)
    wager, comparison_out = so(fi, fo, prev_comparison=None, cascade_rate=0.02)
    assert wager.shape == (8, 1)
    assert comparison_out.shape == (8, 100)


def test_so_threading_prev_comparison_changes_output() -> None:
    """Different ``prev_comparison`` values produce different cascade
    outputs (threading actually does something — guards against the
    bug where a caller silently drops the returned comparison)."""
    so = SecondOrderNetwork(input_dim=8, n_wager_units=1, dropout=0.0)
    so.eval()
    fi = torch.randn(4, 8)
    fo = torch.randn(4, 8)

    _, out_none = so(fi, fo, prev_comparison=None, cascade_rate=0.5)
    _, out_zero = so(fi, fo, prev_comparison=torch.zeros(4, 8), cascade_rate=0.5)
    _, out_ones = so(fi, fo, prev_comparison=torch.ones(4, 8), cascade_rate=0.5)

    assert not torch.allclose(out_none, out_zero)
    assert not torch.allclose(out_zero, out_ones)


def test_so_pasquali_hidden_changes_wager() -> None:
    """Wager output WITH hidden layer ≠ wager WITHOUT hidden layer
    on the same inputs — proves the Pasquali hidden path is wired in,
    not silently dropped like the student bug D.25."""
    torch.manual_seed(0)
    so_flat = SecondOrderNetwork(input_dim=16, hidden_dim=None, dropout=0.0)
    torch.manual_seed(0)
    so_pasquali = SecondOrderNetwork(input_dim=16, hidden_dim=10, dropout=0.0)
    so_flat.eval()
    so_pasquali.eval()

    fi = torch.randn(4, 16)
    fo = torch.randn(4, 16)

    w_flat, _ = so_flat(fi, fo, prev_comparison=None, cascade_rate=0.5)
    w_pasquali, _ = so_pasquali(fi, fo, prev_comparison=None, cascade_rate=0.5)
    assert not torch.allclose(w_flat, w_pasquali)


def test_so_eval_mode_cascade_converges_to_comparator() -> None:
    """In eval mode (dropout off), 50 cascade steps converge to
    ``fi - fo`` — D-sarl-cascade-noop made visible at the composition
    level. Without dropout the cascade has nothing to average."""
    so = SecondOrderNetwork(input_dim=8, dropout=0.5)
    so.eval()  # dropout off → deterministic path
    fi = torch.randn(2, 8)
    fo = torch.randn(2, 8)

    prev: torch.Tensor | None = None
    for _ in range(50):
        _, prev = so(fi, fo, prev_comparison=prev, cascade_rate=0.02)

    assert prev is not None
    assert torch.allclose(prev, fi - fo, atol=1e-5)


def test_so_train_mode_cascade_differs_from_eval() -> None:
    """In train mode the dropout-cascade interaction produces a
    different running state from eval (cascade + dropout = MC-dropout
    averaging à la Gal & Ghahramani 2016). The two paths must NOT
    coincide — that's the whole reason 50 iters exist."""
    so = SecondOrderNetwork(input_dim=8, dropout=0.5)
    fi = torch.randn(2, 8)
    fo = torch.randn(2, 8)

    so.train()
    prev_train: torch.Tensor | None = None
    for _ in range(50):
        _, prev_train = so(fi, fo, prev_comparison=prev_train, cascade_rate=0.02)

    so.eval()
    prev_eval: torch.Tensor | None = None
    for _ in range(50):
        _, prev_eval = so(fi, fo, prev_comparison=prev_eval, cascade_rate=0.02)

    assert prev_train is not None and prev_eval is not None
    assert not torch.allclose(prev_train, prev_eval, atol=1e-3)


def test_so_gradient_flows_through_cascade() -> None:
    """Backward populates gradients on both first-order tensors."""
    so = SecondOrderNetwork(input_dim=8, dropout=0.0)
    so.eval()
    fi = torch.randn(2, 8, requires_grad=True)
    fo = torch.randn(2, 8, requires_grad=True)

    wager, _ = so(fi, fo, prev_comparison=None, cascade_rate=0.5)
    loss = wager.sum()
    loss.backward()

    assert fi.grad is not None and fi.grad.abs().sum() > 0
    assert fo.grad is not None and fo.grad.abs().sum() > 0


def test_so_paper_blindsight_default_dims() -> None:
    """Smoke check on paper-default Blindsight dims (input=100, hidden=100,
    n_wager=1, dropout=0.5) — must build, forward-pass, and back-prop."""
    so = SecondOrderNetwork(input_dim=100, n_wager_units=1, hidden_dim=100, dropout=0.5)
    fi = torch.randn(4, 100)
    fo = torch.randn(4, 100)
    wager, comparison = so(fi, fo, prev_comparison=None, cascade_rate=0.02)
    assert wager.shape == (4, 1)
    assert comparison.shape == (4, 100)
