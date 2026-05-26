"""Unit tests for :mod:`maps.networks.first_order_mlp`."""

from __future__ import annotations

import pytest
import torch

from maps.networks.first_order_mlp import (
    FirstOrderMLP,
    global_sigmoid,
    make_chunked_sigmoid,
)

# ---------------------------------------------------------------------------
# global_sigmoid
# ---------------------------------------------------------------------------


def test_global_sigmoid_output_in_unit_range() -> None:
    x = torch.randn(8, 100) * 10
    out = global_sigmoid(x)
    assert (out >= 0).all() and (out <= 1).all()
    assert out.shape == x.shape


def test_global_sigmoid_matches_torch_sigmoid() -> None:
    x = torch.randn(4, 48)
    assert torch.equal(global_sigmoid(x), torch.sigmoid(x))


# ---------------------------------------------------------------------------
# make_chunked_sigmoid
# ---------------------------------------------------------------------------


def test_chunked_sigmoid_equiv_to_global_when_divisible() -> None:
    """Sigmoid is element-wise → chunked output == global output
    (modulo memory layout: torch.cat may produce a different stride
    than the contiguous tensor returned by sigmoid; values are equal)."""
    chunk = make_chunked_sigmoid(6)
    x = torch.randn(4, 48)  # 48 = 8 * 6
    out_chunk = chunk(x)
    out_global = global_sigmoid(x)
    assert torch.allclose(out_chunk, out_global, atol=1e-7)


def test_chunked_sigmoid_handles_non_divisible_last_dim() -> None:
    """If the last dim is not a multiple of chunk_size, the final
    chunk is shorter — the function still works."""
    chunk = make_chunked_sigmoid(6)
    x = torch.randn(2, 47)  # 47 = 7*6 + 5
    out = chunk(x)
    assert out.shape == x.shape
    # Still equivalent to global sigmoid since sigmoid is element-wise
    assert torch.allclose(out, torch.sigmoid(x), atol=1e-7)


def test_chunked_sigmoid_invalid_chunk_size_raises() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        make_chunked_sigmoid(0)


def test_chunked_sigmoid_preserves_gradient_flow() -> None:
    chunk = make_chunked_sigmoid(6)
    x = torch.randn(2, 12, requires_grad=True)
    out = chunk(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# FirstOrderMLP — structure
# ---------------------------------------------------------------------------


def test_blindsight_dims() -> None:
    """Paper Blindsight: input=output=100, hidden=40 (D12.3)."""
    mlp = FirstOrderMLP(input_dim=100, hidden_dim=40, decoder_activation=global_sigmoid)
    x = torch.randn(8, 100)
    h1, h2 = mlp(x)
    assert h1.shape == (8, 40)
    assert h2.shape == (8, 100)


def test_agl_dims() -> None:
    """Paper AGL: input=output=48, hidden=40, chunked sigmoid."""
    mlp = FirstOrderMLP(input_dim=48, hidden_dim=40, decoder_activation=make_chunked_sigmoid(6))
    x = torch.randn(8, 48)
    h1, h2 = mlp(x)
    assert h1.shape == (8, 40)
    assert h2.shape == (8, 48)


def test_no_bias_anywhere() -> None:
    """Per paper / student: no bias on encoder or decoder."""
    mlp = FirstOrderMLP(input_dim=100, hidden_dim=40, decoder_activation=global_sigmoid)
    assert mlp.fc1.bias is None
    assert mlp.fc2.bias is None


def test_weight_init_in_default_range() -> None:
    """Default weight_init_range=(-1, 1) — all weights must respect it."""
    mlp = FirstOrderMLP(input_dim=200, hidden_dim=100, decoder_activation=global_sigmoid)
    for w in (mlp.fc1.weight, mlp.fc2.weight):
        assert (w >= -1.0).all()
        assert (w <= 1.0).all()


def test_weight_init_custom_range() -> None:
    mlp = FirstOrderMLP(
        input_dim=100,
        hidden_dim=40,
        decoder_activation=global_sigmoid,
        weight_init_range=(-0.05, 0.05),
    )
    for w in (mlp.fc1.weight, mlp.fc2.weight):
        assert (w >= -0.05).all()
        assert (w <= 0.05).all()


def test_invalid_dims_raise() -> None:
    with pytest.raises(ValueError, match="≥ 1"):
        FirstOrderMLP(input_dim=0, hidden_dim=40, decoder_activation=global_sigmoid)
    with pytest.raises(ValueError, match="≥ 1"):
        FirstOrderMLP(input_dim=100, hidden_dim=0, decoder_activation=global_sigmoid)


def test_invalid_weight_init_range_raises() -> None:
    with pytest.raises(ValueError, match="lower < upper"):
        FirstOrderMLP(
            input_dim=100,
            hidden_dim=40,
            decoder_activation=global_sigmoid,
            weight_init_range=(1.0, 1.0),
        )


# ---------------------------------------------------------------------------
# FirstOrderMLP — cascade asymmetry
# ---------------------------------------------------------------------------


def test_cascade_rate_1_no_effect_on_h2() -> None:
    """cascade_rate=1.0 collapses to fresh decoder output (cascade no-op)."""
    torch.manual_seed(0)
    mlp = FirstOrderMLP(input_dim=100, hidden_dim=40, decoder_activation=global_sigmoid)
    mlp.eval()

    x = torch.randn(2, 100)
    _, h2_fresh = mlp(x, prev_h2=None, cascade_rate=1.0)
    _, h2_with_prev = mlp(x, prev_h2=torch.zeros(2, 100), cascade_rate=1.0)
    # α=1 → out = 1·new + 0·prev = new (prev ignored)
    assert torch.equal(h2_fresh, h2_with_prev)


def test_cascade_on_h2_changes_output_when_prev_differs() -> None:
    """Different prev_h2 with α < 1 produces different h2."""
    torch.manual_seed(0)
    mlp = FirstOrderMLP(input_dim=100, hidden_dim=40, decoder_activation=global_sigmoid)
    mlp.eval()

    x = torch.randn(2, 100)
    _, h2_a = mlp(x, prev_h2=torch.zeros(2, 100), cascade_rate=0.5)
    _, h2_b = mlp(x, prev_h2=torch.ones(2, 100), cascade_rate=0.5)
    assert not torch.allclose(h2_a, h2_b)


def test_prev_h1_silently_ignored() -> None:
    """Passing prev_h1 must not crash — it's accepted-but-unused."""
    torch.manual_seed(0)
    mlp = FirstOrderMLP(input_dim=100, hidden_dim=40, decoder_activation=global_sigmoid)
    mlp.eval()

    x = torch.randn(2, 100)
    _, h2_without = mlp(x, prev_h1=None, prev_h2=None, cascade_rate=0.5)

    torch.manual_seed(0)
    mlp2 = FirstOrderMLP(input_dim=100, hidden_dim=40, decoder_activation=global_sigmoid)
    mlp2.eval()
    _, h2_with = mlp2(x, prev_h1=torch.ones(2, 40), prev_h2=None, cascade_rate=0.5)
    # prev_h1 is ignored → same h2
    assert torch.equal(h2_without, h2_with)


def test_gradient_flow_through_forward() -> None:
    mlp = FirstOrderMLP(input_dim=100, hidden_dim=40, decoder_activation=global_sigmoid)
    mlp.eval()
    x = torch.randn(2, 100, requires_grad=True)
    h1, h2 = mlp(x, cascade_rate=0.5)
    (h1.sum() + h2.sum()).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert mlp.fc1.weight.grad is not None
    assert mlp.fc2.weight.grad is not None
