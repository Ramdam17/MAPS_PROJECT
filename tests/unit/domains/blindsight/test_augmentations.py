"""Unit tests for :mod:`maps.domains.blindsight.augmentations` (D12.4)."""

from __future__ import annotations

import pytest
import torch

from maps.domains.blindsight.augmentations import bit_flip


def test_p_zero_returns_unchanged_clone() -> None:
    """p=0 → identical values, but a fresh tensor object (not aliased)."""
    x = torch.rand(8, 100)
    out = bit_flip(x, p=0.0)
    assert torch.equal(out, x)
    assert out is not x  # cloned, not aliased


def test_p_one_replaces_every_element() -> None:
    """p=1 → every element replaced ; out and x almost certainly differ
    in every position (probability of accidental match ≈ 0)."""
    torch.manual_seed(0)
    x = torch.zeros(8, 100)  # all zeros
    out = bit_flip(x, p=1.0)
    # noise ∈ [0, 1] ; probability that a particular U(0,1) draw == 0 ≈ 0
    assert (out != x).all()
    # But shape and dtype preserved
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_p_default_is_0_1() -> None:
    """D12.4 default : p=0.1."""
    torch.manual_seed(42)
    x = torch.zeros(1000)
    out = bit_flip(x)  # no p → default 0.1
    n_changed = (out != x).sum().item()
    # Expect ~100 changed elements out of 1000 ; bound generously
    assert 70 < n_changed < 130


def test_shape_and_dtype_preserved() -> None:
    for shape in [(100,), (8, 100), (4, 8, 16)]:
        x = torch.rand(*shape)
        out = bit_flip(x, p=0.3)
        assert out.shape == shape
        assert out.dtype == x.dtype


def test_invalid_p_raises() -> None:
    x = torch.rand(10)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bit_flip(x, p=-0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bit_flip(x, p=1.1)


def test_generator_is_reproducible() -> None:
    """Same seeded generator → same augmentation."""
    x = torch.zeros(100)
    gen_a = torch.Generator().manual_seed(13)
    gen_b = torch.Generator().manual_seed(13)
    out_a = bit_flip(x, p=0.5, generator=gen_a)
    out_b = bit_flip(x, p=0.5, generator=gen_b)
    assert torch.equal(out_a, out_b)


def test_in_range_after_augmentation_when_input_in_unit_interval() -> None:
    """Replacement values ∈ [0, 1]. If input ∈ [0, 1] too, output ∈ [0, 1]."""
    x = torch.rand(8, 100)  # [0, 1)
    out = bit_flip(x, p=0.5)
    assert (out >= 0).all() and (out <= 1).all()


def test_does_not_modify_input_inplace() -> None:
    """bit_flip operates on a copy ; original x is untouched."""
    x = torch.rand(100)
    x_before = x.clone()
    _ = bit_flip(x, p=0.5)
    assert torch.equal(x, x_before)
