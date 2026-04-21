"""Unit tests for :class:`ValueNorm` (E.9a)."""

from __future__ import annotations

import numpy as np
import torch

from maps.experiments.marl.valuenorm import ValueNorm


def test_valuenorm_initial_state_is_zero():
    vn = ValueNorm(1)
    assert torch.all(vn.running_mean == 0)
    assert torch.all(vn.running_mean_sq == 0)
    assert vn.debiasing_term.item() == 0


def test_valuenorm_update_changes_running_stats():
    vn = ValueNorm(1, beta=0.9)
    data = torch.arange(100, dtype=torch.float32).unsqueeze(-1)
    vn.update(data)
    assert vn.debiasing_term.item() > 0
    assert torch.abs(vn.running_mean).sum() > 0


def test_valuenorm_normalize_then_denormalize_roundtrip():
    """After several updates, denormalize(normalize(x)) ≈ x."""
    vn = ValueNorm(1, beta=0.9)
    torch.manual_seed(0)
    # Train the normalizer on a few batches.
    for _ in range(20):
        data = torch.randn(64, 1) * 5 + 10  # mean=10, std=5
        vn.update(data)

    x = torch.tensor([[8.0], [10.0], [12.0]])
    n = vn.normalize(x)
    d_np = vn.denormalize(n)
    d = torch.from_numpy(d_np).float()
    assert torch.allclose(x, d, atol=1e-4)


def test_valuenorm_normalize_reduces_variance():
    vn = ValueNorm(1, beta=0.5)
    torch.manual_seed(0)
    data = torch.randn(1000, 1) * 10 + 5
    for _ in range(5):
        vn.update(data)
    normed = vn.normalize(data)
    # After normalization, variance should be much closer to 1 than 100.
    assert abs(normed.var().item() - 1.0) < 2.0


def test_valuenorm_accepts_numpy_and_tensor():
    vn = ValueNorm(1, beta=0.9)
    np_data = np.arange(50, dtype=np.float32).reshape(50, 1)
    vn.update(np_data)  # should not raise
    assert vn.debiasing_term.item() > 0
