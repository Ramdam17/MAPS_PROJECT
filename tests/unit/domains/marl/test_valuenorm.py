"""Unit tests for MARL ValueNorm (Sprint 16.D)."""

from __future__ import annotations

import numpy as np
import torch

from maps.domains.marl.valuenorm import ValueNorm


def test_normalize_denormalize_round_trip():
    vn = ValueNorm(1, beta=0.9)
    vn.update(torch.randn(256, 1) * 5.0 + 3.0)
    x = torch.randn(16, 1)
    normalized = vn.normalize(x)
    recovered = torch.from_numpy(vn.denormalize(normalized)).float()
    assert torch.allclose(recovered, x, atol=1e-4)


def test_update_tracks_running_mean():
    vn = ValueNorm(1, beta=0.9)
    for _ in range(500):
        vn.update(torch.full((64, 1), 7.0))
    mean, _var = vn.running_mean_var()
    assert torch.allclose(mean, torch.tensor([7.0]), atol=1e-2)


def test_denormalize_returns_numpy():
    vn = ValueNorm(1, beta=0.9)
    vn.update(torch.randn(64, 1))
    out = vn.denormalize(torch.zeros(4, 1))
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 1)


def test_untrained_variance_floored():
    """Before any update the debiased variance is floored (no div-by-zero blow-up)."""
    vn = ValueNorm(1)
    _mean, var = vn.running_mean_var()
    assert (var >= 1e-2).all()
