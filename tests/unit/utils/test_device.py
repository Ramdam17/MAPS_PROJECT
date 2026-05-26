"""Unit tests for :mod:`maps.utils.device`."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from maps.utils.device import get_device, pick_best_available


def test_pick_best_returns_torch_device() -> None:
    device = pick_best_available()
    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "mps", "cuda"}


def test_get_device_auto_matches_pick_best() -> None:
    assert get_device("auto") == pick_best_available()


def test_get_device_cpu_always_works() -> None:
    """CPU is the universal fallback — must always succeed."""
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_invalid_raises() -> None:
    with pytest.raises(ValueError, match="unknown device preference"):
        get_device("xpu")  # type: ignore[arg-type]


def test_get_device_mps_when_unavailable_falls_back_to_cpu() -> None:
    """If MPS is unavailable, prefer='mps' falls back to CPU with warning."""
    with (
        patch("maps.utils.device.torch.cuda.is_available", return_value=False),
        patch.object(torch.backends, "mps", create=True) as mock_mps_attr,
    ):
        mock_mps_attr.is_available.return_value = False
        device = get_device("mps")
        assert device == torch.device("cpu")


def test_get_device_cuda_when_unavailable_falls_back_to_cpu() -> None:
    """If CUDA is unavailable, prefer='cuda' falls back to CPU."""
    with patch("maps.utils.device.torch.cuda.is_available", return_value=False):
        device = get_device("cuda")
        assert device == torch.device("cpu")


def test_pick_best_prefers_cuda_over_mps() -> None:
    """When CUDA is faked available, it wins over MPS."""
    with patch("maps.utils.device.torch.cuda.is_available", return_value=True):
        device = pick_best_available()
        assert device == torch.device("cuda")
