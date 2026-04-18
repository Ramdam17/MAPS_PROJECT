"""Unit tests for maps.utils.seeding — deterministic behavior."""

from __future__ import annotations

import os
import random
from unittest.mock import patch

import numpy as np
import torch

from maps.utils.seeding import set_all_seeds


def test_python_random_is_seeded():
    set_all_seeds(42)
    a = [random.random() for _ in range(5)]
    set_all_seeds(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_numpy_is_seeded():
    set_all_seeds(42)
    a = np.random.randn(10)
    set_all_seeds(42)
    b = np.random.randn(10)
    assert np.array_equal(a, b)


def test_torch_cpu_is_seeded():
    set_all_seeds(42)
    a = torch.randn(10)
    set_all_seeds(42)
    b = torch.randn(10)
    assert torch.equal(a, b)


def test_pythonhashseed_env_is_set():
    set_all_seeds(1234)
    assert os.environ["PYTHONHASHSEED"] == "1234"


def test_different_seeds_give_different_draws():
    set_all_seeds(42)
    a = torch.randn(10)
    set_all_seeds(43)
    b = torch.randn(10)
    assert not torch.equal(a, b)


def test_cuda_branch_seeds_when_available():
    """If CUDA were available, cuda.manual_seed_all + cudnn flags fire.

    Apple-silicon CI and Mac dev boxes never have CUDA, so we mock the
    is_available check + the CUDA API to exercise the branch without
    needing a GPU.

    Note: torch.manual_seed() itself also calls cuda.manual_seed_all() when
    CUDA is reported as available, so we check "called with the right seed"
    rather than "called once" (the internal call may double the count).
    """
    from unittest.mock import call

    with (
        patch("maps.utils.seeding.torch.cuda.is_available", return_value=True),
        patch("maps.utils.seeding.torch.cuda.manual_seed_all") as mock_cuda_seed,
        patch.object(torch.backends, "cudnn") as mock_cudnn,
    ):
        set_all_seeds(1234, deterministic_cudnn=True)

    assert call(1234) in mock_cuda_seed.call_args_list
    assert mock_cudnn.deterministic is True
    assert mock_cudnn.benchmark is False


def test_cuda_branch_respects_deterministic_cudnn_flag():
    """With `deterministic_cudnn=False`, cudnn flags are not touched."""
    from unittest.mock import call

    with (
        patch("maps.utils.seeding.torch.cuda.is_available", return_value=True),
        patch("maps.utils.seeding.torch.cuda.manual_seed_all") as mock_cuda_seed,
        patch.object(torch.backends, "cudnn") as mock_cudnn,
    ):
        # Save sentinel values so we can verify mock wasn't mutated.
        mock_cudnn.deterministic = "UNTOUCHED"
        mock_cudnn.benchmark = "UNTOUCHED"
        set_all_seeds(99, deterministic_cudnn=False)

    assert call(99) in mock_cuda_seed.call_args_list
    assert mock_cudnn.deterministic == "UNTOUCHED"
    assert mock_cudnn.benchmark == "UNTOUCHED"
