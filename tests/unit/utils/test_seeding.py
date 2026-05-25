"""Unit tests for :mod:`maps.utils.seeding` — deterministic behavior.

These tests intentionally do NOT use the autouse ``seed_everything``
fixture's pre-seeded state — each test re-seeds explicitly so that the
seeding behaviour itself is what's under test.
"""

from __future__ import annotations

import os
import random
from unittest.mock import call, patch

import numpy as np
import pytest
import torch

from maps.utils.seeding import LAB_DEFAULT_SEED, set_all_seeds


def test_python_random_is_seeded() -> None:
    set_all_seeds(42)
    a = [random.random() for _ in range(5)]
    set_all_seeds(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_numpy_is_seeded() -> None:
    set_all_seeds(42)
    a = np.random.randn(10)
    set_all_seeds(42)
    b = np.random.randn(10)
    assert np.array_equal(a, b)


def test_torch_cpu_is_seeded() -> None:
    set_all_seeds(42)
    a = torch.randn(10)
    set_all_seeds(42)
    b = torch.randn(10)
    assert torch.equal(a, b)


def test_pythonhashseed_env_is_set() -> None:
    set_all_seeds(1234)
    assert os.environ["PYTHONHASHSEED"] == "1234"


def test_different_seeds_give_different_draws() -> None:
    set_all_seeds(42)
    a = torch.randn(10)
    set_all_seeds(43)
    b = torch.randn(10)
    assert not torch.equal(a, b)


def test_negative_seed_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_all_seeds(-1)


def test_seed_zero_is_allowed() -> None:
    set_all_seeds(0)  # must not raise — zero is a legal seed


def test_lab_default_constant_is_42() -> None:
    """``config/maps.yaml`` ships with seed=42 — this constant must
    mirror it. Drift here = silent reproducibility regression."""
    assert LAB_DEFAULT_SEED == 42


def test_cuda_branch_fires_when_available() -> None:
    """If CUDA were available, ``cuda.manual_seed_all`` is invoked
    with the requested seed.

    Apple-silicon dev boxes never have CUDA, so we mock
    ``is_available`` + ``manual_seed_all`` to exercise the branch
    without a GPU. Note: ``torch.manual_seed()`` itself may also call
    ``cuda.manual_seed_all`` internally when CUDA is reported
    available, so we check "called WITH the right seed" rather than
    "called once".
    """
    with (
        patch("maps.utils.seeding.torch.cuda.is_available", return_value=True),
        patch("maps.utils.seeding.torch.cuda.manual_seed_all") as mock_cuda_seed,
    ):
        set_all_seeds(1234)

    assert call(1234) in mock_cuda_seed.call_args_list
