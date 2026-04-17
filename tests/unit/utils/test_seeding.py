"""Unit tests for maps.utils.seeding — deterministic behavior."""

from __future__ import annotations

import os
import random

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
