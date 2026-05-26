"""Unit tests for :mod:`maps.domains.blindsight.data`."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from maps.domains.blindsight.data import (
    ConditionParams,
    StimulusCondition,
    TrainingBatch,
    generate_patterns,
)

# ---------------------------------------------------------------------------
# Test fixtures : the 3 paper conditions
# ---------------------------------------------------------------------------


SUPERTHRESHOLD = ConditionParams(random_limit=0.0, baseline=0.0, multiplier=1.0)
SUBTHRESHOLD = ConditionParams(random_limit=0.02, baseline=0.1, multiplier=1.0)
LOW_VISION = ConditionParams(random_limit=0.02, baseline=0.1, multiplier=0.3)


# ---------------------------------------------------------------------------
# StimulusCondition enum
# ---------------------------------------------------------------------------


def test_stimulus_condition_values() -> None:
    """Paper §3.2 condition codes : 0/1/2."""
    assert StimulusCondition.SUPERTHRESHOLD == 0
    assert StimulusCondition.SUBTHRESHOLD == 1
    assert StimulusCondition.LOW_VISION == 2


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------


def test_output_shapes() -> None:
    batch = generate_patterns(n_patterns=100, num_units=100, params=SUPERTHRESHOLD)
    assert isinstance(batch, TrainingBatch)
    assert batch.patterns.shape == (100, 100)
    assert batch.stim_present.shape == (100, 100)
    assert batch.order_2_target.shape == (100, 2)


def test_output_dtypes_are_float32() -> None:
    batch = generate_patterns(n_patterns=10, num_units=100, params=SUPERTHRESHOLD)
    assert batch.patterns.dtype == torch.float32
    assert batch.stim_present.dtype == torch.float32
    assert batch.order_2_target.dtype == torch.float32


def test_all_tensors_require_grad() -> None:
    """Student behaviour preserved (dead code on 2 of 3 — see docstring)."""
    batch = generate_patterns(n_patterns=10, num_units=100, params=SUPERTHRESHOLD)
    assert batch.patterns.requires_grad
    assert batch.stim_present.requires_grad
    assert batch.order_2_target.requires_grad


# ---------------------------------------------------------------------------
# Half-and-half split semantics
# ---------------------------------------------------------------------------


def test_first_half_is_noise_only() -> None:
    """First N//2 rows have no stim_present and order_2_target = [0, 1]."""
    n = 20
    batch = generate_patterns(n_patterns=n, num_units=100, params=SUPERTHRESHOLD)
    half = n // 2

    # First half : stim_present rows are all zeros
    assert torch.equal(
        batch.stim_present[:half],
        torch.zeros(half, 100),
    )
    # First half : order_2_target rows are all [0, 1] (low wager)
    expected = torch.tensor([[0.0, 1.0]] * half)
    assert torch.equal(batch.order_2_target[:half].detach(), expected)


def test_second_half_has_at_most_one_stim_per_row() -> None:
    """Stimulus-present rows mark a single unit (the chosen stim_idx)."""
    n = 20
    batch = generate_patterns(n_patterns=n, num_units=100, params=SUPERTHRESHOLD)
    half = n // 2

    # Second half : each row has 0 or 1 unit marked (1 if above threshold)
    n_nonzero_per_row = (batch.stim_present[half:] != 0).sum(dim=1)
    assert (n_nonzero_per_row <= 1).all()


def test_threshold_logic_high_multiplier_gives_mostly_high_wager() -> None:
    """With multiplier=1.0 and stim_idx getting U(0, 1)·1, ~50% should be
    above multiplier/2 = 0.5 → high wager."""
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    n = 200  # big sample for stable statistics
    batch = generate_patterns(n_patterns=n, num_units=100, params=SUPERTHRESHOLD)
    half = n // 2

    # Count high-wager rows in second half : order_2_target == [1, 0]
    is_high_wager = (batch.order_2_target[half:, 0] == 1.0).detach()
    n_high = is_high_wager.sum().item()
    n_stim_rows = n - half

    # Expect ~50% high wager (median of U(0,1) = 0.5)
    proportion = n_high / n_stim_rows
    assert 0.3 < proportion < 0.7, (
        f"Expected ~50% high wager, got {proportion:.1%} ({n_high}/{n_stim_rows})"
    )


# ---------------------------------------------------------------------------
# Device placement
# ---------------------------------------------------------------------------


def test_device_placement_cpu() -> None:
    batch = generate_patterns(n_patterns=10, num_units=100, params=SUPERTHRESHOLD, device="cpu")
    assert batch.patterns.device.type == "cpu"
    assert batch.stim_present.device.type == "cpu"
    assert batch.order_2_target.device.type == "cpu"


# ---------------------------------------------------------------------------
# ConditionParams
# ---------------------------------------------------------------------------


def test_condition_params_is_frozen() -> None:
    """ConditionParams is `@dataclass(frozen=True)`."""
    from dataclasses import FrozenInstanceError

    params = SUPERTHRESHOLD
    with pytest.raises(FrozenInstanceError):
        params.multiplier = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Determinism — same seed → same batch
# ---------------------------------------------------------------------------


def test_seeded_generation_is_reproducible() -> None:
    """Re-seeding the RNGs identically produces bit-exact same batch."""

    def _gen() -> torch.Tensor:
        torch.manual_seed(7)
        np.random.seed(7)
        random.seed(7)
        return generate_patterns(n_patterns=20, num_units=50, params=SUBTHRESHOLD).patterns

    p1 = _gen()
    p2 = _gen()
    assert torch.equal(p1, p2)


def test_subthreshold_pattern_includes_baseline() -> None:
    """In SUBTHRESHOLD (baseline=0.1), even noise-only patterns should
    have values ≥ 0.1 (just baseline + small uniform noise)."""
    batch = generate_patterns(n_patterns=10, num_units=100, params=SUBTHRESHOLD)
    half = 5
    # Noise-only rows : pattern = multiplier·U(0, 0.02) + 0.1 → in [0.1, 0.12]
    noise_rows = batch.patterns[:half].detach()
    assert (noise_rows >= 0.099).all()  # small float tolerance
    assert (noise_rows <= 0.121).all()
