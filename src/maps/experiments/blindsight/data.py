"""Pattern generation for the Blindsight task.

Ported verbatim from BLINDSIGHT/Blindsight_TMLR.py (`generate_patterns`,
lines 259-328). The numerical behavior must stay bit-identical to the
reference so pre-training losses are reproducible (see
tests/parity/test_blindsight_pretrain.py).

Each batch contains `patterns_number` trials, split evenly between
"noise-only" (first half) and "stimulus-present" (second half).
A stimulus is considered *detected* when its intensity exceeds half the
condition-specific multiplier — the wagering target tracks this rule,
not the ground-truth presence label.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch

__all__ = [
    "ConditionParams",
    "StimulusCondition",
    "TrainingBatch",
    "generate_patterns",
]


class StimulusCondition(IntEnum):
    """The three stimulus regimes from the paper (§3.2).

    The integer values match the reference code's `condition` arg, so you
    can pass either the enum or the raw int.
    """

    SUPERTHRESHOLD = 0  # clean stimulus, no noise
    SUBTHRESHOLD = 1  # stimulus below threshold, noise baseline
    LOW_VISION = 2  # stimulus scaled down, noise baseline


@dataclass(frozen=True)
class ConditionParams:
    """Per-regime stimulus parameters.

    The canonical paper values are declared in `config/env/blindsight.yaml`
    and loaded by `BlindsightTrainer`. This module does **not** hardcode them.
    """

    random_limit: float
    baseline: float
    multiplier: float


@dataclass
class TrainingBatch:
    """One epoch's worth of Blindsight training patterns.

    Attributes
    ----------
    patterns : Tensor, shape (N, num_units)
        The raw input patterns (noisy or stimulus-bearing).
    stim_present : Tensor, shape (N, num_units)
        One-hot encoding of the target stimulus location (zeros for noise trials
        or subthreshold stimuli that did not cross the detection threshold).
    order_2_target : Tensor, shape (N, 2)
        Wagering targets as 2-unit softmax: [high_wager, low_wager].
        For 1-unit sigmoid heads, use `order_2_target[:, 0]`.
    """

    patterns: torch.Tensor
    stim_present: torch.Tensor
    order_2_target: torch.Tensor


def generate_patterns(
    *,
    params: ConditionParams,
    patterns_number: int,
    num_units: int,
    factor: int = 1,
    device: torch.device | str = "cpu",
    rng: np.random.Generator | None = None,
) -> TrainingBatch:
    """Generate one training batch.

    Parameters
    ----------
    params : ConditionParams
        Stimulus-regime parameters (random_limit, baseline, multiplier).
        Normally built from `config/env/blindsight.yaml`; see
        `BlindsightTrainer` for the wiring.
    patterns_number : int
        Number of trials before the `factor` multiplier.
    num_units : int
        Stimulus dimensionality (100 in the paper).
    factor : int, default 1
        Data-augmentation multiplier — ``actual_N = patterns_number * factor``.
    device : torch.device | str
        Where to place the output tensors.
    rng : np.random.Generator, optional
        If provided, used for NumPy draws. Python's ``random.randint`` still
        consults the global state — match the reference code, which uses
        ``random.randint`` directly and expects ``random.seed`` to have been
        set by the caller (typically via ``maps.utils.seeding.set_all_seeds``).

    Returns
    -------
    TrainingBatch
    """
    import random  # local import — stdlib, matches reference behavior

    n = int(patterns_number * factor)

    np_rng = rng or np.random  # fall back to global state for parity

    patterns: list[np.ndarray] = []
    stim_present: list[np.ndarray] = []
    order_2_pr: list[list[float]] = []

    for i in range(n):
        if i < n // 2:
            # Noise-only trial.
            pattern = (
                params.multiplier * np_rng.uniform(0.0, params.random_limit, num_units)
                + params.baseline
            )
            patterns.append(pattern)
            stim_present.append(np.zeros(num_units))
            order_2_pr.append([0.0, 1.0])  # no stimulus → low wager
        else:
            # Stimulus-present trial.
            stim_idx = random.randint(0, num_units - 1)
            pattern = np_rng.uniform(0.0, params.random_limit, num_units) + params.baseline
            pattern[stim_idx] = np_rng.uniform(0.0, 1.0) * params.multiplier
            patterns.append(pattern)

            present = np.zeros(num_units)
            if pattern[stim_idx] >= params.multiplier / 2:
                order_2_pr.append([1.0, 0.0])  # above threshold → high wager
                present[stim_idx] = 1.0
            else:
                order_2_pr.append([0.0, 1.0])  # below threshold → low wager
            stim_present.append(present)

    patterns_t = torch.tensor(np.asarray(patterns), dtype=torch.float32, device=device)
    stim_present_t = torch.tensor(np.asarray(stim_present), dtype=torch.float32, device=device)
    order_2_t = torch.tensor(order_2_pr, dtype=torch.float32, device=device)

    # The reference calls `.requires_grad_(True)` on these; replicate for parity,
    # even though only `patterns` actually feeds into an autograd path.
    patterns_t.requires_grad_(True)
    stim_present_t.requires_grad_(True)
    order_2_t.requires_grad_(True)

    return TrainingBatch(
        patterns=patterns_t,
        stim_present=stim_present_t,
        order_2_target=order_2_t,
    )
