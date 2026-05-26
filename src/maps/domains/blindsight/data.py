"""Blindsight stimulus pattern generation — MAPS paper §3.2.

The task simulates Weiskrantz's (1986) blindsight phenomenon : a
first-order encoder/decoder learns to reconstruct stimulus patterns
while a second-order wager network predicts whether the stimulus is
"above threshold" (detectable).

Three regimes (paper §3.2) :

- **SUPERTHRESHOLD** (0) : clean stimulus, no additive noise.
- **SUBTHRESHOLD** (1) : stimulus buried in baseline noise.
- **LOW_VISION** (2) : stimulus scaled down ×0.3 + noise baseline.

The "blindsight effect" emerges from the threshold logic :
``order_2_target`` follows "detected or not" (amplitude
``≥ multiplier/2``), **not** the ground-truth presence label. When the
amplitude is just below threshold, stimulus is present but the
second-order target says "low wager" — this is the conscious /
unconscious dissociation MAPS tries to learn.

Port verbatim from
``external/paper_reference/blindsight_tmlr.py:259-328`` modulo :
- API : config-driven via :class:`ConditionParams` instead of hardcoded
  ``if condition == N`` branches.
- Output : :class:`TrainingBatch` dataclass instead of 4-tuple
  (the student's 4th tensor ``stim_absent`` was always empty — dropped
  here).
- Numerics : RNG consumption order preserved bit-exact for Tier 1
  parity (Sprint 12.G).

References
----------
Vargas et al. (2025), MAPS, TMLR submission §3.2.
Weiskrantz, L. (1986). Blindsight: a case study and implications.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class StimulusCondition(IntEnum):
    """Paper §3.2 stimulus regime."""

    SUPERTHRESHOLD = 0
    SUBTHRESHOLD = 1
    LOW_VISION = 2


@dataclass(frozen=True)
class ConditionParams:
    """Generation parameters for one stimulus regime.

    Loaded from ``config/domains/blindsight/env.yaml`` per condition.

    Attributes
    ----------
    random_limit : float
        Upper bound of the per-unit uniform noise draw.
    baseline : float
        Additive DC offset applied across all units.
    multiplier : float
        Scale applied to noise AND signal. For LOW_VISION = 0.3 ;
        otherwise 1.0.
    """

    random_limit: float
    baseline: float
    multiplier: float


@dataclass
class TrainingBatch:
    """Output of :func:`generate_patterns`.

    Attributes
    ----------
    patterns : torch.Tensor
        Shape ``(N, num_units)``. First half = noise-only ; second half =
        noise + injected stimulus at a random unit.
    stim_present : torch.Tensor
        Shape ``(N, num_units)``. One-hot indicator of where the stimulus
        is (zeros for noise-only rows AND for subthreshold-detected
        rows in the stimulus half).
    order_2_target : torch.Tensor
        Shape ``(N, 2)``. ``[1, 0]`` = high wager (stimulus detected) ;
        ``[0, 1]`` = low wager (no stimulus or subthreshold).
    """

    patterns: Tensor
    stim_present: Tensor
    order_2_target: Tensor


def generate_patterns(
    n_patterns: int,
    num_units: int,
    params: ConditionParams,
    *,
    device: torch.device | str = "cpu",
) -> TrainingBatch:
    r"""Generate ``n_patterns`` stimulus patterns under one regime.

    Half of the batch is **noise-only** (no stimulus injected). The
    other half is **stimulus-present** : a random unit receives a
    stimulus amplitude :math:`\sim U(0, 1) \cdot \mathrm{multiplier}`.
    Detection threshold = ``multiplier / 2`` (paper §3.2, also median
    of the amplitude distribution).

    Parameters
    ----------
    n_patterns : int
        Total batch size. Will be split into ``n_patterns // 2`` noise
        rows and the remainder stimulus rows.
    num_units : int
        Pattern dimensionality (Blindsight paper : 100).
    params : ConditionParams
        Per-regime generation parameters
        (``random_limit``, ``baseline``, ``multiplier``).
    device : torch.device or str, optional
        Where to allocate the output tensors. Default ``"cpu"``.

    Returns
    -------
    TrainingBatch
        Dataclass with ``patterns``, ``stim_present``, ``order_2_target``.

    Notes
    -----
    **Parity-critical RNG consumption order** (matches student
    ``Blindsight_TMLR.py:259-328``) :

    1. Per pattern, **first** : ``np.random.uniform(0.0, random_limit,
       num_units)``.
    2. For stimulus-only rows (second half), **then** :
       ``random.randint(0, num_units - 1)`` for the stimulus index,
       **then** ``np.random.uniform(0.0, 1.0)`` for stimulus amplitude.

    The student uses ``np.random`` global state and ``random.randint``
    (Python stdlib). Both are seeded by
    :func:`maps.utils.seeding.set_all_seeds`.

    **Threshold = ``multiplier / 2``** : not explicitly justified in the
    paper. Likely chosen as the median of ``U(0, multiplier)``, giving
    a balanced 50/50 mix of detected vs subthreshold stimuli.

    **``requires_grad_(True)`` on all 3 tensors** preserves student
    behaviour ; only ``patterns`` actually flows into autograd (the two
    targets are dead-code requires_grad — harmless, kept for bit-exact
    parity).
    """
    patterns: list[np.ndarray] = []
    stim_present: list[np.ndarray] = []
    order_2_pr: list[list[float]] = []

    for i in range(n_patterns):
        if i < n_patterns // 2:
            # First half : noise-only pattern.
            pattern = (
                params.multiplier * np.random.uniform(0.0, params.random_limit, num_units)  # noqa: NPY002 — student parity
                + params.baseline
            )
            patterns.append(pattern)
            stim_present.append(np.zeros(num_units))
            order_2_pr.append([0.0, 1.0])  # no stim → low wager
        else:
            # Second half : noise + injected stimulus at a random unit.
            stim_idx = random.randint(0, num_units - 1)
            pattern = (
                np.random.uniform(0.0, params.random_limit, num_units)  # noqa: NPY002 — student parity
                + params.baseline
            )
            pattern[stim_idx] = np.random.uniform(0.0, 1.0) * params.multiplier  # noqa: NPY002 — student parity
            patterns.append(pattern)
            present = np.zeros(num_units)
            # Threshold detection : amplitude >= multiplier/2 → "detected"
            if pattern[stim_idx] >= params.multiplier / 2:
                order_2_pr.append([1.0, 0.0])  # stim above threshold → high wager
                present[stim_idx] = 1.0
            else:
                order_2_pr.append([0.0, 1.0])  # subthreshold → low wager
                present[stim_idx] = 0.0
            stim_present.append(present)

    return TrainingBatch(
        patterns=(torch.from_numpy(np.asarray(patterns)).float().to(device).requires_grad_(True)),
        stim_present=(
            torch.from_numpy(np.asarray(stim_present)).float().to(device).requires_grad_(True)
        ),
        order_2_target=(
            torch.tensor(order_2_pr, dtype=torch.float32, device=device).requires_grad_(True)
        ),
    )
