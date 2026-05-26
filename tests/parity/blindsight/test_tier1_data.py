"""Tier 1 parity: :func:`maps.domains.blindsight.data.generate_patterns`
vs verbatim student.

Bit-exact match (1e-6) when the same seeds are set before each call.
The RNG consumption order is the parity-critical detail (np.random
global + random.randint).
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from maps.domains.blindsight.data import ConditionParams, generate_patterns
from tests.parity.blindsight._student_extracts import student_generate_patterns


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Paper §3.2 condition parameters
_PAPER_PARAMS = {
    0: ConditionParams(0.0, 0.0, 1.0),  # superthreshold
    1: ConditionParams(0.02, 0.1, 1.0),  # subthreshold (noise_level=0.1)
    2: ConditionParams(0.02, 0.1, 0.3),  # low_vision
}


@pytest.mark.parametrize("condition", [0, 1, 2])
def test_patterns_bit_exact_vs_student(condition: int) -> None:
    """Same seed → bit-exact patterns tensor between ours & student."""
    n_patterns = 100
    num_units = 100
    noise_level = 0.1

    _seed_all(seed=42)
    p_student, sp_student, _, o2_student = student_generate_patterns(
        patterns_number=n_patterns,
        num_units=num_units,
        factor=1,
        condition=condition,
        noise_level=noise_level,
    )

    _seed_all(seed=42)
    batch = generate_patterns(
        n_patterns=n_patterns, num_units=num_units, params=_PAPER_PARAMS[condition]
    )

    assert torch.allclose(batch.patterns, p_student, atol=1e-6)
    assert torch.allclose(batch.stim_present, sp_student, atol=1e-6)
    assert torch.allclose(batch.order_2_target, o2_student, atol=1e-6)


def test_patterns_different_seed_different_output() -> None:
    """Sanity: seeded differently → different outputs (not a no-op test)."""
    _seed_all(seed=42)
    a = generate_patterns(20, 50, _PAPER_PARAMS[1]).patterns
    _seed_all(seed=99)
    b = generate_patterns(20, 50, _PAPER_PARAMS[1]).patterns
    assert not torch.allclose(a, b, atol=1e-4)
