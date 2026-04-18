"""SARL (MinAtar DQN) training and evaluation.

This package ports ``external/MinAtar/examples/maps.py`` (paper's SARL entry
point) into the ``src/maps/`` layout.

Sprint 04b scope:
* ``model`` — ``SarlQNetwork`` + ``SarlSecondOrderNetwork`` (architectural parity)
* ``data`` — replay buffer + transition sampler (Tier 2 parity)
* ``trainer`` — DQN update rule (Tier 3 parity)
* ``evaluate`` — rollout metrics
"""

from __future__ import annotations

from maps.experiments.sarl.model import (
    NUM_LINEAR_UNITS,
    SarlQNetwork,
    SarlSecondOrderNetwork,
)

__all__ = [
    "NUM_LINEAR_UNITS",
    "SarlQNetwork",
    "SarlSecondOrderNetwork",
]
