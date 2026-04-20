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

from maps.experiments.sarl.data import (
    SarlReplayBuffer,
    Transition,
    get_state,
    target_wager,
)
from maps.experiments.sarl.losses import cae_loss
from maps.experiments.sarl.model import (
    NUM_LINEAR_UNITS,
    SarlQNetwork,
    SarlSecondOrderNetwork,
)
from maps.experiments.sarl.trainer import (
    CAE_LAMBDA,
    SarlUpdateOutput,
    sarl_update_step,
)

__all__ = [
    "CAE_LAMBDA",
    "NUM_LINEAR_UNITS",
    "SarlQNetwork",
    "SarlReplayBuffer",
    "SarlSecondOrderNetwork",
    "SarlUpdateOutput",
    "Transition",
    "cae_loss",
    "get_state",
    "sarl_update_step",
    "target_wager",
]
