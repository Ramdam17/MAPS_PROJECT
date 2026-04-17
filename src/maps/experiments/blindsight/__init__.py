"""Blindsight experiment — perceptual detection under noise.

Vargas et al. (TMLR), §3.2. Adapted from Pasquali & Cleeremans (2010):
the first-order network maps a noisy 100-d stimulus to a 100-d reconstruction;
the second-order network wagers on whether a stimulus was actually present.

Three stimulus regimes (Weiskrantz 1986 "blindsight" phenomenology):

- **Superthreshold** — clean stimulus, no additive noise.
- **Subthreshold**   — stimulus below perceptual threshold, embedded in noise.
- **Low vision**     — stimulus scaled down (×0.3) + embedded in noise.
"""

from maps.experiments.blindsight.data import (
    ConditionParams,
    StimulusCondition,
    TrainingBatch,
    generate_patterns,
)
from maps.experiments.blindsight.trainer import BlindsightSetting, BlindsightTrainer

__all__ = [
    "BlindsightSetting",
    "BlindsightTrainer",
    "ConditionParams",
    "StimulusCondition",
    "TrainingBatch",
    "generate_patterns",
]
