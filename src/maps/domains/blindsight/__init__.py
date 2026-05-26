"""Blindsight perceptual domain — paper §3, Table 5a.

- :mod:`maps.domains.blindsight.data` — stimulus pattern generation.
- :mod:`maps.domains.blindsight.augmentations` — SimCLR positive-pair
  augmentation (D12.4).
- :mod:`maps.domains.blindsight.trainer` — :class:`BlindsightTrainer`
  + :data:`SETTINGS_REGISTRY` (Sprint 12.E).
- :mod:`maps.domains.blindsight.cli` — Typer CLI (Sprint 12.F).
"""

from maps.domains.blindsight.augmentations import bit_flip
from maps.domains.blindsight.data import (
    ConditionParams,
    StimulusCondition,
    TrainingBatch,
    generate_patterns,
)
from maps.domains.blindsight.trainer import (
    SETTINGS_REGISTRY,
    BlindsightSetting,
    BlindsightTrainer,
    EvalMetrics,
    TrainingMetrics,
)

__all__ = [
    "SETTINGS_REGISTRY",
    "BlindsightSetting",
    "BlindsightTrainer",
    "ConditionParams",
    "EvalMetrics",
    "StimulusCondition",
    "TrainingBatch",
    "TrainingMetrics",
    "bit_flip",
    "generate_patterns",
]
