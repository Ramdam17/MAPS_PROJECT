"""SARL+CL dynamic loss weighting — MAPS §4.

Ported verbatim from ``external/paper_reference/sarl_cl_maps.py`` ``DynamicLossWeighter``
L535-607. Each of the three CL loss components (``task``, ``distillation``,
``feature``) is normalised by its **running historical maximum** so the
components sit on comparable scales; the trainer then combines the normalised
components with the fixed mixing weights (:class:`LossMixingWeights`).

The commented-out ``scale_factors`` path in the source is dead code (not ported).
The distillation *term* itself is the L2 anchor
:func:`maps.core.losses.weight_regularization` (D15.4) — not this weighter.

References
----------
Vargas et al. (2025), MAPS, TMLR submission §4.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_KEYS = ("task", "distillation", "feature")


@dataclass(frozen=True)
class LossMixingWeights:
    """Fixed mixing weights for the 3 CL loss components (task/distill/feature).

    Defaults = the source argparse defaults ``weight{1,2,3} = 40/40/20`` → /100
    → ``0.4 / 0.4 / 0.2`` (D15.5, the faithful value; resolves audit CL-M1).
    """

    task: float = 0.4
    distillation: float = 0.4
    feature: float = 0.2


class DynamicLossWeighter:
    """Normalises each loss component by its running historical maximum.

    Verbatim behaviour of the source ``DynamicLossWeighter``:
    - :meth:`update` records the (detached) current value and grows a per-key
      ``historical_max``;
    - :meth:`weight_losses` returns ``value / (historical_max + eps)`` per key.
    """

    def __init__(self) -> None:
        self.moving_avgs = dict.fromkeys(_KEYS, 1.0)
        self.historical_max = dict.fromkeys(_KEYS, float("-inf"))
        self.historical_max_prev = dict.fromkeys(_KEYS, float("-inf"))
        self.steps = 0
        self.update_interval = 10_000

    def update(self, losses: dict) -> None:
        self.steps += 1
        detached = {
            k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in losses.items()
        }
        for key, value in detached.items():
            value_float = float(value.item() if isinstance(value, torch.Tensor) else value)
            self.moving_avgs[key] = value_float
            if self.steps % self.update_interval == (self.update_interval // 2):
                self.historical_max_prev[key] = value_float
            self.historical_max[key] = max(self.historical_max[key], value_float)

    def weight_losses(self, losses: dict) -> dict:
        epsilon = 1e-16
        return {key: value / (self.historical_max[key] + epsilon) for key, value in losses.items()}

    def get_stats(self) -> dict:
        return {
            "moving_averages": self.moving_avgs.copy(),
            "historical_max": self.historical_max.copy(),
        }
