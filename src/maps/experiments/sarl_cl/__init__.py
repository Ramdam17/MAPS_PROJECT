"""SARL + Continual Learning (SARL+CL).

Faithful port of ``SARL_CL/examples_cl/maps.py`` — the paper's continual-
learning extension to SARL. Critically, the CL branch uses **different**
network architectures from the standard SARL domain (see
:mod:`maps.experiments.sarl_cl.model`):

* ``SarlCLQNetwork`` has an explicit ``fc_output`` decoder (not tied weights)
  and applies the cascade to its output rather than to the hidden layer.
* ``SarlCLSecondOrderNetwork`` has an explicit ``comparison_layer`` before
  the wager head.
* ``AdaptiveQNetwork`` handles variable input-channel counts for cross-game
  transfer.

This architectural difference is preserved from the paper's code — do not
refactor it to reuse :mod:`maps.experiments.sarl` modules; doing so would
silently change the model the CL experiments actually ran.

Package layout
--------------
* :mod:`maps.experiments.sarl_cl.model` — CL-specific networks.
* :mod:`maps.experiments.sarl_cl.loss_weighting` — dynamic per-task loss
  weighter + helpers (min-max norm, per-sample loss).
* :mod:`maps.experiments.sarl_cl.trainer` — ``sarl_cl_update_step`` (4.6b).
* :mod:`maps.experiments.sarl_cl.training_loop` — ``run_training_cl`` (4.6b).

Shared components imported from :mod:`maps.components.losses`:

* ``weight_regularization`` — EWC-style L2 drift anchor (the paper's
  ``compute_weight_regularization``).
* ``distillation_loss`` — KD loss (carried over for future teacher-student
  extensions; note that the paper currently uses *only* weight regularization
  and feature MSE, not soft-target distillation, despite the "distillation"
  label in its dictionary keys).
"""

from __future__ import annotations

from maps.experiments.sarl_cl.loss_weighting import (
    DynamicLossWeighter,
    individual_losses,
    min_max_norm,
    update_moving_average,
)
from maps.experiments.sarl_cl.model import (
    AdaptiveQNetwork,
    SarlCLQNetwork,
    SarlCLSecondOrderNetwork,
)

__all__ = [
    "AdaptiveQNetwork",
    "DynamicLossWeighter",
    "SarlCLQNetwork",
    "SarlCLSecondOrderNetwork",
    "individual_losses",
    "min_max_norm",
    "update_moving_average",
]
