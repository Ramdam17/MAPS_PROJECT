"""MAPS core math — paper §2.1: cascade, second-order, losses."""

from maps.core.cascade import cascade_update, n_iterations_from_alpha
from maps.core.losses import (
    cae_loss,
    simclr_loss,
    wagering_bce_loss,
    weight_regularization,
)

__all__ = [
    "cae_loss",
    "cascade_update",
    "n_iterations_from_alpha",
    "simclr_loss",
    "wagering_bce_loss",
    "weight_regularization",
]
