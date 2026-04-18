"""Dynamic loss weighting for SARL+CL continual-learning training.

Ports ``update_moving_average``, ``min_max_norm``, ``individual_losses``
and the ``DynamicLossWeighter`` class from ``SARL_CL/examples_cl/maps.py``
(lines 510-607).

Purpose
-------
In continual learning, we combine **three** loss terms per forward pass:

* ``task`` — current-task loss (typically CAE on the Q-values).
* ``distillation`` — weight regularization against the teacher checkpoint
  (the key is ``distillation`` for parity with the paper even though the
  loss is actually an L2 anchor; see :func:`maps.components.losses.weight_regularization`).
* ``feature`` — feature-preservation MSE between current and teacher hidden
  activations.

These three terms live on different scales (CAE loss ~1e-2, weight
regularization can grow into 1e+1, feature MSE varies). Naively summing
them would let one term dominate the gradient. The paper's remedy is to
track each term's **historical maximum** over training and divide by it
(plus an epsilon for stability) — producing roughly-comparable scales.

Why not EMA?
------------
Earlier paper code used an EMA for normalization; the final code uses the
**running maximum** instead (see the commented-out lines 579-586 in the
source). We keep the final behavior and flag the abandoned EMA path in
:func:`update_moving_average` — a helper that's still in the module's
public API even though the current weighter never calls it.

References
----------
- SARL_CL/examples_cl/maps.py:510-607 (source).
- Vargas et al. (2025), MAPS TMLR submission §4 (continual learning).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

# ── Small helpers ──────────────────────────────────────────────────────────


def update_moving_average(
    current_avg: float | torch.Tensor,
    new_value: float | torch.Tensor,
    momentum: float = 0.9,
) -> float | torch.Tensor:
    """EMA update: ``new_avg = momentum · current_avg + (1 − momentum) · new_value``.

    Kept in the public API for parity with ``SARL_CL/examples_cl/maps.py:531``.
    The current :class:`DynamicLossWeighter` does not call this; it is
    exposed so downstream scripts / notebooks can opt-in to EMA-based
    normalization instead of the running-max approach.
    """
    return momentum * current_avg + (1 - momentum) * new_value


def min_max_norm(
    mean: float | np.floating, values: list[float] | np.ndarray
) -> tuple[float | np.floating, float | np.floating]:
    """Paper's ``min_max_norm`` — divides ``mean`` by the max of ``values``.

    Port of SARL_CL/examples_cl/maps.py:510-514. Returns ``(mean / max,
    max)``. Does **not** compute a true min-max normalization despite its
    name; this is a paper-code quirk preserved verbatim.
    """
    values = np.array(values)
    max_val = np.max(values)
    return mean / max_val, max_val


def individual_losses(
    output: torch.Tensor,
    target: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    """Per-sample loss → 95th percentile summary.

    Port of SARL_CL/examples_cl/maps.py:517-528. Runs ``loss_fn`` on each
    sample independently, collects scalar values, and returns the 95th
    percentile of the per-sample distribution. Used by some exploratory
    analyses in the paper; kept for parity.

    Parameters
    ----------
    output, target : torch.Tensor
        Batched predictions and labels, same leading dim.
    loss_fn : callable
        Any ``(pred, target) -> tensor`` loss that accepts single-row
        batches (most torch functional losses do).
    """
    batch_size = output.size(0)
    losses: list[float] = []
    for i in range(batch_size):
        loss = loss_fn(output[i : i + 1], target[i : i + 1])
        losses.append(loss.item())
    return float(np.percentile(np.array(losses), 95))


# ── DynamicLossWeighter ─────────────────────────────────────────────────────


_KEYS: tuple[str, str, str] = ("task", "distillation", "feature")


class DynamicLossWeighter:
    """Per-term loss normalization via historical maximum.

    Given a dict of three losses ``{task, distillation, feature}`` at each
    training step, tracks the running max of each term's scalar value and
    divides the term by ``(running_max + epsilon)`` at weighting time.

    Usage
    -----
    >>> weighter = DynamicLossWeighter()
    >>> losses = {"task": loss_task, "distillation": loss_reg, "feature": loss_feat}
    >>> weighter.update(losses)  # detaches, updates running max
    >>> weighted = weighter.weight_losses(losses)  # returns {k: loss_k / max_k}

    The ``update_interval`` mechanism in the paper's source is effectively
    inert (the historical-max reset code is commented out in lines 579-586);
    we keep the ``update_interval`` field for API symmetry but use ONLY the
    lifetime running max for weighting — matching the actual paper behavior.

    Parameters
    ----------
    update_interval : int, default 10_000
        Retained for parity; not currently used to reset the running max.
    keys : tuple of str, optional
        Names of the three loss components. Defaults to ``("task",
        "distillation", "feature")`` matching the paper.
    """

    def __init__(
        self,
        update_interval: int = 10_000,
        keys: tuple[str, ...] = _KEYS,
    ):
        self.keys = tuple(keys)
        self.moving_avgs: dict[str, float] = {k: 1.0 for k in self.keys}
        self.historical_max: dict[str, float] = {k: float("-inf") for k in self.keys}
        # Retained for paper parity — never read by weight_losses in the
        # canonical path, but present for checkpointing and diagnostics.
        self.historical_max_prev: dict[str, float] = {k: float("-inf") for k in self.keys}
        self.steps: int = 0
        self.update_interval: int = update_interval
        self.scale_factors: dict[str, float] = {k: 1.0 for k in self.keys}

    # -- Public API -------------------------------------------------------

    def update(self, losses: dict[str, float | torch.Tensor]) -> None:
        """Detach all losses, update the most-recent value and running max."""
        self.steps += 1

        for key in self.keys:
            value = losses[key]
            value_float = float(value.item() if isinstance(value, torch.Tensor) else value)

            self.moving_avgs[key] = value_float

            # Midway through the interval, snapshot the current max into
            # ``historical_max_prev`` — paper parity (lines 571-572).
            if self.steps % self.update_interval == (self.update_interval // 2):
                self.historical_max_prev[key] = value_float

            self.historical_max[key] = max(self.historical_max[key], value_float)

    def weight_losses(
        self,
        losses: dict[str, float | torch.Tensor],
        epsilon: float = 1e-16,
    ) -> dict[str, float | torch.Tensor]:
        """Divide each loss by its historical max (+ epsilon)."""
        return {key: losses[key] / (self.historical_max[key] + epsilon) for key in self.keys}

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Diagnostic snapshot of moving averages, historical max, scales."""
        return {
            "moving_averages": self.moving_avgs.copy(),
            "historical_max": self.historical_max.copy(),
            "scale_factors": self.scale_factors.copy(),
        }
