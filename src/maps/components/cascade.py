"""Cascade accumulation (McClelland 1979, 1989).

The cascade model replaces a single-shot forward pass with a graded integration
of activations over `n_iterations` steps:

    a(t) = α · f(x) + (1 - α) · a(t - 1)

with α ∈ (0, 1]. For α = 1, the recurrence collapses to a single forward pass
(no integration). For α < 1, activations build up gradually toward the asymptote
f(x); after ~int(1/α) iterations they approach the steady state. The MAPS paper
uses α = 0.02, n_iterations = 50 — i.e. exactly one "effective" pass spread
across 50 graded steps.

References
----------
- McClelland, J. L. (1979). On the time relations of mental processes: An
  examination of systems of processes in cascade. Psychological Review, 86(4),
  287–330.
- McClelland, J. L. (1989). Parallel distributed processing: Implications for
  cognition and development. In R. G. M. Morris (Ed.), Parallel distributed
  processing: Implications for psychology and neurobiology (pp. 8–45).
- MAPS paper (Vargas et al., TMLR) §2.1 eq.6.
"""

from __future__ import annotations

import torch


def cascade_update(
    new_activation: torch.Tensor,
    prev_activation: torch.Tensor | None,
    cascade_rate: float,
) -> torch.Tensor:
    """Apply one step of the cascade recurrence.

    Parameters
    ----------
    new_activation : torch.Tensor
        The fresh activation f(x) from the current forward pass.
    prev_activation : torch.Tensor | None
        The cascade state from the previous iteration. If None, the update
        degenerates to `new_activation` (bootstrap step).
    cascade_rate : float
        Cascade rate α in (0, 1]. `cascade_rate=1` disables cascading (pure
        feed-forward); `cascade_rate=0` would freeze the state and is rejected.
        **Distinct** from the EMA wagering α (= 0.45, paper Table 11); every
        caller passes this positionally so renaming here does not break them.

    Returns
    -------
    torch.Tensor
        The updated cascade state `α·new + (1-α)·prev`.

    Raises
    ------
    ValueError
        If `cascade_rate` is outside (0, 1].
    """
    if not (0.0 < cascade_rate <= 1.0):
        raise ValueError(
            f"cascade_rate must lie in (0, 1]; got {cascade_rate}. "
            "cascade_rate=0 freezes the state (degenerate); negative/larger-than-1 "
            "values are not physically meaningful."
        )
    if prev_activation is None:
        return new_activation
    return cascade_rate * new_activation + (1.0 - cascade_rate) * prev_activation


def n_iterations_from_alpha(alpha: float) -> int:
    """Return the conventional cascade length `int(1/α)` (Blindsight/AGL convention).

    For α = 0.02 this gives 50 iterations — the paper's empirically selected
    depth. Documented here because two domains hardcode `cascade_iterations =
    int(1.0 / cascade_rate)` (Blindsight_TMLR.py:507, AGL_TMLR.py analogous).
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
    return int(1.0 / alpha)
