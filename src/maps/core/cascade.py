"""Cascade dynamics — McClelland (1979, 1989), MAPS paper §2.1 eq.6.

One-step integrator that blends new evidence with a running activation
state at rate α (typically 0.02). After roughly ``1/α`` iterations the
activation converges to the steady-state response of the network.

This is the **single canonical** implementation of eq.6 in MAPS — every
domain (Blindsight, AGL, SARL, SARL+CL) imports it. Duplicating the
formula inline anywhere in the codebase is a smell.

References
----------
McClelland, J. L. (1989). Parallel distributed processing implications
for cognition and development. *Parallel distributed processing:
Implications for psychology and neurobiology*.

Vargas et al. (2025). MAPS, TMLR submission, §2.1 eq.6.
"""

from __future__ import annotations

from torch import Tensor


def cascade_update(
    new_activation: Tensor,
    prev_activation: Tensor | None,
    cascade_rate: float,
) -> Tensor:
    r"""One iteration of the McClelland cascade — MAPS paper eq.6.

    Computes

    .. math::
       a(t) = \alpha \cdot \text{new} + (1 - \alpha) \cdot a(t - 1)

    At ``t = 0`` (bootstrap), ``prev_activation`` is ``None`` and the
    function returns ``new_activation`` unchanged.

    Parameters
    ----------
    new_activation : torch.Tensor
        Output of the network's forward pass at this iteration. In MAPS
        eq.6 this is the contribution
        :math:`\sum_j w_{ij} \cdot a_{js}(t)` from upstream layer
        ``j`` into layer ``i``.
    prev_activation : torch.Tensor or None
        Activation from the previous iteration. If ``None``, treated
        as the bootstrap ``t = 0`` and ``new_activation`` is returned
        as-is.
    cascade_rate : float
        Integration rate :math:`\alpha \in (0, 1]`. Paper §2.1 fixes
        ``α = 0.02`` (≈50 iterations to convergence). ``α = 1`` is
        allowed as the no-cascade fallback for factorial OFF settings;
        ``α = 0`` is rejected — it would freeze the activation at
        ``prev`` and admit no new information.

    Returns
    -------
    torch.Tensor
        Updated activation, same shape and dtype as ``new_activation``.

    Raises
    ------
    ValueError
        If ``cascade_rate`` is not in ``(0, 1]``.

    Notes
    -----
    **The α/EMA collision.** MAPS uses two distinct ``α`` symbols.
    This one (``cascade_rate``) is the McClelland eq.6 integration
    rate (paper §2.1, default 0.02). The other ``α`` lives in the
    wagering EMA (paper §2.2 eq.13, default 0.45). They have nothing
    to do with each other — the explicit name ``cascade_rate`` in this
    signature exists to make that distinction unmissable.

    **No-op on deterministic paths (D-sarl-cascade-noop).** When
    ``new_activation`` does not depend on stochastic dropout, iterating
    ``cascade_update`` N times produces the same tensor as one
    iteration (analytical closure :math:`a(\infty) \to \text{new}`).
    SARL Q-network Setting 2 (cascade 1st-order only, no dropout) is
    therefore mathematically equivalent to Setting 1 (no cascade). The
    paper Table 6 Setting 2 ≠ Setting 1 numbers must be RNG noise
    (N = 3 seeds, sensitive). See
    ``docs/reproduction/deviations.md`` D-sarl-cascade-noop.

    **The mechanism on stochastic paths.** When dropout is active
    *inside* the cascade (the BS/AGL ``SecondOrderNetwork`` case),
    each iteration sees a fresh mask. The running activation averages
    ~50 masks, becoming equivalent to a Monte-Carlo dropout estimate
    (Gal & Ghahramani 2016). This is what makes the cascade
    non-trivial — and why ``SecondOrderNetwork`` keeps the 50-iter
    unroll despite the apparent cost.

    References
    ----------
    .. [McClelland1989] McClelland, J. L. (1989). Parallel distributed
       processing implications for cognition and development.
    .. [Vargas2025] Vargas et al. (2025). MAPS, TMLR submission, §2.1
       eq.6.
    """
    if not 0.0 < cascade_rate <= 1.0:
        raise ValueError(f"cascade_rate must be in (0, 1], got {cascade_rate}")
    if prev_activation is None:
        return new_activation
    return cascade_rate * new_activation + (1.0 - cascade_rate) * prev_activation


def n_iterations_from_alpha(cascade_rate: float) -> int:
    r"""Convert a cascade rate to the conventional iteration count.

    Returns :math:`\lfloor 1 / \alpha \rfloor`. With the paper's
    ``α = 0.02``, returns ``50``.

    Parameters
    ----------
    cascade_rate : float
        Integration rate :math:`\alpha \in (0, 1]`.

    Returns
    -------
    int
        Number of cascade iterations to run.

    Raises
    ------
    ValueError
        If ``cascade_rate`` is not in ``(0, 1]``.

    Examples
    --------
    >>> n_iterations_from_alpha(0.02)
    50
    >>> n_iterations_from_alpha(1.0)
    1
    """
    if not 0.0 < cascade_rate <= 1.0:
        raise ValueError(f"cascade_rate must be in (0, 1], got {cascade_rate}")
    return int(1.0 / cascade_rate)
