"""SimCLR positive-pair augmentation for Blindsight stimuli — D12.4.

The SimCLR loss (:func:`maps.core.losses.simclr_loss`) needs a matched
pair ``(z_i, z_j)`` for each sample, where ``z_j`` is an augmented
view of the same sample.

D12.4 (Sprint 12 Day-1 decision) selects **bit-flip p=0.1** as the
augmentation strategy. The implementation here is the continuous analog
applicable to Blindsight stimuli (which are floats in
``[0, multiplier]``, not binary) : with probability ``p`` per element,
the value is **replaced** by fresh uniform noise in ``[0, 1]``.

Why "bit-flip" for continuous values
------------------------------------
The D12.4 baseline names it "bit-flip" because for the **AGL** domain
(Sprint 13) the stimulus is binary 8-letter × 6-bit and the natural
augmentation is literal bit flipping. Blindsight stimuli are
continuous, so the "flip" generalises to "replace with random noise"
— the spirit (uncorrelated perturbation of ``p%`` of elements) is
preserved.

If empirical comparison post-Sprint 12 (CAE vs SimCLR on Blindsight)
shows that this baseline drastically under-performs CAE, revisit
D12.4 options B (param sweep), C (combined with noise resampling), or
D (bit-dropout).
"""

from __future__ import annotations

import torch
from torch import Tensor


def bit_flip(
    x: Tensor,
    p: float = 0.1,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""SimCLR positive-pair augmentation : flip / re-sample ``p%`` of
    elements.

    For each element of ``x``, with probability ``p``, replace it with
    a fresh uniform random value in ``[0, 1]``. With probability
    ``1 - p`` the element is preserved.

    Parameters
    ----------
    x : torch.Tensor
        Input pattern of arbitrary shape. Float dtype recommended.
    p : float, optional
        Per-element probability of replacement. Default ``0.1`` (D12.4
        baseline). Must satisfy ``0 ≤ p ≤ 1``.
    generator : torch.Generator, optional
        For deterministic / reproducible test fixtures. If ``None``,
        uses the default global RNG (seeded by
        :func:`maps.utils.seeding.set_all_seeds`).

    Returns
    -------
    torch.Tensor
        Augmented copy of ``x``, same shape and dtype. Operates on a
        fresh tensor (does not modify input in-place).

    Raises
    ------
    ValueError
        If ``p`` is outside ``[0, 1]``.

    Examples
    --------
    >>> import torch
    >>> torch.manual_seed(0)
    >>> x = torch.zeros(10)
    >>> bit_flip(x, p=0.1)  # ~1 element replaced  # doctest: +SKIP
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if p == 0.0:
        return x.clone()

    mask = torch.rand(x.shape, generator=generator, device=x.device) < p
    noise = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return torch.where(mask, noise, x)
