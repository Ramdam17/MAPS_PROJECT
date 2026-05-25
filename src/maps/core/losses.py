"""MAPS first-order, wagering, regularization, and contrastive losses.

Four loss families that span the MAPS pipeline:

- :func:`cae_loss` — Contractive AutoEncoder (Rifai et al. 2011), the
  first-order loss in the reference student code. The paper §2.1 eq.4
  prose describes a SimCLR-style contrastive loss instead; we keep
  CAE here because the paper Tables 5/6/7 numbers were produced by
  the student code. See D-002.
- :func:`wagering_bce_loss` — BCE on the second-order wager output
  (paper §2.1 eq.5).
- :func:`weight_regularization` — L2 parameter-drift anchor between
  student and (frozen) teacher; used by SARL+CL as the practical
  "distillation" signal (EWC-style without Fisher weighting,
  Kirkpatrick et al. 2017).
- :func:`simclr_loss` — paper-prose-faithful SimCLR / NT-Xent loss
  (Chen et al. 2020). New in Sprint 11.D (D11.7): implemented as a
  real loss, NOT a stub. Domain augmentation strategies arrive in
  Sprint 12+; empirical CAE-vs-SimCLR comparison on Blindsight is
  reserved for post-Sprint 12.

Notes
-----
``distillation_loss`` (KL Hinton 2015) existed in the student code as
``DistillationLoss`` but was never invoked. **Deleted in this rewrite
(D11.5).** Recover from git history if needed:

.. code-block:: bash

   git show main:src/maps/components/losses.py

References
----------
- Rifai, S., et al. (2011). Contractive auto-encoders.
- Chen, T., et al. (2020). A simple framework for contrastive
  learning of visual representations (SimCLR / NT-Xent).
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting
  in neural networks (EWC — inspires ``weight_regularization``).
- Vargas et al. (2025). MAPS, TMLR submission, §2.1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_RECON_VARIANTS = ("bce_sum", "mse_mean", "mse_sum")


def cae_loss(
    weight: Tensor,
    x: Tensor,
    recons_x: Tensor,
    hidden: Tensor,
    lam: float,
    *,
    recon: str = "bce_sum",
) -> Tensor:
    r"""Contractive AutoEncoder loss — Rifai et al. (2011).

    Computes

    .. math::
       L_{CAE} = \text{recon}(x, \hat{x})
                 + \lambda \cdot \|J_h(x)\|^2_F

    using the sigmoid-derivative closed form for the Frobenius norm of
    the Jacobian:

    .. math::
       \|J_h\|^2_F = \sum_j (h_j(1-h_j))^2 \sum_i W_{ji}^2.

    Parameters
    ----------
    weight : torch.Tensor
        Encoder weight matrix of shape ``(N_hidden, N_input)``.
        Detached before the Jacobian computation (matches the
        student's accidental ``state_dict()`` keep_vars=False default
        — gradient flows through ``hidden`` only, not directly
        through ``weight``).
    x : torch.Tensor
        Input batch of shape ``(N_batch, N_input)``.
    recons_x : torch.Tensor
        Decoder output of the same shape as ``x``. For
        ``recon="bce_sum"`` (default), values must be in ``[0, 1]``
        (post-sigmoid).
    hidden : torch.Tensor
        Encoder activations of shape ``(N_batch, N_hidden)``.
    lam : float
        Weight of the Jacobian regulariser. Paper §2.1 uses
        ``λ = 1e-4`` (also in ``config/maps.yaml``).
    recon : str, optional
        Reconstruction term. One of ``"bce_sum"`` (default, matches
        student), ``"mse_mean"``, ``"mse_sum"``.

    Returns
    -------
    torch.Tensor
        Scalar loss.

    Raises
    ------
    ValueError
        If ``recon`` is not a supported variant.

    Notes
    -----
    **The ``h(1-h)`` quirk on ReLU encoders.** The closed-form
    Jacobian :math:`h(1-h)` is the sigmoid derivative — exact only
    when the encoder activation is sigmoid. All three MAPS domains
    (Blindsight, AGL, SARL) use ReLU encoders, for which this formula
    is **mathematically incorrect** (the ReLU derivative is
    :math:`\mathbb{1}[z > 0]`). The student code does this universally
    and the paper Tables 5/6/7 were produced by it; we preserve the
    quirk byte-for-byte for parity. See D-001 in
    ``docs/reproduction/deviations.md`` and the deeper discussion in
    ``docs/learning/reverse-prompts/core/losses.md``.

    References
    ----------
    .. [Rifai2011] Rifai et al. (2011). Contractive auto-encoders.
    """
    if recon == "bce_sum":
        reconstruction = F.binary_cross_entropy(recons_x, x, reduction="sum")
    elif recon == "mse_mean":
        reconstruction = F.mse_loss(recons_x, x, reduction="mean")
    elif recon == "mse_sum":
        reconstruction = F.mse_loss(recons_x, x, reduction="sum")
    else:
        raise ValueError(f"recon must be one of {_RECON_VARIANTS}; got {recon!r}")

    # h(1-h) — sigmoid-derivative closed-form, preserved byte-for-byte
    # on ReLU activations for parity (see Notes).
    dh = hidden * (1.0 - hidden)
    # Detach W: gradient flows through `hidden` only.
    w_detached = weight.detach()
    # Row-wise sum of squared weights, shape (N_hidden, 1) for matmul.
    w_sum = torch.sum(w_detached**2, dim=1).unsqueeze(1)
    # (N_batch, N_hidden) @ (N_hidden, 1) → (N_batch, 1), then sum.
    contractive = torch.sum(torch.mm(dh**2, w_sum), dim=0).squeeze()

    return reconstruction + lam * contractive


def wagering_bce_loss(
    wager: Tensor,
    target: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    r"""Wagering binary cross-entropy — MAPS paper §2.1 eq.5.

    Computes

    .. math::
       L_{BCE} = -[y \log(p) + (1-y) \log(1-p)]

    on **probabilities** (post-sigmoid wager outputs), not raw logits.
    For the ``WageringHead(n_wager_units=2)`` raw-logits variant,
    callers should use
    :func:`torch.nn.functional.binary_cross_entropy_with_logits`
    directly (more numerically stable).

    Parameters
    ----------
    wager : torch.Tensor
        Wager probability in ``[0, 1]`` (post-sigmoid).
    target : torch.Tensor
        Binary target in ``{0, 1}`` (or soft target in ``[0, 1]``),
        same shape as ``wager``.
    reduction : str, optional
        ``"mean"`` (default — modern PyTorch convention), ``"sum"``,
        or ``"none"``. The reference student trainer uses ``"sum"``
        (``nn.BCELoss(size_average=False)``) — pass ``reduction="sum"``
        explicitly when bit-exact parity is needed.

    Returns
    -------
    torch.Tensor
        Scalar loss for ``"mean"``/``"sum"``, per-element tensor for
        ``"none"``.

    Notes
    -----
    Sprint 11 removed the legacy ``pos_weight`` parameter (D11.6 —
    YAGNI, mis-named, zero callers). Re-add via
    ``F.binary_cross_entropy(..., weight=...)`` in one line if class
    imbalance handling becomes a real need.
    """
    return F.binary_cross_entropy(wager, target, reduction=reduction)


def weight_regularization(
    student_model: nn.Module,
    teacher_model: nn.Module,
) -> Tensor:
    r"""L2 parameter-drift anchor between student and frozen teacher.

    Computes

    .. math::
       L_{reg} = \sum_i (\theta_i^{student} - \theta_i^{teacher})^2.

    This is the practical "distillation" loss used by SARL+CL — an
    EWC-style anchor (Kirkpatrick et al. 2017) **without Fisher
    weighting**.

    Parameters
    ----------
    student_model : torch.nn.Module
        The trainable model.
    teacher_model : torch.nn.Module
        A frozen reference model. **The caller MUST freeze it**
        (``teacher_model.requires_grad_(False)``) — otherwise
        gradients flow into the teacher and silently corrupt the
        regularization signal.

    Returns
    -------
    torch.Tensor
        Scalar L2 drift loss. Will be ``0`` for identical models.

    Raises
    ------
    ValueError
        If ``student_model`` and ``teacher_model`` have different
        parameter topology (zip ``strict=True``).

    Notes
    -----
    Paper §2.2 calls this "distillation" in the
    ``DynamicLossWeighter`` dict keys, which is misleading — it is an
    L2 parameter-space anchor, not a KL-divergence distillation
    (cf. Hinton et al. 2015). The genuine KL distillation loss
    (``distillation_loss``) was deleted in Sprint 11 (D11.5) since
    no caller ever used it.
    """
    return sum(
        torch.sum((p_s - p_t) ** 2)
        for p_s, p_t in zip(
            student_model.parameters(),
            teacher_model.parameters(),
            strict=True,
        )
    )


def simclr_loss(
    z_i: Tensor,
    z_j: Tensor,
    *,
    temperature: float = 0.5,
    reduction: str = "mean",
) -> Tensor:
    r"""SimCLR / NT-Xent contrastive loss — Chen et al. (2020).

    Computes

    .. math::
       L_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}
                            {\sum_{k \neq i}^{2N} \exp(\text{sim}(z_i, z_k)/\tau)}

    where :math:`\text{sim}` is cosine similarity. The positive pair
    for sample ``i`` is the matched row from ``z_j``; the negatives
    are all other rows from both ``z_i`` and ``z_j``.

    This is the **paper-prose-faithful** variant of MAPS eq.4 (see
    D-002). **It is NOT used by the reference student runs** — those
    use :func:`cae_loss` instead. SimCLR is reserved for empirical
    comparison on Blindsight Sprint 12+ once domain-specific
    augmentation strategies exist.

    Parameters
    ----------
    z_i : torch.Tensor
        First view of the batch, shape ``(N, D)``.
    z_j : torch.Tensor
        Second (augmented) view, same shape as ``z_i``. The
        augmentation strategy (how ``z_j`` is derived from the same
        sample as ``z_i``) is the caller's responsibility — typically
        a stochastic transform is applied before encoding.
    temperature : float, optional
        Temperature ``τ > 0``. Lower values sharpen the softmax
        (typical range ``[0.05, 0.5]``, default ``0.5``). Paper §2.1
        eq.4 leaves ``τ`` unspecified.
    reduction : str, optional
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns
    -------
    torch.Tensor
        Scalar loss for ``"mean"``/``"sum"``, per-row tensor of shape
        ``(2N,)`` for ``"none"``.

    Raises
    ------
    ValueError
        If ``z_i`` and ``z_j`` shapes mismatch, are not 2D, or
        ``temperature <= 0``.

    Notes
    -----
    Implementation stacks ``[z_i; z_j]`` into a ``(2N, D)`` tensor,
    L2-normalises rows for cosine similarity, computes the full
    ``(2N, 2N)`` similarity matrix divided by ``τ``, masks the
    diagonal (self-similarity) with ``-inf``, and treats the index
    of the paired view as the target for cross-entropy. This is the
    canonical NT-Xent formulation.

    **No parity reference.** The paper Tables 5/6/7 were produced by
    ``cae_loss``, not by SimCLR. Tests for this function are sanity
    only (monotonicity in similarity, temperature scaling, gradient
    flow). For an extra safety net, cross-check against external
    references (e.g. ``lightly.loss.NTXentLoss``) at fixed seed.

    References
    ----------
    .. [Chen2020] Chen et al. (2020). A simple framework for
       contrastive learning of visual representations. ICML.
    """
    if z_i.shape != z_j.shape:
        raise ValueError(
            f"z_i and z_j must have the same shape; got {tuple(z_i.shape)} vs {tuple(z_j.shape)}"
        )
    if z_i.dim() != 2:
        raise ValueError(f"z_i must be 2D (batch, features); got shape {tuple(z_i.shape)}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0; got {temperature}")

    batch_size = z_i.size(0)
    device = z_i.device

    # Stack [z_i; z_j] → (2N, D), L2-normalize for cosine similarity.
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)

    # (2N, 2N) cosine similarity matrix, scaled by 1/τ.
    sim = (z @ z.T) / temperature

    # Mask self-similarity (diagonal) with -inf so it contributes 0
    # to the softmax denominator.
    diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    sim = sim.masked_fill(diag_mask, float("-inf"))

    # Positive target index per row:
    #   row i ∈ [0, N)  → positive is i + N (the j-th view of sample i)
    #   row i ∈ [N, 2N) → positive is i - N (the i-th view of sample i-N)
    positives = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device),
        ]
    )

    return F.cross_entropy(sim, positives, reduction=reduction)
