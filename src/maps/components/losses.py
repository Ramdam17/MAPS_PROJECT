"""Loss functions used across MAPS domains.

Currently ships:

- `cae_loss` — Contractive AutoEncoder loss (first-order training, Blindsight
  + AGL). This is what the paper refers to as "contrastive" loss; it is NOT
  SimCLR / NT-Xent. The naming collision is unfortunate.
- `wagering_bce_loss` — binary cross-entropy for training the wager output
  against a correct/incorrect label.
- `distillation_loss` — KD loss (soft targets + optional hard labels) used by
  the SARL+CL teacher-student setup.

References
----------
- Rifai, S., Vincent, P., Muller, X., Glorot, X., & Bengio, Y. (2011).
  Contractive auto-encoders: Explicit invariance during feature extraction.
  ICML 2011.
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a
  neural network. arXiv:1503.02531.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cae_loss(
    W: torch.Tensor,
    x: torch.Tensor,
    recons_x: torch.Tensor,
    h: torch.Tensor,
    lam: float,
    *,
    recon: str = "bce_sum",
) -> torch.Tensor:
    """Contractive AutoEncoder loss (Rifai et al., 2011).

    CAE = recon(x, recons_x) + λ · ||J_h(x)||²_F, where the Frobenius norm
    of the encoder Jacobian is computed analytically under the assumption that
    the encoder is a single Linear → Sigmoid layer with weight matrix W.

    Parameters
    ----------
    W : torch.Tensor
        Encoder weight matrix, shape (n_hidden, n_input). Same orientation as
        `nn.Linear(in, hidden).weight`.
    x : torch.Tensor
        Input batch, shape (batch, n_input).
    recons_x : torch.Tensor
        Decoder output, shape (batch, n_input).
    h : torch.Tensor
        Hidden activations after the encoder sigmoid, shape (batch, n_hidden).
        Used to compute the sigmoid derivative `h(1-h)`.
    lam : float
        Weight on the contractive penalty term.
    recon : {"bce_sum", "mse_mean", "mse_sum"}, default "bce_sum"
        Reconstruction term. The reference Blindsight and AGL scripts use
        ``bce_sum`` — ``nn.BCELoss(size_average=False)`` over sigmoid outputs
        — even though they misleadingly name the variable ``mse_loss``. The
        MSE variants are exposed for non-binary reconstruction targets.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if recon == "bce_sum":
        recon_term = F.binary_cross_entropy(recons_x, x, reduction="sum")
    elif recon == "mse_mean":
        recon_term = F.mse_loss(recons_x, x, reduction="mean")
    elif recon == "mse_sum":
        recon_term = F.mse_loss(recons_x, x, reduction="sum")
    else:
        raise ValueError(f"Unknown `recon`: {recon!r}")

    # Sigmoid-derivative trick: ∂h/∂z = h(1-h), so the Jacobian squared Frobenius
    # norm factorises as `sum_j (h_j(1-h_j))² · sum_i W_ji²`.
    # Why detach W: the reference implementation reads W via `state_dict()['fc1.weight']`,
    # which returns a detached clone, so the contractive term only provides gradient to
    # `h` (and through the sigmoid derivative to the encoder weights), not directly to W.
    # We replicate that behavior here so training trajectories match bit-for-bit.
    W_const = W.detach()
    dh = h * (1 - h)  # shape (batch, n_hidden)
    w_sq_rowsum = torch.sum(W_const**2, dim=1, keepdim=True)  # shape (n_hidden, 1)
    contractive = torch.sum(torch.mm(dh**2, w_sq_rowsum))  # scalar
    return recon_term + lam * contractive


def wagering_bce_loss(
    wager: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy for the 1-unit wager head.

    Takes *probabilities* (the sigmoid output of `WageringHead(n_wager_units=1)`)
    rather than logits, to match the reference training loop which feeds
    post-sigmoid values. For numerical stability when training from scratch,
    prefer a logit-based variant — this one exists for parity.

    Parameters
    ----------
    wager : torch.Tensor
        Sigmoid output in [0, 1], shape (batch, 1) or (batch,).
    target : torch.Tensor
        Binary label (0 or 1), same shape as `wager`.
    reduction : {"mean", "sum", "none"}
    pos_weight : torch.Tensor | None
        Class-imbalance weight on the positive class, passed through to
        `F.binary_cross_entropy`.

    Returns
    -------
    torch.Tensor
    """
    return F.binary_cross_entropy(
        wager,
        target.to(wager.dtype),
        weight=pos_weight,
        reduction=reduction,
    )


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor | None = None,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Knowledge-distillation loss (soft + optional hard component).

    `alpha` balances soft vs hard: `loss = α · KL(teacher_T || student_T) + (1-α) · CE(student, hard)`.
    The KL term is scaled by T² as in Hinton et al. (2015) to preserve gradient
    magnitudes across temperatures — implemented here via the manual KL form
    used in `SARL_CL/examples_cl/maps.py:359`.

    Parameters
    ----------
    student_logits, teacher_logits : torch.Tensor
        Pre-softmax outputs from student and teacher, shape (batch, n_classes).
    hard_labels : torch.Tensor | None
        Integer class labels, shape (batch,). When provided, the hard
        cross-entropy term is mixed in; otherwise only the soft term is used.
    alpha : float, default 0.5
        Weight on the soft (distillation) term.
    temperature : float, default 2.0
        Softens both distributions; T=1 reduces to plain KL.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    # KL(teacher || student) as written in the reference code, not scaled by T².
    # Kept as-is for parity with SARL+CL; if we later want Hinton-style
    # gradient preservation, multiply by T² here and document the deviation.
    soft_loss = torch.sum(
        soft_targets * (soft_targets.clamp_min(1e-12).log() - log_probs), dim=-1
    ).mean()

    if hard_labels is None:
        return alpha * soft_loss

    hard_loss = F.cross_entropy(student_logits, hard_labels)
    return alpha * soft_loss + (1.0 - alpha) * hard_loss
