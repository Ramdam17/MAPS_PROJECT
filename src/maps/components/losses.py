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
    of the encoder Jacobian uses the analytical sigmoid-derivative form
    `h(1-h)`. **Important: this form is mathematically valid only for sigmoid
    encoders** — a quirk of the original CAE paper (Rifai 2011). All three
    reference domains (Blindsight, AGL, SARL) use **ReLU** encoders and apply
    `h(1-h)` regardless. We preserve that quirk byte-for-byte for parity with
    the student code; do NOT "fix" to `(h > 0).float()` without discussion.
    See `docs/reviews/first_order_mlp.md §C.11 (b)` and `docs/reviews/losses.md
    §C.7` for the full audit.

    D-002 note
    ----------
    The paper (eq.4) describes a **SimCLR / NT-Xent contrastive loss** (Chen
    et al. 2020), not a CAE. The reference student code implements CAE
    (Rifai 2011) despite using "contrastive" phraseology in comments.
    We faithfully port the student's CAE. See `docs/reproduction/deviations.md`
    D-002 and `docs/reviews/losses.md §C.7` for the full analysis; the decision
    of whether to add a paper-faithful SimCLR variant is tracked as sub-phase
    D.22b in the Sprint-08 plan.

    Assumptions on inputs
    ---------------------
    * ``recons_x`` **must lie in [0, 1]** (typically post-sigmoid decoder) when
      ``recon="bce_sum"`` because `F.binary_cross_entropy` rejects values
      outside that range. Blindsight uses a global sigmoid decoder; AGL uses
      `make_chunked_sigmoid(6)` per-letter.
    * ``h`` is passed as the **post-encoder** activations — in this codebase
      that means **post-ReLU** (Blindsight/AGL via `FirstOrderMLP`, SARL via
      `SarlQNetwork.fc_hidden`). The `h*(1-h)` formula is therefore applied
      on ReLU output rather than a sigmoid — paper-faithful quirk, not a
      correct Jacobian. This has been the behaviour across all three domains
      since the original student code.

    Parameters
    ----------
    W : torch.Tensor
        Encoder weight matrix, shape (n_hidden, n_input). Same orientation as
        `nn.Linear(in, hidden).weight`. We detach W inside to match the
        student's `state_dict()['fc1.weight']` access pattern (PyTorch
        `state_dict(keep_vars=False)` returns detached tensors by default —
        gradient does NOT flow directly to W through the contractive term,
        only via `h`'s backward path through the encoder).
    x : torch.Tensor
        Input batch, shape (batch, n_input).
    recons_x : torch.Tensor
        Decoder output, shape (batch, n_input). See "Assumptions on inputs".
    h : torch.Tensor
        Post-encoder activations (post-ReLU in current callers), shape
        (batch, n_hidden). Used to compute the sigmoid-derivative-shaped
        term `h(1-h)` — see "Assumptions on inputs" for the ReLU quirk.
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
) -> torch.Tensor:
    """Binary cross-entropy for the 1-unit wager head.

    Takes *probabilities* (the sigmoid output of `WageringHead(n_wager_units=1)`)
    rather than logits, to match the reference training loop which feeds
    post-sigmoid values. For numerical stability when training from scratch,
    prefer a logit-based variant — this one exists for parity.

    Incompatible with ``WageringHead(n_wager_units=2)`` post-C.6
    --------------------------------------------------------
    After Phase C.6, `n_wager_units=2` returns **raw logits** (paper eq.3
    faithful). Passing those logits here will make `F.binary_cross_entropy`
    raise because it requires inputs in [0, 1]. For the 2-unit path, use
    `F.binary_cross_entropy_with_logits` directly.

    Parameters
    ----------
    wager : torch.Tensor
        Sigmoid output in [0, 1], shape (batch, 1) or (batch,).
    target : torch.Tensor
        Binary label (0 or 1), same shape as `wager`.
    reduction : {"mean", "sum", "none"}
        Reduction applied to the per-element BCE. Reference callers pass
        ``"sum"`` explicitly (matching student `nn.BCELoss(size_average=False)`);
        default is the PyTorch-standard ``"mean"``.

    Returns
    -------
    torch.Tensor
    """
    return F.binary_cross_entropy(
        wager,
        target.to(wager.dtype),
        reduction=reduction,
    )


# ⚠️ `distillation_loss` is NOT called by any production training code (grep
# confirmed: 0 callers in src/ other than re-exports). The paper's SARL+CL
# protocol uses `weight_regularization` (L2 param anchor) as the "distillation"
# signal, not this KL soft-target loss. We port it faithfully because the
# student code does the same (`DistillationLoss` class defined but never
# invoked). Kept exposed via __init__ in case future work wants Hinton-style
# distillation. See `docs/reports/sprint-04b-report.md:37` and
# `docs/reviews/losses.md` §C.9 (e) / DETTE-3.
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


def weight_regularization(model: torch.nn.Module, teacher_model: torch.nn.Module) -> torch.Tensor:
    """Elastic-Weight-Consolidation-inspired L2 penalty on parameter drift.

    Computes ``Σ_i (θ_i − θ_i^teacher)²`` where parameters are paired by
    position in ``parameters()`` — i.e. the two modules must have identical
    structure and parameter order. This mirrors the paper's
    ``compute_weight_regularization`` (SARL_CL/examples_cl/maps.py:410-416)
    which is used as the "distillation" loss component in continual learning.

    Unlike true EWC (Kirkpatrick et al., 2017), this does NOT weight per-
    parameter drift by Fisher-information; all weights contribute equally.
    This is a deliberate simplification in the paper's code and carries to
    this port.

    Parameters
    ----------
    model : nn.Module
        Student network currently being trained.
    teacher_model : nn.Module
        Frozen teacher checkpoint. Must have the same parameter **count** as
        ``model``; we enforce this via ``zip(..., strict=True)`` so length
        mismatches raise ``ValueError`` immediately instead of producing a
        silently-truncated loss. Per-parameter shape mismatches still surface
        as ``RuntimeError`` from ``torch.sum((p - p_teacher)**2)``.

        **Caller contract:** ``teacher_model`` MUST be frozen
        (``teacher_model.requires_grad_(False)``) before calling. Otherwise
        the L2 term has non-zero gradient w.r.t. teacher parameters and the
        anchor drifts — a silent correctness bug. We do not assert this at
        runtime because iterating all params every call is wasteful.

    Returns
    -------
    torch.Tensor
        Scalar L2 drift. Requires grad through ``model`` only (provided the
        teacher has been correctly frozen by the caller).

    Notes
    -----
    The paper calls this "distillation" but strictly speaking it is an L2
    regularization anchor, not output-distillation. We keep the ``distillation``
    key in ``DynamicLossWeighter`` for faithful parity with the paper's
    dictionary keys, but code-level we use ``weight_regularization`` to avoid
    confusion with :func:`distillation_loss` above.
    """
    reg_loss = model.parameters().__next__().new_zeros(())  # scalar on the right device
    for param, param_teacher in zip(model.parameters(), teacher_model.parameters(), strict=True):
        reg_loss = reg_loss + torch.sum((param - param_teacher) ** 2)
    return reg_loss
