"""SARL loss functions — contractive autoencoder loss with Huber reconstruction.

Ports ``CAE_loss`` from ``external/MinAtar/examples/maps.py:330-379``
(commit ``ec5bcb7``).

The function is used by the paper with a *misleading* call signature:

.. code-block:: python

    loss = CAE_loss(W, target, Q_s_a, h1, lam)
    #              ↑  ↑       ↑      ↑   ↑
    #              W  x       recons h   lam

i.e. ``x = target`` (TD target) and ``recons_x = Q_s_a`` (predicted Q-value).
The Huber term therefore measures ``huber(Q_s_a, target)`` — the usual DQN
TD error. The "contractive autoencoder" naming is historical; the loss here
is really `Huber(Q, target) + λ · Jacobian regularization on fc_hidden`.

We keep the parameter order identical so future readers comparing to the
paper code don't get confused by a reshuffle.

References
----------
- Rifai et al. (2011). Contractive Auto-Encoders: Explicit Invariance During
  Feature Extraction. ICML. — source of the Jacobian regularizer.
- Huber (1964). Robust Estimation of a Location Parameter. Ann. Math. Statist.
  — the reconstruction term is smooth-L1 (Huber), not MSE.
- Vargas et al. (2025), MAPS TMLR submission §3.

Notes on parity with the paper
------------------------------
- ``W`` is passed in by the caller — typically as ``policy_net.fc_hidden.weight``
  (a live `nn.Parameter`) or equivalently via
  ``policy_net.state_dict()['fc_hidden.weight']``. **Important:** PyTorch's
  ``state_dict(keep_vars=False)`` (the default) returns **detached** tensors,
  so accessing W through state_dict already strips gradient tracking. The
  student and this port are therefore equivalent on this point: the Jacobian
  term does **not** propagate gradient directly back to ``fc_hidden.weight``;
  only ``h`` carries the contractive backward signal (through its own
  autograd graph back to the encoder weights). See
  `docs/reviews/losses.md` §C.7 (d) for the full audit.
- ``dh = h * (1 - h)`` uses Hadamard product and assumes ``h`` is post-ReLU
  (the paper applies this after ``f.relu(self.fc_hidden(...))``). The
  derivative formula is mathematically valid for **sigmoid** activations
  (Rifai 2011), not ReLU — so here it is a paper-faithful **quirk**, not a
  correct Jacobian. We preserve it byte-for-byte for parity; do NOT "fix" it
  to ``(h > 0).float()`` without discussion.
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
) -> torch.Tensor:
    """Huber reconstruction + contractive Jacobian regularizer.

    Parameters
    ----------
    W : Tensor of shape (N_hidden, N_input)
        Weight matrix of the hidden layer. Typically passed via
        ``policy_net.state_dict()['fc_hidden.weight']``, which PyTorch
        **detaches by default** (``keep_vars=False``). The Jacobian term
        therefore does NOT propagate gradient directly back to W — only
        via ``h``'s own autograd graph through the encoder. See the module
        docstring (L32-41) and `docs/reviews/losses.md §C.7 (d)` for the
        full audit. (Corrected 2026-04-20, D.11-fix-1 — the earlier
        wording "so gradients flow back through it" was a residual from
        before the C.10 module-docstring correction.)
    x : Tensor of shape (N_batch, ...)
        "Target" signal. In the DQN context this is the bootstrapped TD
        target ``r + γ · max_a Q_target(s', a)``.
    recons_x : Tensor of shape (N_batch, ...)
        "Reconstruction" signal. In the DQN context this is the predicted
        Q-value ``Q_policy(s, a)`` for the action taken.
    h : Tensor of shape (N_batch, N_hidden)
        Hidden activations (post-ReLU).
    lam : float
        Weight on the contractive (Jacobian) regularization term. Paper
        uses ``lam = 1e-4``.

    Returns
    -------
    loss : Tensor (scalar-like, shape (1,))
        ``huber(recons_x, x) + λ · Σ (h·(1-h))² · W²``.
    """
    # Huber (smooth L1) reconstruction term.
    mse = F.huber_loss(recons_x, x)

    # Jacobian regularizer on ReLU hidden layer.
    # Sum along input dim of W² gives per-hidden-unit scale.
    dh = h * (1 - h)  # (N_batch, N_hidden)
    w_sum = torch.sum(W**2, dim=1)  # (N_hidden,)
    w_sum = w_sum.unsqueeze(1)  # (N_hidden, 1)
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)  # (1,)

    return mse + lam * contractive_loss
