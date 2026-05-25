"""MAPS second-order network — Pasquali & Cleeremans (2010) comparator + wager.

Three building blocks of the metacognitive layer (paper §2.1, §2.2):

- :class:`ComparatorMatrix` — stateless eq.1
  :math:`C = X - \\hat{Y}`.
- :class:`WageringHead` — linear (optionally Linear → ReLU → Linear)
  readout producing the confidence wager (eq.3); the hidden-layer
  path restores the Pasquali architecture the student code silently
  dropped (D.25 / D.28).
- :class:`SecondOrderNetwork` — composition
  Comparator → Dropout → cascade → Wager (eq.1 → eq.2 → eq.6 → eq.3).

The triplet is shared by Blindsight and AGL. SARL/SARL+CL use a
separate ``SarlSecondOrderNetwork`` (tied-weight decoder, smaller
dropout, raw 2-unit logits) — DETTE-1, unification reserved for
post-Phase F.

References
----------
- Pasquali, A., & Cleeremans, A. (2010). Know thyself: Metacognitive
  networks and measures of consciousness. *Cognition*, 117(2),
  182–190.
- Koch, C., & Preuschoff, K. (2007). Betting the house on
  consciousness. *Nature Neuroscience*, 10(2), 140–141.
- Vargas et al. (2025). MAPS, TMLR submission, §2.1, §2.2.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from maps.core.cascade import cascade_update


class ComparatorMatrix(nn.Module):
    r"""Element-wise comparator — MAPS paper §2.1 eq.1.

    .. math::
       \mathbf{C}_t = \mathbf{X}_t - \hat{\mathbf{Y}}^{(1)}_t

    Stateless (no parameters), but kept as :class:`nn.Module` for
    symmetry with :class:`WageringHead` and :class:`SecondOrderNetwork`,
    and to leave the door open for a future learned comparator without
    breaking the composition API. See D11.4 in
    ``docs/sprints/sprint-11-core-rewrite.md`` for the design
    discussion.
    """

    def forward(
        self,
        first_order_input: Tensor,
        first_order_output: Tensor,
    ) -> Tensor:
        r"""Compute :math:`C = X - \hat{Y}`.

        Parameters
        ----------
        first_order_input : torch.Tensor
            Original first-order input :math:`X_t`.
        first_order_output : torch.Tensor
            First-order reconstruction :math:`\hat{Y}^{(1)}_t`. Must
            share shape with ``first_order_input``.

        Returns
        -------
        torch.Tensor
            Element-wise difference, same shape as the inputs.

        Raises
        ------
        ValueError
            If shapes don't match. PyTorch would broadcast silently
            here — refusing makes mistakes visible.
        """
        if first_order_input.shape != first_order_output.shape:
            raise ValueError(
                f"first_order_input and first_order_output must have the same "
                f"shape; got {tuple(first_order_input.shape)} vs "
                f"{tuple(first_order_output.shape)}"
            )
        return first_order_input - first_order_output


class WageringHead(nn.Module):
    r"""Wagering output head — MAPS paper §2.1 eq.3.

    With ``hidden_dim=None`` (default), reduces to a single linear
    readout:

    .. math::
       \mathbf{W}_t = \mathbf{W} \cdot \mathbf{C}'_t + \mathbf{b}.

    With ``hidden_dim > 0``, inserts a ReLU bottleneck per
    Pasquali & Cleeremans (2010):

    .. math::
       h_t &= \mathrm{ReLU}(\mathbf{W}_1 \mathbf{C}'_t + \mathbf{b}_1) \\
       \mathbf{W}_t &= \mathbf{W}_2 h_t + \mathbf{b}_2.

    The student code had a ``hidden_2nd`` parameter but never wired
    it in (silent bug). Restored here as a paper-faithful option —
    see ``docs/reproduction/deviations.md`` D-blindsight-wager-hidden
    (D.25 / D.28).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the (post-cascade, post-dropout) comparator
        output.
    n_wager_units : int, optional
        Wager output dimensionality. ``1`` (default) → sigmoid-activated
        scalar (student parity). ``2`` → raw logits — paper-faithful
        per Koch & Preuschoff (2007); consume with
        :func:`torch.nn.functional.binary_cross_entropy_with_logits`,
        NOT :func:`maps.core.losses.wagering_bce_loss` (which expects
        probs).
    hidden_dim : int or None, optional
        ``None`` (default — D11.2): direct linear readout, student
        behaviour. Positive int: Pasquali hidden layer of that width.

    Attributes
    ----------
    readout : torch.nn.Linear
        Final linear layer. ``readout.weight`` initialized
        uniform :math:`[0, 0.1]` to match the student code.
    hidden : torch.nn.Linear or None
        Optional pre-readout linear; ``None`` when ``hidden_dim`` is
        ``None``.
    n_wager_units : int
        Stored for use in :meth:`forward` (controls sigmoid vs logits).
    """

    def __init__(
        self,
        input_dim: int,
        *,
        n_wager_units: int = 1,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if n_wager_units < 1:
            raise ValueError(f"n_wager_units must be ≥ 1; got {n_wager_units}")
        if hidden_dim is not None and hidden_dim < 1:
            raise ValueError(f"hidden_dim must be ≥ 1 or None; got {hidden_dim}")

        self.n_wager_units = n_wager_units

        self.hidden: nn.Linear | None
        if hidden_dim is None:
            self.hidden = None
            self.readout = nn.Linear(input_dim, n_wager_units)
        else:
            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.readout = nn.Linear(hidden_dim, n_wager_units)

        # Init readout weight uniformly (matches student); leave bias and
        # (if present) hidden layer at PyTorch defaults.
        nn.init.uniform_(self.readout.weight, a=0.0, b=0.1)

    def forward(self, comparison: Tensor) -> Tensor:
        """Compute the wager from a comparator-derived feature vector.

        Parameters
        ----------
        comparison : torch.Tensor
            Post-cascade comparator output of shape
            ``(N_batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Wager of shape ``(N_batch, n_wager_units)``. Sigmoid-activated
            if ``n_wager_units == 1``; raw logits otherwise.
        """
        if self.hidden is not None:
            comparison = F.relu(self.hidden(comparison))
        logits = self.readout(comparison)
        if self.n_wager_units == 1:
            return torch.sigmoid(logits)
        return logits


class SecondOrderNetwork(nn.Module):
    r"""Second-order metacognitive network — paper §2.1, §2.2.

    Composes, **per cascade step**:

    1. :class:`ComparatorMatrix` — :math:`C_t = X_t - \hat{Y}_t` (eq.1).
    2. :class:`torch.nn.Dropout` — :math:`C'_t = \text{Drop}(C_t)` (eq.2).
    3. :func:`maps.core.cascade.cascade_update` — accumulate with
       previous step (eq.6).
    4. :class:`WageringHead` — :math:`W_t` (eq.3).

    **The cascade loop is the caller's responsibility**: this module
    does one cascade step per forward call. The caller threads the
    returned ``comparison_out`` back as ``prev_comparison`` at the
    next iteration. The trainer runs the loop ~50 times at
    ``cascade_rate=0.02``.

    Cascade + dropout = MC-dropout averaging. When the cascade runs
    for many iterations **with dropout active**, each iteration sees
    a fresh dropout mask and the running ``comparison_out`` becomes
    equivalent to a Monte-Carlo dropout estimate (Gal & Ghahramani
    2016). This is what makes the 50-iter unroll non-trivial — see
    :func:`maps.core.cascade.cascade_update` Notes (and
    D-sarl-cascade-noop) for the deterministic-path no-op.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the first-order I/O space (and of the
        comparator). Blindsight: 100. AGL: 48.
    n_wager_units : int, optional
        Wager output dimensionality (see :class:`WageringHead`).
    hidden_dim : int or None, optional
        Pasquali hidden width (see :class:`WageringHead`).
    dropout : float, optional
        Dropout probability for the comparator (eq.2). Paper §2.1
        uses ``0.5`` (Blindsight/AGL). The SARL variant uses ``0.1``
        but lives in its own class.

    Attributes
    ----------
    comparator : ComparatorMatrix
    dropout : torch.nn.Dropout
    wager : WageringHead
        Attribute named ``wager`` (Koch & Preuschoff vocabulary) per
        D11.3.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        n_wager_units: int = 1,
        hidden_dim: int | None = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.comparator = ComparatorMatrix()
        self.dropout = nn.Dropout(p=dropout)
        self.wager = WageringHead(
            input_dim,
            n_wager_units=n_wager_units,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        first_order_input: Tensor,
        first_order_output: Tensor,
        prev_comparison: Tensor | None,
        cascade_rate: float,
    ) -> tuple[Tensor, Tensor]:
        """One cascade step of the second-order network.

        Parameters
        ----------
        first_order_input : torch.Tensor
            Original first-order input :math:`X_t`.
        first_order_output : torch.Tensor
            First-order reconstruction :math:`\\hat{Y}_t`.
        prev_comparison : torch.Tensor or None
            ``comparison_out`` from the previous cascade step, or
            ``None`` at bootstrap (t=0).
        cascade_rate : float
            Cascade integration rate (see
            :func:`maps.core.cascade.cascade_update`).

        Returns
        -------
        wager : torch.Tensor
            Wager of shape ``(N_batch, n_wager_units)``.
        comparison_out : torch.Tensor
            Post-cascade comparator output, shape
            ``(N_batch, input_dim)``. **Thread this back as**
            ``prev_comparison`` **at the next step** — otherwise the
            cascade never converges.
        """
        comparison_matrix = self.comparator(first_order_input, first_order_output)
        comparison_dropped = self.dropout(comparison_matrix)
        comparison_out = cascade_update(comparison_dropped, prev_comparison, cascade_rate)
        wager = self.wager(comparison_out)
        return wager, comparison_out
