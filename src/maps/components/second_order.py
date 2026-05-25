"""Second-order (metacognitive) network: comparator + wagering head.

The second-order network watches the first-order network and produces a
confidence ("wager") about the first-order decision, modelling metacognitive
self-evaluation. It has two components:

1. **Comparator** — computes the element-wise mismatch `C = x - ŷ`
   between the first-order input and reconstruction (eq.1).
2. **Wagering head** — a small linear readout over the (dropout-regularised,
   cascade-integrated) comparator that outputs a confidence signal.

References
----------
- Pasquali, A., Timmermans, B., & Cleeremans, A. (2010). Know thyself:
  Metacognitive networks and measures of consciousness. Cognition, 117(2),
  182–190.
- Koch, C., & Preuschoff, K. (2007). Betting the house on consciousness. Nature
  Neuroscience, 10(2), 140–141.
- MAPS paper (Vargas et al., TMLR) §2.1 eq.1-3.

Deviation from the paper
------------------------
The paper describes a 2-unit wagering head (high-bet / low-bet; Koch &
Preuschoff 2007 style) producing **raw logits** — eq.3 is `W = W·C' + b`
with no activation, and eq.5 applies a per-unit `binary_cross_entropy_with_logits`
(i.e. an internal sigmoid per unit, **not** a softmax over units). The reference
implementations in `Blindsight_TMLR.py` and `AGL_TMLR.py` use a single sigmoid
unit (scalar confidence in [0, 1]) instead. We default to `n_wager_units=1` for
parity with that reference code; set `n_wager_units=2` to get the paper-faithful
raw-logit variant (downstream loss applies its own sigmoid). See
`docs/reproduction/deviations.md` (D-001) for rationale and
`src/maps/experiments/sarl/model.SarlSecondOrderNetwork` for the raw-logit SARL
variant that DOES match the paper.

Note on SARL duplication
------------------------
SARL uses a separate `SarlSecondOrderNetwork` (tied-weight decoder, raw logits,
different dims, dropout p=0.1). This duplication is tracked as DETTE-1 in
`docs/reproduction/deviations.md` — not unified to preserve paper-faithful
reproduction until validated.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

from maps.components.cascade import cascade_update

#: Uniform init range for wager-head weights. Matches student
#: `Blindsight_TMLR.py:237`, `AGL_TMLR.py:239`, `sarl_maps.py:264`.
DEFAULT_WAGER_INIT_RANGE: tuple[float, float] = (0.0, 0.1)


class ComparatorMatrix(nn.Module):
    """Element-wise difference `C = first_order_input - first_order_output` (paper eq.1).

    Stateless by design — kept as an `nn.Module` only for symmetry with the
    rest of the architecture (you can swap it for a learned comparator later
    without rewiring the second-order network).

    Inputs **must** have identical shapes. Torch would broadcast silently on a
    mismatch like `(B, D) - (B, 1)`, which is almost never intentional for a
    reconstruction residual — we assert shape equality to catch refactor drift.

    Note on cross-domain usage
    --------------------------
    SARL does **not** use this module: `SarlQNetwork.forward` computes the
    comparison inline via a tied-weight decoder
    (`flat_input - ReLU(W^T · hidden)`) and passes it directly to
    `SarlSecondOrderNetwork`. `ComparatorMatrix` is used only by Blindsight
    and AGL via `SecondOrderNetwork`.
    """

    def forward(
        self, first_order_input: torch.Tensor, first_order_output: torch.Tensor
    ) -> torch.Tensor:
        if first_order_input.shape != first_order_output.shape:
            raise ValueError(
                f"ComparatorMatrix expects matching shapes; got "
                f"{tuple(first_order_input.shape)} vs {tuple(first_order_output.shape)}. "
                "Silent broadcasting is disabled to catch refactor drift."
            )
        return first_order_input - first_order_output


class WageringHead(nn.Module):
    """Linear readout producing a wager (confidence) from the comparator output.

    The input `comparator_out` is **assumed to be already dropout-masked and
    cascade-integrated** by the caller (see `SecondOrderNetwork.forward`). This
    mirrors paper eq.2 (`C' = Dropout(C)`) and eq.6 (cascade), which happen
    before the wager readout.

    Parameters
    ----------
    input_dim : int
        Width of the comparator output (matches the first-order input dim).
    n_wager_units : int, default 1
        1 → single sigmoid confidence in [0, 1] (Blindsight/AGL reference code).
        2 → **raw logits** of shape `(B, 2)` (paper-faithful eq.3, Koch &
        Preuschoff 2007 style); the downstream loss is expected to apply its
        own per-unit sigmoid via `binary_cross_entropy_with_logits` (eq.5).
        No softmax is applied — see `docs/reproduction/deviations.md` D-001.
    hidden_dim : int, default 0
        If > 0, insert a `Linear(input_dim, hidden_dim) + ReLU` before the
        final readout — Pasquali & Cleeremans (2010) architecture (cited by
        paper §2.2 "as in Pasquali & Cleeremans (2010)"). Student code passes
        `hidden_2nd` but never uses it — this parameter restores that dropped
        layer. Default 0 preserves the student single-linear path (RG-002 H10).
    weight_init_range : tuple[float, float], default `DEFAULT_WAGER_INIT_RANGE`
        Uniform init range for the readout weights. Matches
        `Blindsight_TMLR.py:237`, `AGL_TMLR.py:239`, `sarl_maps.py:264`.
    """

    def __init__(
        self,
        input_dim: int,
        n_wager_units: int = 1,
        hidden_dim: int = 0,
        weight_init_range: tuple[float, float] = DEFAULT_WAGER_INIT_RANGE,
    ):
        super().__init__()
        if n_wager_units not in (1, 2):
            raise ValueError(
                f"n_wager_units must be 1 (reference code, sigmoid) or "
                f"2 (paper, raw logits); got {n_wager_units}"
            )
        self.n_wager_units = n_wager_units
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim > 0:
            self.hidden = nn.Linear(input_dim, self.hidden_dim)
            self.wager = nn.Linear(self.hidden_dim, n_wager_units)
        else:
            self.hidden = None
            self.wager = nn.Linear(input_dim, n_wager_units)
        init.uniform_(self.wager.weight, *weight_init_range)
        if self.hidden is not None:
            init.uniform_(self.hidden.weight, *weight_init_range)
        # Bias left at default PyTorch init (uniform from Kaiming-style bound);
        # matches reference code which never re-initialises bias either.

    def forward(self, comparator_out: torch.Tensor) -> torch.Tensor:
        x = comparator_out
        if self.hidden is not None:
            x = torch.relu(self.hidden(x))
        logits = self.wager(x)
        if self.n_wager_units == 1:
            return torch.sigmoid(logits)
        # n_wager_units == 2: paper-faithful raw logits (eq.3).
        # Downstream loss applies per-unit sigmoid (eq.5 BCE-with-logits).
        return logits


class SecondOrderNetwork(nn.Module):
    """Second-order network = Comparator + Dropout + (optional cascade) + WageringHead.

    The cascade is applied to the *comparator output* (post-dropout), not to
    the wager — matching Blindsight/AGL reference code. The caller manages the
    cascade state across iterations via `prev_comparison`.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the first-order input/output vectors being compared.
        Blindsight = 100, AGL = 48, SARL depends on environment.
    dropout : float, default 0.5
        Dropout rate on the comparator output. Reference value.
    n_wager_units : int, default 1
        Passed through to `WageringHead`. See class docstring for the
        code-vs-paper deviation (D-001).
    weight_init_range : tuple[float, float], default `DEFAULT_WAGER_INIT_RANGE`
        Passed through to `WageringHead`.

    Returns (from `forward`)
    -----------------------
    wager : torch.Tensor
        Shape `(batch, n_wager_units)`; confidence in [0, 1] (single-unit
        sigmoid) or raw logits (two-unit, paper-faithful eq.3 — downstream
        loss applies per-unit sigmoid via BCE-with-logits).
    comparison_out : torch.Tensor
        Shape `(batch, input_dim)`; the **post-cascade** comparator output.
        The caller MUST thread this back as `prev_comparison` on the next
        cascade iteration, otherwise the cascade cannot accumulate toward
        its steady state (see `docs/reviews/second_order.md` §C.5 (c)).
    """

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.5,
        n_wager_units: int = 1,
        hidden_dim: int = 0,
        weight_init_range: tuple[float, float] = DEFAULT_WAGER_INIT_RANGE,
    ):
        super().__init__()
        self.comparator = ComparatorMatrix()
        self.dropout = nn.Dropout(dropout)
        self.wagering_head = WageringHead(
            input_dim=input_dim,
            n_wager_units=n_wager_units,
            hidden_dim=hidden_dim,
            weight_init_range=weight_init_range,
        )

    def forward(
        self,
        first_order_input: torch.Tensor,
        first_order_output: torch.Tensor,
        prev_comparison: torch.Tensor | None,
        cascade_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        comparison_matrix = self.comparator(first_order_input, first_order_output)
        comparison_out = self.dropout(comparison_matrix)
        comparison_out = cascade_update(comparison_out, prev_comparison, cascade_rate)
        wager = self.wagering_head(comparison_out)
        return wager, comparison_out
