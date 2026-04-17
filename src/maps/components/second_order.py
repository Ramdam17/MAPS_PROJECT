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
The paper describes a 2-unit softmax wagering head (high-bet / low-bet; Koch &
Preuschoff 2007 style). The reference implementations in both
`Blindsight_TMLR.py` and `AGL_TMLR.py` use a single sigmoid unit instead (a
scalar confidence in [0, 1]). We default to `n_wager_units=1` for parity with
that reference code; set `n_wager_units=2` to get the paper-faithful variant.
See `docs/reproduction/deviations.md` for rationale.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

from maps.components.cascade import cascade_update


class ComparatorMatrix(nn.Module):
    """Element-wise difference `C = first_order_input - first_order_output`.

    Stateless by design — kept as an `nn.Module` only for symmetry with the
    rest of the architecture (you can swap it for a learned comparator later
    without rewiring the second-order network).
    """

    def forward(
        self, first_order_input: torch.Tensor, first_order_output: torch.Tensor
    ) -> torch.Tensor:
        return first_order_input - first_order_output


class WageringHead(nn.Module):
    """Linear readout producing a wager (confidence) from the comparator output.

    Parameters
    ----------
    input_dim : int
        Width of the comparator output (matches the first-order input dim).
    n_wager_units : int, default 1
        1 → single sigmoid confidence (Blindsight/AGL reference code).
        2 → softmax over {bet, no-bet} (paper-faithful, see Koch & Preuschoff 2007).
    weight_init_range : tuple[float, float], default (0.0, 0.1)
        Uniform init range for the readout weights. Matches
        `Blindsight_TMLR.py:237` and `AGL_TMLR.py:239`.
    """

    def __init__(
        self,
        input_dim: int,
        n_wager_units: int = 1,
        weight_init_range: tuple[float, float] = (0.0, 0.1),
    ):
        super().__init__()
        if n_wager_units not in (1, 2):
            raise ValueError(
                f"n_wager_units must be 1 (reference code) or 2 (paper); got {n_wager_units}"
            )
        self.n_wager_units = n_wager_units
        self.wager = nn.Linear(input_dim, n_wager_units)
        init.uniform_(self.wager.weight, *weight_init_range)
        # Bias left at default PyTorch init (uniform from Kaiming-style bound);
        # matches reference code which never re-initialises bias either.

    def forward(self, comparator_out: torch.Tensor) -> torch.Tensor:
        logits = self.wager(comparator_out)
        if self.n_wager_units == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)


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
        code-vs-paper deviation.
    weight_init_range : tuple[float, float], default (0.0, 0.1)
        Passed through to `WageringHead`.

    Returns (from `forward`)
    -----------------------
    wager : torch.Tensor
        Shape `(batch, n_wager_units)`; confidence in [0, 1] (single-unit) or
        softmax over {bet, no-bet} (two-unit).
    comparison_out : torch.Tensor
        Shape `(batch, input_dim)`; the (post-cascade) comparator output, to
        be threaded back as `prev_comparison` on the next cascade step.
    """

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.5,
        n_wager_units: int = 1,
        weight_init_range: tuple[float, float] = (0.0, 0.1),
    ):
        super().__init__()
        self.comparator = ComparatorMatrix()
        self.dropout = nn.Dropout(dropout)
        self.wagering_head = WageringHead(
            input_dim=input_dim,
            n_wager_units=n_wager_units,
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
