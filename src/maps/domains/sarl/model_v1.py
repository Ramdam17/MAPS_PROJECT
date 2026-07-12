"""SARL v1 networks — MAPS §2.1/§3, Table 6 (paper-canonical).

Ported verbatim from ``external/paper_reference/sarl/maps_v1.py`` (v1, D14.1):
- ``QNetwork``           L117-158  → :class:`SarlQNetworkV1`
- ``SecondOrderNetwork`` L221-253  → :class:`SarlSecondOrderNetworkV1`

The **v1** architecture (distinct from the v2 wrong-variant — post-mortem
``docs/reproduction/sarl-postmortem-20260524.md``):
- Q-head reads the **1024-d Output** (post-decoder), not the 128-d Hidden.
- dedicated reconstruction decoder ``Linear(128, 1024)`` (v2 tied the weights).
- second-order ``comparison_layer = Linear(1024, 1024)`` is **active** (v2
  commented it out).
- cascade accumulates on the **1024-d Output** (v2 cascaded the 128-d Hidden).

The cascade blend reuses :func:`maps.core.cascade.cascade_update`
(``rate·new + (1-rate)·prev``; bootstraps to ``new`` when ``prev is None``) —
numerically identical to the source's inline expression.

References
----------
Vargas et al. (2025), MAPS, TMLR submission §3, Table 6.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init

from maps.core.cascade import cascade_update


def size_linear_unit(size: int, kernel_size: int = 3, stride: int = 1) -> int:
    """Conv output side length (verbatim ``size_linear_unit`` L111-112)."""
    return (size - (kernel_size - 1) - 1) // stride + 1


# MinAtar states are 10×10; conv(3,1) → 8×8 × 16 channels = 1024 (source L114).
NUM_LINEAR_UNITS = size_linear_unit(10) * size_linear_unit(10) * 16


class SarlQNetworkV1(nn.Module):
    """First-order Q-network (v1). Verbatim ``QNetwork`` L117-158."""

    def __init__(self, in_channels: int, num_actions: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        # Autoencoder: 1024 → 128 (hidden) → 1024 (output, dedicated decoder).
        self.fc_hidden = nn.Linear(in_features=NUM_LINEAR_UNITS, out_features=128)
        self.fc_output = nn.Linear(in_features=128, out_features=NUM_LINEAR_UNITS)
        # Q-head reads the 1024-d Output (v1).
        self.actions = nn.Linear(in_features=NUM_LINEAR_UNITS, out_features=num_actions)

    def forward(
        self, x: Tensor, prev_h2: Tensor | None, cascade_rate: float
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = F.relu(self.conv(x))
        inp = x.view(x.size(0), -1)  # (B, 1024)
        hidden = F.relu(self.fc_hidden(inp))  # (B, 128)
        output = F.relu(self.fc_output(hidden))  # (B, 1024)
        output = cascade_update(output, prev_h2, cascade_rate)  # cascade on Output
        q = self.actions(output)  # (B, num_actions)
        comparison = inp - output  # (B, 1024)
        return q, hidden, comparison, output


class SarlSecondOrderNetworkV1(nn.Module):
    """Second-order comparator + wager (v1). Verbatim ``SecondOrderNetwork`` L221-253."""

    def __init__(self, in_channels: int | None = None) -> None:
        super().__init__()
        # Active 1024×1024 comparison layer (v1) — v2 commented this out.
        self.comparison_layer = nn.Linear(NUM_LINEAR_UNITS, NUM_LINEAR_UNITS)
        self.wager = nn.Linear(NUM_LINEAR_UNITS, 2)
        self.dropout = nn.Dropout(p=0.1)
        self._init_weights()

    def _init_weights(self) -> None:
        init.uniform_(self.comparison_layer.weight, -1.0, 1.0)
        init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(
        self, comparison_matrix: Tensor, prev_comparison: Tensor | None, cascade_rate: float
    ) -> tuple[Tensor, Tensor]:
        comparison_out = self.dropout(F.relu(self.comparison_layer(comparison_matrix)))
        comparison_out = cascade_update(comparison_out, prev_comparison, cascade_rate)
        wager = self.wager(comparison_out)  # raw logits, (B, 2)
        return wager, comparison_out
