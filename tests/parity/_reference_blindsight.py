"""Frozen snapshot of the original Blindsight_TMLR.py network classes.

Copied verbatim from BLINDSIGHT/Blindsight_TMLR.py (commit 4a738604, lines
136-252) to serve as a behavioral reference for parity tests. Do not modify —
the whole point is that this is the ground-truth implementation that new
`maps.components` / `maps.networks` code must reproduce bit-for-bit.

Deviations from the TMLR paper (documented in docs/reproduction/deviations.md):
- `wager` is a single linear unit (n_out=1) → sigmoid, not the 2-unit softmax
  over {bet, no-bet} described in paper eq.2-3.
- First-order decoder applies a single global sigmoid on the 100-d output
  (Blindsight is perceptual, not discrete-symbol like AGL).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init


class ReferenceFirstOrderNetwork(nn.Module):
    """Verbatim copy of Blindsight FirstOrderNetwork (lines 136-206)."""

    def __init__(self, hidden_units: int, data_factor: int, use_gelu: bool):
        super().__init__()
        self.fc1 = nn.Linear(100, hidden_units, bias=False)  # Encoder
        self.fc2 = nn.Linear(hidden_units, 100, bias=False)  # Decoder
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
        self.data_factor = data_factor
        self.initialize_weights()

    def initialize_weights(self):
        init.uniform_(self.fc1.weight, -1.0, 1.0)
        init.uniform_(self.fc2.weight, -1.0, 1.0)

    def encoder(self, x):
        return self.dropout(self.relu(self.fc1(x.view(-1, 100))))

    def decoder(self, z, prev_h2, cascade_rate):
        h2 = self.sigmoid(self.fc2(z))
        if prev_h2 is not None:
            h2 = cascade_rate * h2 + (1 - cascade_rate) * prev_h2
        return h2

    def forward(self, x, prev_h1, prev_h2, cascade_rate):
        h1 = self.encoder(x)
        h2 = self.decoder(h1, prev_h2, cascade_rate)
        return h1, h2


class ReferenceSecondOrderNetwork(nn.Module):
    """Verbatim copy of Blindsight SecondOrderNetwork (lines 213-252)."""

    def __init__(self, use_gelu: bool, hidden_2nd: int):
        super().__init__()
        self.wager = nn.Linear(100, 1)
        self.dropout = nn.Dropout(0.5)
        self.activation = torch.relu
        self.sigmoid = torch.sigmoid
        self._init_weights()

    def _init_weights(self):
        init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(self, first_order_input, first_order_output, prev_comparison, cascade_rate):
        comparison_matrix = first_order_input - first_order_output
        comparison_out = self.dropout(comparison_matrix)
        if prev_comparison is not None:
            comparison_out = cascade_rate * comparison_out + (1 - cascade_rate) * prev_comparison
        wager = self.sigmoid(self.wager(comparison_out))
        return wager, comparison_out
