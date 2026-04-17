"""Frozen snapshot of the original AGL_TMLR.py FirstOrderNetwork.

Copied verbatim from AGL/AGL_TMLR.py (commit 4a738604, lines 134-203).
`bits_per_letter=6` matches line 1676 of the reference.

Only FirstOrderNetwork is mirrored here because the AGL SecondOrderNetwork
differs from Blindsight only in its input dimension (48 vs 100) — the logic
is identical and is already covered by the Blindsight parity test.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.init as init

BITS_PER_LETTER = 6


class ReferenceAGLFirstOrderNetwork(nn.Module):
    """Verbatim copy of AGL_TMLR.py FirstOrderNetwork (lines 134-203)."""

    def __init__(self, hidden_units: int, data_factor: int, use_gelu: bool):
        super().__init__()
        self.fc1 = nn.Linear(48, hidden_units, bias=False)
        self.fc2 = nn.Linear(hidden_units, 48, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
        self.data_factor = data_factor
        self.initialize_weights()

    def initialize_weights(self):
        init.uniform_(self.fc1.weight, -1.0, 1.0)
        init.uniform_(self.fc2.weight, -1.0, 1.0)

    def encoder(self, x):
        return self.dropout(self.relu(self.fc1(x)))

    def decoder(self, z, prev_h2, cascade_rate):
        h2 = self.fc2(z)
        for i in range(0, h2.size(1), BITS_PER_LETTER):
            h2[:, i : i + BITS_PER_LETTER] = self.sigmoid(h2[:, i : i + BITS_PER_LETTER])
        if prev_h2 is not None:
            h2 = cascade_rate * h2 + (1 - cascade_rate) * prev_h2
        return h2

    def forward(self, x, prev_h1, prev_h2, cascade_rate):
        h1 = self.encoder(x)
        h2 = self.decoder(h1, prev_h2, cascade_rate)
        return h1, h2
