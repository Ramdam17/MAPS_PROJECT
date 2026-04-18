"""Frozen snapshots of the original AGL_TMLR.py networks.

Copied verbatim from AGL/AGL_TMLR.py (commit 4a738604).
- `ReferenceAGLFirstOrderNetwork` ← L134-203
- `ReferenceAGLSecondOrderNetwork` ← L211-256

`bits_per_letter=6` matches L1676 of the reference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

BITS_PER_LETTER = 6


class ReferenceAGLFirstOrderNetwork(nn.Module):
    """Verbatim copy of AGL_TMLR.py FirstOrderNetwork (L134-203)."""

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


class ReferenceAGLSecondOrderNetwork(nn.Module):
    """Verbatim copy of AGL_TMLR.py SecondOrderNetwork (L211-256).

    Note: `hidden_second` is unused (the reference hard-codes input_dim=48 on
    the wager layer). We keep the kwarg only for signature compatibility with
    the reference `prepare_pre_training`.
    """

    def __init__(self, use_gelu: bool, hidden_second: int):
        super().__init__()
        self.wager = nn.Linear(48, 1)
        self.dropout = nn.Dropout(0.5)
        if use_gelu:
            self.activation = torch.nn.GELU()
        else:
            self.activation = torch.nn.ReLU()
        self.sigmoid = torch.sigmoid
        self.softmax = nn.Softmax()
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
