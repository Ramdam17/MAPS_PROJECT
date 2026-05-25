"""Paper-faithful reference for SARL v1 parity tests.

Origin
------
Extracted verbatim from ``external/paper_reference/sarl/maps_v1.py``
(restored by Rémy 2026-05-24 in commit ``6c59744``, from Juan's fork
pre-commit ``8aa1138``).

``maps_v1.py`` is invoked by ``SARL_Training_Standard.sh`` with
``base=2000000`` frames and ``-ema 25`` (α=0.25), making it the
canonical paper source for Table 6 numbers.

Stripped vs original source
---------------------------
- No module-level ``NvidiaEnergyTracker`` instantiation (import side effect).
- No module-level ``print("using ", device)``.
- ``device`` resolved to CPU (parity tests run on CPU; forward-pass math is
  device-independent at atol=1e-6 for MinAtar sizes).
- No argparse / training loop / dqn() / plot() / evaluation() — only the
  two network classes needed for Tier-1 parity.

Identifier preservation
-----------------------
The class names, variable names (``Hidden``, ``Output``, ``Comparisson``
typo, etc.) and comment layout from the v1 source are preserved exactly
to keep diffs against the original ``maps_v1.py`` readable. ``ruff: noqa``
tolerates the paper naming style.
"""

# ruff: noqa: UP008

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init


def size_linear_unit(size, kernel_size=3, stride=1):
    return (size - (kernel_size - 1) - 1) // stride + 1


num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16


class QNetwork(nn.Module):
    """v1 QNetwork — verbatim from ``maps_v1.py:117-159`` (stripped)."""

    def __init__(self, in_channels, num_actions):
        super(QNetwork, self).__init__()

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units

        self.sigmoid = nn.Sigmoid()

        # autoencoder
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
        self.fc_output = nn.Linear(in_features=128, out_features=num_linear_units)

        # Output layer:
        self.actions = nn.Linear(in_features=num_linear_units, out_features=num_actions)

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x, prev_h2, cascade_rate):  # torch.Size([32, 4, 10, 10])
        x = f.relu(self.conv(x))  # torch.Size([32, 16, 8, 8])
        Input = x.view(x.size(0), -1)  # torch.Size([32, 1024])
        Hidden = f.relu(self.fc_hidden(Input))  # torch.Size([32, 128])
        Output = f.relu(self.fc_output(Hidden))  # torch.Size([32, 1024])

        if prev_h2 is not None:
            Output = cascade_rate * Output + (1 - cascade_rate) * prev_h2

        # Returns the output from the fully-connected linear layer
        x = self.actions(Output)  # torch.Size([32, 6])
        Comparisson = Input - Output

        return x, Hidden, Comparisson, Output


class SecondOrderNetwork(nn.Module):
    """v1 SecondOrderNetwork — verbatim from ``maps_v1.py:221-253`` (stripped)."""

    def __init__(self, in_channels):
        super(SecondOrderNetwork, self).__init__()

        # Define a linear layer for comparing the difference between input and output of the first-order network
        self.comparison_layer = nn.Linear(
            in_features=num_linear_units, out_features=num_linear_units
        )

        # Linear layer for determining wagers
        self.wager = nn.Linear(num_linear_units, 2)
        self.dropout = nn.Dropout(p=0.1)  # 10% dropout
        self.softmax = nn.Softmax()  # Specify dimension for Softmax
        self.sigmoid = nn.Sigmoid()

        # Initialize the weights of the network
        self._init_weights()

    def _init_weights(self):
        # Kaiming initialization for stability
        init.uniform_(self.comparison_layer.weight, -1.0, 1.0)
        init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(self, comparison_matrix, prev_comparison, cascade_rate):
        # Pass the input through the comparison layer and apply dropout and activation
        comparison_out = self.dropout(f.relu(self.comparison_layer(comparison_matrix)))

        if prev_comparison is not None:
            comparison_out = cascade_rate * comparison_out + (1 - cascade_rate) * prev_comparison

        # Pass through wager layer
        wager = self.wager(comparison_out)

        return wager, comparison_out
