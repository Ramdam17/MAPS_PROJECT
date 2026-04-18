"""Reference extracts from the paper's SARL MinAtar implementation.

Source: ``external/MinAtar/examples/maps.py``
Commit: ``ec5bcb7`` (refactor(external): consolidate MinAtar into external/)
Authors: Juan David Vargas et al., MAPS TMLR submission (2025).

Verbatim copies of `size_linear_unit`, `num_linear_units`, `QNetwork`,
`SecondOrderNetwork`, `replay_buffer` (+ `transition`), `get_state`, and
`target_wager` used as the **ground truth** for Sprint 04b parity tests.

These definitions are *immutable* — if the paper source changes, bump the
commit SHA and re-extract. Do not refactor, rename, or simplify any identifier
here even if ruff complains; that's what the per-file ignore at the bottom of
`pyproject.toml` is for.

Stripped vs source:
- No module-level ``NvidiaEnergyTracker`` instantiation (import side effect)
- No module-level ``print("using ", device)``
- ``device`` resolved to CPU (parity tests run on CPU; the paper device-switch
  does not affect forward-pass math at atol=1e-6)

Nothing else is altered — formatting, comment layout, variable capitalisation,
and (commented-out) lines are preserved exactly.
"""
# ruff: noqa: N801, N802, N803, N806, E741, B007, SIM108, UP008, UP032, RUF001
# flake8: noqa

from collections import namedtuple

import random

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init

# Parity tests pin CPU for reproducibility. The paper uses CUDA when available,
# but forward-pass math is device-independent at atol=1e-6 for the sizes here.
device = torch.device("cpu")


# ─── QNetwork support ────────────────────────────────────────────────────────
# Source: external/MinAtar/examples/maps.py:129-133

def size_linear_unit(size, kernel_size=3, stride=1):
    return (size - (kernel_size - 1) - 1) // stride + 1

num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16


# ─── QNetwork ────────────────────────────────────────────────────────────────
# Source: external/MinAtar/examples/maps.py:134-182

#default version
class QNetwork(nn.Module):
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

        #autoencoder
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        # Output layer:
        self.actions = nn.Linear(in_features=128, out_features=num_actions)

        #self.fc_comparison = nn.Linear(in_features=128, out_features=num_linear_units)


    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x , prev_h2, cascade_rate): #torch.Size([32, 4, 10, 10])

        x = f.relu(self.conv(x)) #torch.Size([32, 16, 8, 8])
        Input= x.view(x.size(0),-1)  # torch.Size([32, 1024])
        Hidden = f.relu(self.fc_hidden(Input))   # torch.Size([32, 128])

        if prev_h2 is not None:
            Hidden= cascade_rate*Hidden +  (1-cascade_rate)*prev_h2

        # Returns the output from the fully-connected linear layer
        x=self.actions(Hidden) # torch.Size([32, 6])

        Output_comparison = f.relu(f.linear(Hidden, self.fc_hidden.weight.t()))

        #Output_comparison = f.relu(self.fc_comparison(Hidden))   # torch.Size([32, 1024])

        Comparisson= Input - Output_comparison

        return x , Hidden , Comparisson, Hidden


# ─── SecondOrderNetwork ──────────────────────────────────────────────────────
# Source: external/MinAtar/examples/maps.py:245-278

class SecondOrderNetwork(nn.Module):
    def __init__(self, in_channels):
        super(SecondOrderNetwork, self).__init__()

        # Define a linear layer for comparing the difference between input and output of the first-order network
        #self.comparison_layer = nn.Linear(in_features=num_linear_units, out_features=128)

        # Linear layer for determining wagers
        self.wager = nn.Linear(num_linear_units, 2)
        self.dropout = nn.Dropout(p=0.1)  # 10% dropout

        self.sigmoid = torch.sigmoid

        # Initialize the weights of the network
        self._init_weights()

    def _init_weights(self):
        # Kaiming initialization for stability
        #init.uniform_(self.comparison_layer.weight, -1.0, 1.0)
        init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(self, comparison_matrix, prev_comparison, cascade_rate):

        # Pass the input through the comparison layer and apply dropout and activation
        #comparison_out = self.dropout(f.relu(self.comparison_layer(comparison_matrix)))
        comparison_out= self.dropout(comparison_matrix)

        if prev_comparison is not None:
          comparison_out = cascade_rate * comparison_out + (1 - cascade_rate) * prev_comparison

        # Pass through wager layer
        wager = self.wager(comparison_out)

        return wager ,  comparison_out


# ─── replay_buffer ───────────────────────────────────────────────────────────
# Source: external/MinAtar/examples/maps.py:289-307

transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


# ─── get_state ───────────────────────────────────────────────────────────────
# Source: external/MinAtar/examples/maps.py:322-323

def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


# ─── target_wager ────────────────────────────────────────────────────────────
# Source: external/MinAtar/examples/maps.py:514-532

def target_wager(rewards, alpha):
    flattened_rewards = rewards.view(-1)  # Flatten rewards to 1D for easy indexing
    alpha = float(alpha/100)  # EMA hyperparameter
    EMA = 0.0

    batch_size = rewards.size(0)  # Get the batch size (first dimension of rewards)
    new_tensor = torch.zeros(batch_size, 2, device=rewards.device)  # Create [batch_size, 2] tensor on the same device

    for i in range(batch_size):
        G = flattened_rewards[i]  # Current reward
        EMA = alpha * G + (1 - alpha) * EMA  # Update EMA

        # Set values based on comparison with EMA
        if G > EMA:
            new_tensor[i] = torch.tensor([1, 0], device=rewards.device)
        else:
            new_tensor[i] = torch.tensor([0, 1], device=rewards.device)

    return new_tensor
