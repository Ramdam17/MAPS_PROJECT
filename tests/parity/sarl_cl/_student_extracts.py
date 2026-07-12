"""Verbatim extracts from ``external/paper_reference/sarl/maps_v1.py`` (v1).

Reference implementations behind the paper Table 6 SARL numbers. The Tier-1
parity test compares our port (``maps.domains.sarl.data``) against these
bit-by-bit under an identical seed. Only cosmetic deltas (module ``device``
global → ``cpu``) are applied; the maths and RNG-consuming calls are identical.
"""

# ruff: noqa  (verbatim student extract — do not lint/refactor)
from __future__ import annotations

import random
from collections import namedtuple

import torch

transition = namedtuple("transition", "state, next_state, action, reward, is_terminal")


class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


def get_state(s, device="cpu"):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


def target_wager(rewards, alpha):
    flattened_rewards = rewards.view(-1)
    alpha = float(alpha / 100)
    EMA = 0.0
    batch_size = rewards.size(0)
    new_tensor = torch.zeros(batch_size, 2, device=rewards.device)
    for i in range(batch_size):
        G = flattened_rewards[i]
        EMA = alpha * G + (1 - alpha) * EMA
        if G > EMA:
            new_tensor[i] = torch.tensor([1, 0], device=rewards.device)
        else:
            new_tensor[i] = torch.tensor([0, 1], device=rewards.device)
    return new_tensor


# ── Verbatim networks (maps_v1.py:110-253) ──────────────────────────────────
import torch.nn.functional as f  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import init  # noqa: E402


def size_linear_unit(size, kernel_size=3, stride=1):
    return (size - (kernel_size - 1) - 1) // stride + 1


num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16


class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
        self.fc_output = nn.Linear(in_features=128, out_features=num_linear_units)
        self.actions = nn.Linear(in_features=num_linear_units, out_features=num_actions)

    def forward(self, x, prev_h2, cascade_rate):
        x = f.relu(self.conv(x))
        Input = x.view(x.size(0), -1)
        Hidden = f.relu(self.fc_hidden(Input))
        Output = f.relu(self.fc_output(Hidden))
        if prev_h2 is not None:
            Output = cascade_rate * Output + (1 - cascade_rate) * prev_h2
        x = self.actions(Output)
        Comparisson = Input - Output
        return x, Hidden, Comparisson, Output


class SecondOrderNetwork(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.comparison_layer = nn.Linear(
            in_features=num_linear_units, out_features=num_linear_units
        )
        self.wager = nn.Linear(num_linear_units, 2)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        init.uniform_(self.comparison_layer.weight, -1.0, 1.0)
        init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(self, comparison_matrix, prev_comparison, cascade_rate):
        comparison_out = self.dropout(f.relu(self.comparison_layer(comparison_matrix)))
        if prev_comparison is not None:
            comparison_out = cascade_rate * comparison_out + (1 - cascade_rate) * prev_comparison
        wager = self.wager(comparison_out)
        return wager, comparison_out


# ── Verbatim CAE_loss (maps_v1.py:305-354; huber recon active) ──────────────
def CAE_loss(W, x, recons_x, h, lam):
    mse = f.huber_loss(recons_x, x)
    dh = h * (1 - h)
    w_sum = torch.sum(W**2, dim=1)
    w_sum = w_sum.unsqueeze(1)
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)


# ── Verbatim AdaptiveQNetwork (sarl_cl_maps.py:160-219) ─────────────────────
import numpy  # noqa: E402


class AdaptiveQNetwork(nn.Module):
    def __init__(self, max_input_channels, num_actions):
        super().__init__()
        self.max_input_channels = max_input_channels
        self.input_adapter = nn.Sequential(
            nn.Conv2d(max_input_channels, max_input_channels, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.conv = nn.Conv2d(max_input_channels, 16, kernel_size=3, stride=1)
        conv_output_size = self._get_conv_output_size((max_input_channels, 10, 10))
        self.fc_hidden = nn.Linear(in_features=conv_output_size, out_features=128)
        self.fc_output = nn.Linear(in_features=128, out_features=conv_output_size)
        self.actions = nn.Linear(in_features=conv_output_size, out_features=num_actions)

    def _get_conv_output_size(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output = self.conv(input)
        return int(numpy.prod(output.size()[1:]))

    def adapt_input(self, x):
        if x.size(1) < self.max_input_channels:
            padding = torch.zeros(
                x.size(0),
                self.max_input_channels - x.size(1),
                x.size(2),
                x.size(3),
                device=x.device,
            )
            x = torch.cat([x, padding], dim=1)
        return x

    def forward(self, x, prev_h2, cascade_rate):
        x = self.adapt_input(x)
        x = self.input_adapter(x)
        x = f.relu(self.conv(x))
        Input = x.view(x.size(0), -1)
        Hidden = f.relu(self.fc_hidden(Input))
        Output = f.relu(self.fc_output(Hidden))
        if prev_h2 is not None:
            Output = cascade_rate * Output + (1 - cascade_rate) * prev_h2
        x = self.actions(Output)
        Comparison = Input - Output
        return x, Hidden, Comparison, Output
