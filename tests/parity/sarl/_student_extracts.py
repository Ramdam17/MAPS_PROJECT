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
