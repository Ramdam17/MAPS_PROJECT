"""SARL replay buffer, state helper, and wager target — MAPS §3, Table 6/11.

Ported verbatim from ``external/paper_reference/sarl/maps_v1.py`` (the v1
paper-canonical source, D14.1):
- ``transition`` namedtuple  L264
- ``replay_buffer``          L265-282  (Mnih et al. 2015 cyclic buffer)
- ``get_state``              L297-298
- ``target_wager``           L489-507

The wager target is SARL-specific: for each reward ``G`` in the sampled batch,
maintain a scalar EMA of the rewards and label ``[1, 0]`` ("bet") when
``G > EMA`` else ``[0, 1]`` ("no-bet") — a *relative-surprise* signal ("did I
beat my running average?"). ``alpha`` is passed in **percent** (e.g. 25 → 0.25,
the v1 paper value) and divided by 100 inside.

The buffer samples with Python ``random.sample`` (not ``np.random`` / torch) —
load-bearing for RNG-stream parity with the source.

References
----------
Vargas et al. (2025), MAPS, TMLR submission §3, Table 6.
Mnih et al. (2015). Human-level control through deep reinforcement learning.
"""

from __future__ import annotations

import random
from collections import namedtuple

import torch
from torch import Tensor

# Field order is fixed (source L264) — parity convention.
Transition = namedtuple("Transition", "state, next_state, action, reward, is_terminal")


class SarlReplayBuffer:
    """Cyclic fixed-size replay buffer (Mnih et al. 2015). Verbatim L265-282."""

    def __init__(self, buffer_size: int) -> None:
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer: list[Transition] = []

    def add(self, *args) -> None:
        # Append while not full, overwrite cyclically once full.
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.location] = Transition(*args)
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size: int) -> list[Transition]:
        # Python random.sample — NOT np.random / torch (RNG-stream parity).
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def get_state(s, *, device: torch.device | str = "cpu") -> Tensor:
    """MinAtar numpy state ``(10, 10, C)`` → tensor ``(1, C, 10, 10)`` float.

    Verbatim ``get_state`` L297-298 (permute + unsqueeze + float).
    """
    return torch.tensor(s, device=device).permute(2, 0, 1).unsqueeze(0).float()


def target_wager(rewards: Tensor, alpha: float) -> Tensor:
    """EMA-surprise wager target, shape ``(B, 2)``. Verbatim ``target_wager`` L489-507.

    ``alpha`` is in **percent** (25 → 0.25). For each reward ``G`` in batch order,
    ``EMA ← α·G + (1-α)·EMA``; label ``[1, 0]`` if ``G > EMA`` else ``[0, 1]``.
    The EMA is a scalar recurrence over the (arbitrarily ordered) sampled batch —
    a documented quirk of the source, preserved.
    """
    flattened_rewards = rewards.view(-1)
    alpha = float(alpha / 100)  # EMA hyperparameter
    ema = 0.0

    batch_size = rewards.size(0)
    new_tensor = torch.zeros(batch_size, 2, device=rewards.device)

    for i in range(batch_size):
        g = flattened_rewards[i]
        ema = alpha * g + (1 - alpha) * ema
        if g > ema:
            new_tensor[i] = torch.tensor([1, 0], device=rewards.device)
        else:
            new_tensor[i] = torch.tensor([0, 1], device=rewards.device)

    return new_tensor
