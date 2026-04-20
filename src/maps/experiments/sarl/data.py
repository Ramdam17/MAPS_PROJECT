"""SARL replay buffer and state-tensor conversion.

Ports ``replay_buffer`` (class) + ``get_state`` (helper) + the ``transition``
namedtuple from ``external/MinAtar/examples/maps.py`` (commit ``ec5bcb7``).

The sampling algorithm is **intentionally identical** to the paper's: a cyclic
Python list storing ``namedtuple`` transitions, with ``random.sample`` drawing
without replacement from Python's ``random`` module. This matters for parity:

* Swapping to ``np.random.choice`` or ``torch.randperm`` would consume a
  different RNG stream, so even with the same seed the returned transitions
  would differ.
* Storing transitions as a tensor (e.g. ``torch.stack`` on every ``add``)
  would change insertion semantics under wrap-around and break the cyclic
  overwrite index math.

The field order of ``Transition`` is fixed:
``(state, next_state, action, reward, is_terminal)`` — matching the paper's
``transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')``.

References
----------
- Mnih et al. (2015). Human-level control through deep reinforcement learning.
  Nature 518:529-533 — cyclic replay buffer.
- Vargas et al. (2025), MAPS TMLR submission §3.
"""

from __future__ import annotations

import random
from collections import namedtuple
from typing import Any

import torch

# Field order fixed for parity with paper — do not reorder.
Transition = namedtuple("Transition", "state, next_state, action, reward, is_terminal")


class SarlReplayBuffer:
    """Cyclic replay buffer with ``random.sample``-based uniform sampling.

    Parameters
    ----------
    buffer_size : int
        Maximum number of transitions retained. Older transitions are
        overwritten once capacity is reached (FIFO with a cyclic write head).

    Notes
    -----
    Uses Python's top-level ``random`` module for sampling, matching the
    paper. To make sampling deterministic across a run, call
    ``maps.utils.seeding.set_all_seeds(seed)`` before the first ``.sample()``
    call — that seeds ``random``, ``numpy``, and ``torch`` simultaneously.
    """

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer: list[Transition] = []

    def add(self, *args: Any) -> None:
        """Append a transition; overwrite the oldest slot once full.

        Arguments are forwarded positionally to ``Transition(*args)`` — see
        the field order in the module docstring.
        """
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.location] = Transition(*args)
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size: int) -> list[Transition]:
        """Uniformly sample ``batch_size`` transitions without replacement."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def get_state(s: Any, device: torch.device | str = "cpu") -> torch.Tensor:
    """Convert a MinAtar numpy state to a (1, C, 10, 10) float tensor.

    Paper source: ``maps.py:322-323``. The paper's ``get_state`` hard-wires
    the device to a module-level ``device`` global; we accept it as a kwarg
    with a CPU default so the helper is usable in tests without leaking CUDA
    assumptions.
    """
    return torch.tensor(s, device=device).permute(2, 0, 1).unsqueeze(0).float()


def target_wager(rewards: torch.Tensor, alpha: float) -> torch.Tensor:
    """EMA-based target wager label for the second-order network.

    Paper source: ``external/MinAtar/examples/maps.py:514-532``.

    The paper passes ``alpha`` in *percent* (e.g. ``alpha=45`` → 0.45 EMA
    smoothing factor) — this function divides by 100 internally, exactly as
    the paper does. Do not pre-scale ``alpha`` at the call site.

    For each reward ``G`` in the batch, compute the running EMA
    ``EMA_i = α·G_i + (1-α)·EMA_{i-1}`` with ``EMA_0 = 0`` then label
    ``[1, 0]`` if ``G > EMA`` (bet) else ``[0, 1]`` (no-bet). Return shape:
    ``(batch, 2)``.

    Shape contract
    --------------
    ``rewards`` must be shape ``(B,)`` or ``(B, 1)``. Multi-dim inputs with
    ``rewards.shape = (B, k>1)`` will be silently truncated because the
    function reads ``rewards.size(0)`` as batch size and then indexes into
    a flat view — the last ``B·(k-1)`` elements are ignored. Same behaviour
    as the student reference; documented to prevent a silent correctness bug.

    Implementation
    --------------
    The EMA recurrence is scalar and depends on the previous value, so it
    must be computed sequentially. We keep the Python loop for the EMA
    update (bit-parity with student) but vectorise the label construction
    via boolean indexing on the final EMA series — removes 2·B intermediate
    ``torch.tensor([1, 0])`` allocations without changing numerics.
    """
    flattened_rewards = rewards.view(-1)
    scaled_alpha = float(alpha / 100)
    batch_size = rewards.size(0)

    # Sequential EMA scan — required by the scalar recurrence.
    # Keeping the Python loop (rather than a cumsum-based vectorisation)
    # preserves bit-parity with the student reference `target_wager` used
    # in `tests/parity/sarl/_reference_sarl.py:189`.
    #
    # EMA series is always float (default torch dtype). The student builds
    # `new_tensor` with `torch.zeros(batch, 2, device=...)` (default float32)
    # then assigns int `torch.tensor([1, 0])` into it — the assignment casts
    # to float32. We match that by pinning float32 explicitly here so that
    # int-valued `rewards` (typical in MinAtar) don't silently truncate the
    # float EMA updates.
    ema_series = torch.zeros(batch_size, device=rewards.device, dtype=torch.float32)
    ema: torch.Tensor | float = 0.0
    for i in range(batch_size):
        ema = scaled_alpha * flattened_rewards[i] + (1 - scaled_alpha) * ema
        ema_series[i] = ema  # type: ignore[assignment]

    # Vectorised label construction: [1, 0] where G > EMA, else [0, 1].
    # Comparison promotes int rewards to float via ema_series.dtype — bit-exact
    # with the student loop which also compared float EMA against int G.
    bet_mask = (flattened_rewards > ema_series).to(torch.float32)  # (B,)
    new_tensor = torch.zeros(batch_size, 2, device=rewards.device, dtype=torch.float32)
    new_tensor[:, 0] = bet_mask
    new_tensor[:, 1] = 1.0 - bet_mask
    return new_tensor
