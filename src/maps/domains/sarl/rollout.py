"""SARL environment interaction — ε-greedy + cascade action selection.

Ported verbatim from ``external/paper_reference/sarl/maps_v1.py``:
``world_dynamics`` L437-470 (Mnih et al. 2015 ε-greedy behaviour policy).

RNG-consuming calls are preserved bit-for-bit for parity:
- ``numpy.random.binomial(1, epsilon)`` decides explore vs exploit;
- ``random.randrange(num_actions)`` picks the random action.

ε is annealed **linearly** from ``start_epsilon`` (1.0) to ``end_epsilon`` (0.1)
over ``first_n_frames`` (100k) after ``replay_start_size`` (5k), then held flat.

References
----------
Vargas et al. (2025), MAPS, TMLR submission §3.
Mnih et al. (2015). Human-level control through deep reinforcement learning.
"""

from __future__ import annotations

import random

import numpy
import torch
from torch import Tensor, nn

from maps.domains.sarl.data import get_state


def epsilon_at(
    t: int,
    replay_start_size: int,
    *,
    first_n_frames: int = 100_000,
    start_epsilon: float = 1.0,
    end_epsilon: float = 0.1,
) -> float:
    """Linearly-annealed exploration ε (verbatim schedule, source L449-450)."""
    if t - replay_start_size >= first_n_frames:
        return end_epsilon
    return ((end_epsilon - start_epsilon) / first_n_frames) * (
        t - replay_start_size
    ) + start_epsilon


def select_greedy_action(policy_net: nn.Module, s: Tensor, cascade_iterations_1: int) -> Tensor:
    """Cascade the policy net ``cascade_iterations_1`` times, return argmax action.

    Verbatim greedy branch (source L458-462): no-grad, cascade on the Output,
    ``argmax_a Q(s, a)`` shaped ``(1, 1)``.
    """
    cascade_rate_1 = float(1.0 / cascade_iterations_1)
    main_task_out = None
    with torch.no_grad():
        for _ in range(cascade_iterations_1):
            q, _hn, _comparison, main_task_out = policy_net(s, main_task_out, cascade_rate_1)
        return q.max(1)[1].view(1, 1)


def world_dynamics(
    t: int,
    replay_start_size: int,
    num_actions: int,
    s: Tensor,
    env,
    policy_net: nn.Module,
    cascade_iterations_1: int,
    *,
    first_n_frames: int = 100_000,
    start_epsilon: float = 1.0,
    end_epsilon: float = 0.1,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One environment step. Verbatim ``world_dynamics`` L437-470.

    Returns ``(s_prime, action, reward, is_terminal)`` as tensors. A uniform
    random policy runs while ``t < replay_start_size`` (buffer warm-up); after
    that, ε-greedy with the annealed schedule.
    """
    if t < replay_start_size:
        action = torch.tensor([[random.randrange(num_actions)]], device=device)
    else:
        epsilon = epsilon_at(
            t,
            replay_start_size,
            first_n_frames=first_n_frames,
            start_epsilon=start_epsilon,
            end_epsilon=end_epsilon,
        )
        # Legacy global-RNG binomial is deliberate — RNG-stream parity with the
        # source (maps_v1.py). Do NOT switch to np.random.Generator (NPY002).
        if numpy.random.binomial(1, epsilon) == 1:  # noqa: NPY002
            action = torch.tensor([[random.randrange(num_actions)]], device=device)
        else:
            action = select_greedy_action(policy_net, s, cascade_iterations_1)

    reward, terminated = env.act(action)
    s_prime = get_state(env.state(), device=device)
    return (
        s_prime,
        action,
        torch.tensor([[reward]], device=device).float(),
        torch.tensor([[terminated]], device=device),
    )
