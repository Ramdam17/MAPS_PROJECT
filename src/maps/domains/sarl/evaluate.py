"""SARL greedy-policy evaluation — MAPS §3.

Runs the trained policy net greedily (no exploration, no learning) for a fixed
number of episodes and returns the mean/std episode return. Distilled from the
``evaluation`` function of ``external/paper_reference/sarl/maps_v1.py`` L862
(the continual-learning bookkeeping there is out of scope for SARL).

References
----------
Vargas et al. (2025), MAPS, TMLR submission §3, Table 6.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from maps.domains.sarl.data import get_state
from maps.domains.sarl.rollout import select_greedy_action

logger = logging.getLogger(__name__)


def evaluate_greedy(
    policy_net: nn.Module,
    env,
    *,
    n_episodes: int,
    cascade_iterations_1: int,
    max_steps_per_episode: int = 100_000,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Play ``n_episodes`` greedily; return mean/std/min/max episode return."""
    policy_net.eval()
    returns: list[float] = []
    for _ in range(n_episodes):
        env.reset()
        s = get_state(env.state(), device=device)
        is_terminated = False
        ep_return = 0.0
        steps = 0
        while not is_terminated and steps < max_steps_per_episode:
            action = select_greedy_action(policy_net, s, cascade_iterations_1)
            reward, terminated = env.act(action)
            s = get_state(env.state(), device=device)
            ep_return += float(reward)
            is_terminated = bool(terminated)
            steps += 1
        returns.append(ep_return)

    arr = np.asarray(returns, dtype=np.float64)
    return {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
        "n_episodes": float(n_episodes),
    }
