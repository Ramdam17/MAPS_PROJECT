"""Action selection for SARL — ε-greedy over cascade-integrated Q values.

Ports the action-selection half of ``external/MinAtar/examples/maps.py:462-495``
(``world_dynamics``) as a pure function. The env stepping half (``env.act``,
``env.state``) stays at the caller level so this function is testable without
MinAtar.

Why split the paper's ``world_dynamics``
----------------------------------------
In the paper, ``world_dynamics(t, replay_start_size, num_actions, s, env, …)``
does two jobs in one call: (1) pick an action given the current state and
policy, (2) step the environment and return ``(s_prime, action, reward, done)``.
That coupling makes action selection hard to test — you need a full MinAtar
env on the test bench.

Our split:

* :func:`epsilon_greedy_action` — pure, deterministic-given-seeds, returns
  just the action tensor and a debug dict. Testable with a fake env.
* Env stepping — stays in ``training_loop.run_training`` where it belongs.

ε-schedule (paper §3.2)
-----------------------
During the warmup window ``t < replay_start_size`` a **uniform random policy**
is used (pure exploration to fill the replay buffer). Afterwards, ε is
annealed linearly from ``EPSILON`` (typically 1.0) to ``END_EPSILON`` (0.1)
over ``FIRST_N_FRAMES`` frames and then pinned at ``END_EPSILON``::

    ε(t) = END_EPSILON                                        if warmup
           EPSILON + (END_EPSILON - EPSILON) · progress       during anneal
           END_EPSILON                                        after anneal

where ``progress = min(1, (t - replay_start_size) / FIRST_N_FRAMES)``.

References
----------
- Mnih et al. (2015). Human-level control through deep reinforcement learning.
  Nature 518:529-533. — canonical DQN ε-schedule.
- Vargas et al. (2025), MAPS TMLR submission §3.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy
import torch

# Paper constants (source: external/MinAtar/examples/maps.py:95-105).
# Re-exposed here so callers don't have to cross-import the paper's globals.
EPSILON_START = 1.0
END_EPSILON = 0.1
FIRST_N_FRAMES = 100_000


@dataclass
class ActionSelection:
    """Debug info returned alongside the chosen action.

    Populated so callers / loggers can introspect exploration vs exploitation
    behavior without re-running the forward pass.
    """

    action: torch.Tensor  # shape (1, 1), int64
    epsilon: float  # ε value used (NaN during warmup)
    was_exploration: bool  # True if the action was chosen uniformly
    q_values: torch.Tensor | None = None  # populated only for greedy branch


def anneal_epsilon(t: int, replay_start_size: int) -> float:
    """Linear ε anneal with warmup + clamp (paper §3.2).

    Parameters
    ----------
    t : int
        Current frame counter (paper's global ``t``).
    replay_start_size : int
        Warmup size — before this many frames we use a uniform random policy.

    Returns
    -------
    eps : float
        ε value in [END_EPSILON, EPSILON_START]. Returns ``EPSILON_START``
        during warmup (caller may ignore this and use pure random, but we
        return a meaningful value for logging consistency).
    """
    if t < replay_start_size:
        return EPSILON_START
    progress_frames = t - replay_start_size
    if progress_frames >= FIRST_N_FRAMES:
        return END_EPSILON
    # Paper line 474-475: EPSILON + (END_EPSILON - EPSILON) * (t - replay_start_size) / FIRST_N_FRAMES
    # Rewritten for readability; identical math.
    slope = (END_EPSILON - EPSILON_START) / FIRST_N_FRAMES
    return EPSILON_START + slope * progress_frames


def epsilon_greedy_action(
    state: torch.Tensor,
    policy_net: torch.nn.Module,
    t: int,
    replay_start_size: int,
    num_actions: int,
    cascade_iterations_1: int,
    *,
    python_rng: random.Random | None = None,
    numpy_rng: numpy.random.Generator | None = None,
    device: torch.device | str = "cpu",
) -> ActionSelection:
    """Pick an action from the current policy with paper's ε-schedule.

    Parameters
    ----------
    state : Tensor of shape (1, C, 10, 10)
        Current environment state (already converted via
        :func:`maps.experiments.sarl.data.get_state`).
    policy_net : nn.Module
        First-order Q-network. Expected to return a 4-tuple
        ``(q_values, hidden, comparison, hidden_copy)`` per
        :class:`SarlQNetwork`.
    t : int
        Current frame count.
    replay_start_size : int
        Warmup size — below this, action is uniform random.
    num_actions : int
        Action-space cardinality.
    cascade_iterations_1 : int
        Paper's ``cascade_iterations_1`` (1 when cascade off, 50 when on).
        The Q-network forward is called this many times with its own
        carry-state so the paper's cascade integration is applied.
    python_rng, numpy_rng : optional
        Inject custom RNGs for deterministic tests. Default ``None`` means
        use the module-level ``random`` and ``numpy.random`` globals (which
        is what the paper does; :func:`maps.utils.seeding.set_all_seeds`
        takes care of seeding them).
    device : torch.device or str
        Device to place the zero-length action tensor on during warmup.

    Returns
    -------
    ActionSelection
        See dataclass doc.
    """
    # Paper line 468: warmup uses uniform random.
    if t < replay_start_size:
        idx = (python_rng or random).randrange(num_actions)
        action = torch.tensor([[idx]], device=device, dtype=torch.int64)
        return ActionSelection(
            action=action,
            epsilon=float("nan"),  # undefined during warmup
            was_exploration=True,
        )

    epsilon = anneal_epsilon(t, replay_start_size)

    # Paper line 477: `numpy.random.binomial(1, epsilon) == 1` → explore.
    # NPY002 silenced: we deliberately mirror the paper's use of the legacy
    # ``numpy.random.binomial`` global so deterministic seeding set by
    # ``maps.utils.seeding.set_all_seeds`` produces the paper's draw sequence.
    draw = numpy_rng.binomial(1, epsilon) if numpy_rng else numpy.random.binomial(1, epsilon)  # noqa: NPY002
    if draw == 1:
        idx = (python_rng or random).randrange(num_actions)
        action = torch.tensor([[idx]], device=device, dtype=torch.int64)
        return ActionSelection(
            action=action,
            epsilon=epsilon,
            was_exploration=True,
        )

    # Greedy branch: forward pass through cascade, take argmax.
    cascade_rate_1 = 1.0 / cascade_iterations_1
    main_task_out: Any = None
    q_values: torch.Tensor | None = None
    with torch.no_grad():
        for _ in range(cascade_iterations_1):
            q_values, _, _, main_task_out = policy_net(state, main_task_out, cascade_rate_1)
    assert q_values is not None  # guaranteed by loop having cascade_iterations_1 >= 1

    # `.view(1, 1)` preserves the paper's tensor shape contract (see line 487).
    action = q_values.max(1)[1].view(1, 1)
    return ActionSelection(
        action=action,
        epsilon=epsilon,
        was_exploration=False,
        q_values=q_values,
    )


def greedy_action(
    state: torch.Tensor,
    policy_net: torch.nn.Module,
    cascade_iterations_1: int,
) -> torch.Tensor:
    """Deterministic greedy action — no exploration. Used by evaluation rollouts.

    Separate from :func:`epsilon_greedy_action` so evaluation code doesn't
    have to pass sentinel values that disable exploration; the intent is
    clearer at the call site.
    """
    cascade_rate_1 = 1.0 / cascade_iterations_1
    main_task_out: Any = None
    q_values: torch.Tensor | None = None
    with torch.no_grad():
        for _ in range(cascade_iterations_1):
            q_values, _, _, main_task_out = policy_net(state, main_task_out, cascade_rate_1)
    assert q_values is not None
    return q_values.max(1)[1].view(1, 1)
