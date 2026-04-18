"""Rollout-based evaluation for SARL agents.

Replaces the paper's ``evaluation()`` function (``maps.py:890-1068``) with
a cleaner two-function API:

* :func:`run_greedy_episode` — single-episode rollout under a **deterministic
  greedy policy** (no ε, no RNG consumption). Returns the episodic return,
  step count, and (optionally) wager statistics. This is what we report as
  "validation return" in figures.
* :func:`aggregate_validation` — run N greedy episodes, return mean/std.

Why the split from paper's ``evaluation()``
-------------------------------------------
The paper's ``evaluation()`` conflates **three concerns**:

1. Rollout an episode under the current ε-greedy policy (still explores!).
2. Simultaneously call ``train()`` with ``train_or_test=False`` to accumulate
   a validation *loss* per frame.
3. Share the running replay buffer with the training loop (mutating it).

Mixing #1 and #2 means "validation return" is coupled to whatever ε is at
frame ``t``, which drifts. It also means `evaluate()` touches `policy_net` in
eval mode but uses ε-greedy action selection — an unprincipled mix.

Our version: **pure rollout, greedy, no buffer mutation, no loss side-effect**.
Validation loss (if needed) lives in a separate hook.

Wager logging
-------------
When ``second_order_net`` is provided, we record the 2-unit wager logits at
each step and return their mean + bet-vs-no-bet ratio across the episode —
useful metrics for the MAPS paper's confidence analysis.

References
----------
- Mnih et al. (2015). — uses deterministic evaluation with fixed ε=0.05
  for consistency; we go further and set ε=0 (purely greedy) to remove any
  RNG drift from validation numbers.
- Vargas et al. (2025), MAPS TMLR submission §3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from maps.experiments.sarl.data import get_state
from maps.experiments.sarl.rollout import greedy_action


class MinAtarLike(Protocol):
    """Minimal interface we need from a MinAtar environment.

    Typed as a Protocol so tests can supply a fake env without importing
    MinAtar. The real ``minatar.Environment`` satisfies this interface.
    """

    def reset(self) -> Any: ...
    def state(self) -> np.ndarray: ...
    def act(self, action: torch.Tensor) -> tuple[float, bool]: ...
    def num_actions(self) -> int: ...


@dataclass
class EpisodeMetrics:
    """Per-episode evaluation rollout summary."""

    episodic_return: float
    n_steps: int
    # Populated when second_order_net is provided:
    mean_wager_bet: float | None = None  # mean softmax-prob(bet)
    mean_wager_nobet: float | None = None  # mean softmax-prob(no-bet)
    bet_ratio: float | None = None  # fraction of steps where argmax=bet
    wager_logits_trace: list[np.ndarray] = field(default_factory=list)


@dataclass
class ValidationSummary:
    """Aggregate over multiple validation episodes."""

    mean_return: float
    std_return: float
    mean_steps: float
    n_episodes: int
    # Wager stats averaged over episodes when available:
    mean_bet_ratio: float | None = None


def run_greedy_episode(
    env: MinAtarLike,
    policy_net: torch.nn.Module,
    cascade_iterations_1: int,
    *,
    second_order_net: torch.nn.Module | None = None,
    cascade_iterations_2: int | None = None,
    max_steps: int = 100_000,
    device: torch.device | str = "cpu",
    collect_wager_trace: bool = False,
) -> EpisodeMetrics:
    """Roll out one episode under greedy action selection.

    Parameters
    ----------
    env : MinAtar-like
        Environment. Reset internally before the rollout.
    policy_net : nn.Module
        First-order Q-network. Set to ``eval()`` for the duration of the call
        (restored to ``train()`` on exit if it was in training mode).
    cascade_iterations_1 : int
        Cascade depth for the first-order forward pass.
    second_order_net : nn.Module, optional
        If provided, wager logits are computed at each step and summarized.
        Requires ``cascade_iterations_2``.
    cascade_iterations_2 : int, optional
        Cascade depth for the second-order forward pass. Required iff
        ``second_order_net`` is provided.
    max_steps : int
        Safety cap on episode length. MinAtar games normally terminate on
        their own but we guard against infinite loops in test doubles.
    device : torch.device or str
        Used for tensor device of the initial state.
    collect_wager_trace : bool
        If True, store raw wager logits per step in the returned metrics
        (costly — only for offline analysis).

    Returns
    -------
    EpisodeMetrics
        See dataclass.
    """
    # Preserve and restore training mode (rollouts should be in eval).
    was_training_policy = policy_net.training
    policy_net.eval()
    was_training_second = None
    if second_order_net is not None:
        assert cascade_iterations_2 is not None, (
            "cascade_iterations_2 is required when second_order_net is provided"
        )
        was_training_second = second_order_net.training
        second_order_net.eval()

    try:
        env.reset()
        state = get_state(env.state(), device=device)
        g_return = 0.0
        n_steps = 0
        bet_probs: list[float] = []
        nobet_probs: list[float] = []
        bet_count = 0
        wager_trace: list[np.ndarray] = []

        done = False
        while not done and n_steps < max_steps:
            action = greedy_action(state, policy_net, cascade_iterations_1)

            if second_order_net is not None:
                # Recompute the forward through policy_net to get comparison,
                # then through second_order_net for wager. Small duplicate
                # work — acceptable outside the training loop.
                cascade_rate_1 = 1.0 / cascade_iterations_1
                cascade_rate_2 = 1.0 / cascade_iterations_2  # type: ignore[operator]
                main_task_out: Any = None
                comparison_1: torch.Tensor | None = None
                with torch.no_grad():
                    for _ in range(cascade_iterations_1):
                        _, _, comparison_1, main_task_out = policy_net(
                            state, main_task_out, cascade_rate_1
                        )
                    comparison_out: Any = None
                    wager_logits: torch.Tensor | None = None
                    for _ in range(cascade_iterations_2):  # type: ignore[arg-type]
                        wager_logits, comparison_out = second_order_net(
                            comparison_1, comparison_out, cascade_rate_2
                        )
                    assert wager_logits is not None
                    probs = torch.softmax(wager_logits, dim=-1).squeeze(0).cpu().numpy()
                bet_probs.append(float(probs[0]))
                nobet_probs.append(float(probs[1]))
                if probs[0] >= probs[1]:
                    bet_count += 1
                if collect_wager_trace:
                    wager_trace.append(wager_logits.squeeze(0).cpu().numpy())

            reward, done = env.act(action)
            g_return += float(reward)
            state = get_state(env.state(), device=device)
            n_steps += 1

        metrics = EpisodeMetrics(
            episodic_return=g_return,
            n_steps=n_steps,
            wager_logits_trace=wager_trace,
        )
        if second_order_net is not None and n_steps > 0:
            metrics.mean_wager_bet = float(np.mean(bet_probs))
            metrics.mean_wager_nobet = float(np.mean(nobet_probs))
            metrics.bet_ratio = bet_count / n_steps
        return metrics

    finally:
        if was_training_policy:
            policy_net.train()
        if second_order_net is not None and was_training_second:
            second_order_net.train()


def aggregate_validation(
    env: MinAtarLike,
    policy_net: torch.nn.Module,
    cascade_iterations_1: int,
    n_episodes: int,
    *,
    second_order_net: torch.nn.Module | None = None,
    cascade_iterations_2: int | None = None,
    max_steps: int = 100_000,
    device: torch.device | str = "cpu",
) -> ValidationSummary:
    """Run ``n_episodes`` greedy rollouts and summarize.

    Returns the mean/std of episodic returns, mean step count, and (if
    available) mean bet ratio.
    """
    returns: list[float] = []
    steps: list[int] = []
    bet_ratios: list[float] = []

    for _ in range(n_episodes):
        m = run_greedy_episode(
            env,
            policy_net,
            cascade_iterations_1,
            second_order_net=second_order_net,
            cascade_iterations_2=cascade_iterations_2,
            max_steps=max_steps,
            device=device,
        )
        returns.append(m.episodic_return)
        steps.append(m.n_steps)
        if m.bet_ratio is not None:
            bet_ratios.append(m.bet_ratio)

    summary = ValidationSummary(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_steps=float(np.mean(steps)),
        n_episodes=n_episodes,
    )
    if bet_ratios:
        summary.mean_bet_ratio = float(np.mean(bet_ratios))
    return summary
