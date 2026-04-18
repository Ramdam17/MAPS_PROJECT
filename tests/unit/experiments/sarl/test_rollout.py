"""Unit tests for ``maps.experiments.sarl.rollout``.

Covers:
* ``anneal_epsilon`` boundary math (warmup / anneal / plateau).
* ``epsilon_greedy_action`` branch selection (warmup → uniform,
  post-warmup with Bernoulli=1 → uniform, Bernoulli=0 → argmax).
* ``greedy_action`` returns argmax of the cascaded Q-values.

The policy network is faked so tests run in milliseconds and don't depend on
weights / cascade convergence — we just verify *which* branch fires.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pytest
import torch

from maps.experiments.sarl.rollout import (
    END_EPSILON,
    EPSILON_START,
    FIRST_N_FRAMES,
    anneal_epsilon,
    epsilon_greedy_action,
    greedy_action,
)

NUM_ACTIONS = 6
REPLAY_START = 5_000


class _FakePolicyNet(torch.nn.Module):
    """Deterministic stand-in for ``SarlQNetwork``.

    Returns a Q-vector whose argmax is fixed at ``argmax_idx`` — makes
    greedy-branch assertions trivial to verify.

    Forward contract matches the real net: returns
    ``(q_values, hidden, comparison, hidden_copy)``.
    """

    def __init__(self, num_actions: int = NUM_ACTIONS, argmax_idx: int = 3):
        super().__init__()
        self.num_actions = num_actions
        self.argmax_idx = argmax_idx
        self.call_count = 0

    def forward(
        self, x: torch.Tensor, prev_h2: Any, cascade_rate: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.call_count += 1
        q = torch.full((x.size(0), self.num_actions), -1.0)
        q[:, self.argmax_idx] = 1.0
        hidden = torch.zeros(x.size(0), 128)
        comparison = torch.zeros(x.size(0), 1024)
        return q, hidden, comparison, hidden


# ── anneal_epsilon ──────────────────────────────────────────────────────────


def test_anneal_epsilon_warmup_returns_start():
    assert anneal_epsilon(t=0, replay_start_size=REPLAY_START) == EPSILON_START
    assert anneal_epsilon(t=REPLAY_START - 1, replay_start_size=REPLAY_START) == EPSILON_START


def test_anneal_epsilon_at_replay_start_equals_start():
    """Frame == replay_start_size ⇒ progress=0 ⇒ ε = EPSILON_START."""
    assert anneal_epsilon(t=REPLAY_START, replay_start_size=REPLAY_START) == EPSILON_START


def test_anneal_epsilon_at_midpoint():
    t = REPLAY_START + FIRST_N_FRAMES // 2
    eps = anneal_epsilon(t, REPLAY_START)
    expected = EPSILON_START + 0.5 * (END_EPSILON - EPSILON_START)
    assert math.isclose(eps, expected, abs_tol=1e-9)


def test_anneal_epsilon_pinned_at_end_after_anneal_window():
    t = REPLAY_START + FIRST_N_FRAMES
    assert anneal_epsilon(t, REPLAY_START) == END_EPSILON
    assert anneal_epsilon(t + 10_000_000, REPLAY_START) == END_EPSILON


# ── epsilon_greedy_action: warmup branch ────────────────────────────────────


def test_warmup_action_is_uniform_random():
    rng = random.Random(42)
    net = _FakePolicyNet()
    state = torch.zeros(1, 4, 10, 10)
    sel = epsilon_greedy_action(
        state,
        net,
        t=0,
        replay_start_size=REPLAY_START,
        num_actions=NUM_ACTIONS,
        cascade_iterations_1=1,
        python_rng=rng,
    )
    assert sel.was_exploration is True
    assert math.isnan(sel.epsilon)
    assert net.call_count == 0, "warmup must skip the forward pass"
    assert sel.action.shape == (1, 1)
    assert 0 <= int(sel.action.item()) < NUM_ACTIONS


# ── epsilon_greedy_action: exploration branch ───────────────────────────────


class _NumpyRngExplore:
    """Forces numpy.random.binomial to return 1 (= explore)."""

    def binomial(self, n: int, p: float) -> int:
        return 1


class _NumpyRngExploit:
    """Forces numpy.random.binomial to return 0 (= exploit)."""

    def binomial(self, n: int, p: float) -> int:
        return 0


def test_post_warmup_explore_branch_is_uniform():
    rng = random.Random(7)
    net = _FakePolicyNet(argmax_idx=2)
    state = torch.zeros(1, 4, 10, 10)
    t = REPLAY_START + 100  # annealing, epsilon ~ EPSILON_START

    sel = epsilon_greedy_action(
        state,
        net,
        t=t,
        replay_start_size=REPLAY_START,
        num_actions=NUM_ACTIONS,
        cascade_iterations_1=1,
        python_rng=rng,
        numpy_rng=_NumpyRngExplore(),
    )
    assert sel.was_exploration is True
    assert not math.isnan(sel.epsilon)
    assert net.call_count == 0, "explore branch must skip the forward pass"
    assert sel.q_values is None
    assert 0 <= int(sel.action.item()) < NUM_ACTIONS


def test_post_warmup_exploit_branch_returns_argmax():
    net = _FakePolicyNet(argmax_idx=4)
    state = torch.zeros(1, 4, 10, 10)
    t = REPLAY_START + FIRST_N_FRAMES  # pinned at END_EPSILON

    sel = epsilon_greedy_action(
        state,
        net,
        t=t,
        replay_start_size=REPLAY_START,
        num_actions=NUM_ACTIONS,
        cascade_iterations_1=1,
        numpy_rng=_NumpyRngExploit(),
    )
    assert sel.was_exploration is False
    assert sel.epsilon == END_EPSILON
    assert net.call_count == 1, "exploit must run exactly cascade_iterations_1 passes"
    assert int(sel.action.item()) == 4
    assert sel.q_values is not None
    assert sel.q_values.shape == (1, NUM_ACTIONS)


@pytest.mark.parametrize("cascade_iters", [1, 5, 50])
def test_exploit_cascade_iterations_call_count(cascade_iters: int):
    net = _FakePolicyNet(argmax_idx=0)
    state = torch.zeros(1, 4, 10, 10)
    sel = epsilon_greedy_action(
        state,
        net,
        t=REPLAY_START + FIRST_N_FRAMES,
        replay_start_size=REPLAY_START,
        num_actions=NUM_ACTIONS,
        cascade_iterations_1=cascade_iters,
        numpy_rng=_NumpyRngExploit(),
    )
    assert sel.was_exploration is False
    assert net.call_count == cascade_iters


# ── greedy_action ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("argmax_idx", [0, 3, 5])
def test_greedy_action_returns_argmax(argmax_idx: int):
    net = _FakePolicyNet(argmax_idx=argmax_idx)
    state = torch.zeros(1, 4, 10, 10)
    action = greedy_action(state, net, cascade_iterations_1=1)
    assert action.shape == (1, 1)
    assert int(action.item()) == argmax_idx


def test_greedy_action_does_not_accumulate_grad():
    """Evaluation must not leak autograd state onto the live network."""
    net = _FakePolicyNet()
    state = torch.zeros(1, 4, 10, 10, requires_grad=True)
    action = greedy_action(state, net, cascade_iterations_1=3)
    # greedy_action runs under no_grad → returned action has no grad_fn
    assert action.grad_fn is None


def test_rng_injection_is_deterministic():
    """Injecting the same ``random.Random`` twice gives identical warmup actions."""
    state = torch.zeros(1, 4, 10, 10)
    net = _FakePolicyNet()
    rng_a = random.Random(1234)
    rng_b = random.Random(1234)
    seqs = []
    for rng in (rng_a, rng_b):
        seqs.append(
            [
                int(
                    epsilon_greedy_action(
                        state,
                        net,
                        t=0,
                        replay_start_size=REPLAY_START,
                        num_actions=NUM_ACTIONS,
                        cascade_iterations_1=1,
                        python_rng=rng,
                    ).action.item()
                )
                for _ in range(20)
            ]
        )
    assert seqs[0] == seqs[1]


# ── Paper-constant smoke (guards against accidental edits) ──────────────────


def test_paper_constants_match_spec():
    assert EPSILON_START == 1.0
    assert END_EPSILON == 0.1
    assert FIRST_N_FRAMES == 100_000


def test_fake_policy_q_values_are_not_nan():
    """Sanity: the fake net returns finite Q-values (used everywhere above)."""
    net = _FakePolicyNet()
    q, *_ = net(torch.zeros(1, 4, 10, 10), None, 1.0)
    assert not torch.isnan(q).any()
    assert not torch.isinf(q).any()
    assert torch.allclose(q.argmax(dim=1), torch.tensor([net.argmax_idx]))
    assert isinstance(np.asarray(q.detach().numpy())[0, 0], np.floating)
