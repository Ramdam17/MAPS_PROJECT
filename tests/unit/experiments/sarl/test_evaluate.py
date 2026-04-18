"""Unit tests for ``maps.experiments.sarl.evaluate``.

Exercises ``run_greedy_episode`` and ``aggregate_validation`` without a real
MinAtar environment. Verifies:

* Episodic return is the sum of step rewards.
* Step count matches the trace length.
* ``policy_net`` is left in its original training mode after the rollout
  (eval mode is restored).
* Wager statistics are correct when ``second_order_net`` is provided.
* ``aggregate_validation`` returns mean / std over N episodes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from maps.experiments.sarl.evaluate import (
    EpisodeMetrics,
    ValidationSummary,
    aggregate_validation,
    run_greedy_episode,
)

IN_CHANNELS = 4
NUM_ACTIONS = 6


# ── Fakes ───────────────────────────────────────────────────────────────────


class _FakeEnv:
    """Minimal MinAtar-like env driven by a scripted (reward, done) trace.

    Produces a zero state at every step — sufficient because the greedy action
    only depends on what the fake policy returns, not the state contents.
    """

    def __init__(self, trace: list[tuple[float, bool]]):
        self.trace = trace
        self.step = 0
        self._state = np.zeros((10, 10, IN_CHANNELS), dtype=np.float32)

    def reset(self) -> None:
        self.step = 0

    def state(self) -> np.ndarray:
        return self._state

    def act(self, action: torch.Tensor) -> tuple[float, bool]:
        reward, done = self.trace[self.step]
        self.step += 1
        return reward, done

    def num_actions(self) -> int:
        return NUM_ACTIONS


class _FakePolicyNet(torch.nn.Module):
    """Fixed-argmax policy net matching SarlQNetwork forward contract."""

    def __init__(self, argmax_idx: int = 2):
        super().__init__()
        # One real parameter so ``.train()`` / ``.eval()`` mode is observable.
        self.dummy = torch.nn.Linear(1, 1)
        self.argmax_idx = argmax_idx
        self.training_modes_seen: list[bool] = []

    def forward(
        self, x: torch.Tensor, prev_h2: Any, cascade_rate: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.training_modes_seen.append(self.training)
        q = torch.full((x.size(0), NUM_ACTIONS), -1.0)
        q[:, self.argmax_idx] = 1.0
        hidden = torch.zeros(x.size(0), 128)
        comparison = torch.zeros(x.size(0), 1024)
        return q, hidden, comparison, hidden


class _FakeSecondOrderNet(torch.nn.Module):
    """Wager head that returns fixed logits so bet/no-bet stats are predictable."""

    def __init__(self, bet_logit: float = 2.0, nobet_logit: float = 0.0):
        super().__init__()
        self.dummy = torch.nn.Linear(1, 1)
        self.bet_logit = bet_logit
        self.nobet_logit = nobet_logit

    def forward(
        self,
        comparison_matrix: torch.Tensor,
        prev_comparison: Any,
        cascade_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.tensor(
            [[self.bet_logit, self.nobet_logit]],
            dtype=torch.float32,
        ).expand(comparison_matrix.size(0), -1)
        return logits, comparison_matrix


# ── run_greedy_episode ──────────────────────────────────────────────────────


def test_episodic_return_sums_rewards():
    trace = [(1.0, False), (0.5, False), (2.0, False), (0.0, True)]
    env = _FakeEnv(trace)
    net = _FakePolicyNet()
    m = run_greedy_episode(env, net, cascade_iterations_1=1)
    assert isinstance(m, EpisodeMetrics)
    assert m.episodic_return == pytest.approx(3.5)
    assert m.n_steps == 4


def test_terminates_on_done_flag():
    trace = [(0.0, False), (0.0, True), (999.0, False)]  # 3rd never reached
    env = _FakeEnv(trace)
    net = _FakePolicyNet()
    m = run_greedy_episode(env, net, cascade_iterations_1=1)
    assert m.n_steps == 2
    assert m.episodic_return == 0.0


def test_respects_max_steps_safety_cap():
    trace = [(1.0, False)] * 100  # env never terminates on its own
    env = _FakeEnv(trace)
    net = _FakePolicyNet()
    m = run_greedy_episode(env, net, cascade_iterations_1=1, max_steps=10)
    assert m.n_steps == 10
    assert m.episodic_return == pytest.approx(10.0)


def test_training_mode_restored_after_rollout():
    trace = [(0.0, True)]
    env = _FakeEnv(trace)
    net = _FakePolicyNet()
    net.train()  # explicitly put into training mode
    _ = run_greedy_episode(env, net, cascade_iterations_1=1)
    assert net.training is True, "training mode should be restored on exit"
    # During rollout, the forward pass must have been in eval mode.
    assert all(mode is False for mode in net.training_modes_seen)


def test_eval_mode_preserved_when_net_was_already_in_eval():
    trace = [(0.0, True)]
    env = _FakeEnv(trace)
    net = _FakePolicyNet()
    net.eval()
    _ = run_greedy_episode(env, net, cascade_iterations_1=1)
    assert net.training is False


# ── Wager statistics ────────────────────────────────────────────────────────


def test_wager_stats_populated_when_second_net_provided():
    """With bet_logit > nobet_logit, bet_ratio should be 1.0."""
    trace = [(0.0, False), (1.0, False), (0.0, True)]
    env = _FakeEnv(trace)
    net = _FakePolicyNet()
    second = _FakeSecondOrderNet(bet_logit=3.0, nobet_logit=-1.0)
    m = run_greedy_episode(
        env,
        net,
        cascade_iterations_1=1,
        second_order_net=second,
        cascade_iterations_2=1,
    )
    assert m.bet_ratio == pytest.approx(1.0)
    # softmax([3, -1]) = [~0.982, ~0.018]
    assert m.mean_wager_bet > 0.9
    assert m.mean_wager_nobet < 0.1


def test_wager_stats_none_when_no_second_net():
    trace = [(0.0, True)]
    env = _FakeEnv(trace)
    net = _FakePolicyNet()
    m = run_greedy_episode(env, net, cascade_iterations_1=1)
    assert m.bet_ratio is None
    assert m.mean_wager_bet is None
    assert m.mean_wager_nobet is None


def test_bet_ratio_zero_when_nobet_dominates():
    trace = [(0.0, False), (0.0, True)]
    env = _FakeEnv(trace)
    net = _FakePolicyNet()
    second = _FakeSecondOrderNet(bet_logit=-5.0, nobet_logit=5.0)
    m = run_greedy_episode(
        env,
        net,
        cascade_iterations_1=1,
        second_order_net=second,
        cascade_iterations_2=1,
    )
    assert m.bet_ratio == pytest.approx(0.0)


def test_second_net_requires_cascade_iterations_2():
    trace = [(0.0, True)]
    env = _FakeEnv(trace)
    net = _FakePolicyNet()
    second = _FakeSecondOrderNet()
    with pytest.raises(AssertionError):
        run_greedy_episode(
            env,
            net,
            cascade_iterations_1=1,
            second_order_net=second,  # cascade_iterations_2 omitted
        )


def test_collect_wager_trace_fills_when_requested():
    trace = [(0.0, False), (0.0, False), (0.0, True)]
    env = _FakeEnv(trace)
    net = _FakePolicyNet()
    second = _FakeSecondOrderNet()
    m = run_greedy_episode(
        env,
        net,
        cascade_iterations_1=1,
        second_order_net=second,
        cascade_iterations_2=1,
        collect_wager_trace=True,
    )
    assert len(m.wager_logits_trace) == m.n_steps
    for row in m.wager_logits_trace:
        assert row.shape == (2,)


def test_collect_wager_trace_default_is_empty():
    trace = [(0.0, False), (0.0, True)]
    env = _FakeEnv(trace)
    net = _FakePolicyNet()
    second = _FakeSecondOrderNet()
    m = run_greedy_episode(
        env,
        net,
        cascade_iterations_1=1,
        second_order_net=second,
        cascade_iterations_2=1,
    )
    assert m.wager_logits_trace == []


# ── aggregate_validation ────────────────────────────────────────────────────


def test_aggregate_validation_summary_fields():
    """Each episode yields reward 1.0 × 3 steps → mean=3.0, std=0."""

    def make_env() -> _FakeEnv:
        return _FakeEnv([(1.0, False), (1.0, False), (1.0, True)])

    net = _FakePolicyNet()
    summary = aggregate_validation(make_env(), net, cascade_iterations_1=1, n_episodes=5)
    assert isinstance(summary, ValidationSummary)
    assert summary.n_episodes == 5
    assert summary.mean_return == pytest.approx(3.0)
    assert summary.std_return == pytest.approx(0.0)
    assert summary.mean_steps == pytest.approx(3.0)
    assert summary.mean_bet_ratio is None


def test_aggregate_validation_reports_bet_ratio():
    env = _FakeEnv([(0.0, True)])  # single-step episode
    net = _FakePolicyNet()
    second = _FakeSecondOrderNet(bet_logit=3.0, nobet_logit=-1.0)
    summary = aggregate_validation(
        env,
        net,
        cascade_iterations_1=1,
        n_episodes=2,
        second_order_net=second,
        cascade_iterations_2=1,
    )
    # bet_ratio per episode = 1.0, so mean = 1.0
    assert summary.mean_bet_ratio == pytest.approx(1.0)
