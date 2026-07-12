"""Unit tests for SARL rollout — ε schedule, greedy selection, world_dynamics.

The ε schedule and greedy action selection are pure/net-only (fast). The
``world_dynamics`` smoke uses a real MinAtar env (first ``import minatar`` is
slow on this filesystem, ~60s).
"""

from __future__ import annotations

import pytest
import torch

from maps.domains.sarl.model_v1 import SarlQNetworkV1
from maps.domains.sarl.rollout import epsilon_at, select_greedy_action, world_dynamics
from maps.utils.seeding import set_all_seeds


def test_epsilon_schedule_endpoints_and_midpoint():
    rss = 5000
    # At warm-up end, ε = start (1.0).
    assert epsilon_at(rss, rss) == pytest.approx(1.0)
    # After first_n_frames, ε floored to end (0.1).
    assert epsilon_at(rss + 100_000, rss) == pytest.approx(0.1)
    assert epsilon_at(rss + 200_000, rss) == pytest.approx(0.1)
    # Linear midpoint.
    assert epsilon_at(rss + 50_000, rss) == pytest.approx(0.55)


def test_select_greedy_action_shape_and_determinism():
    set_all_seeds(42)
    net = SarlQNetworkV1(4, 6).eval()
    s = torch.rand(1, 4, 10, 10)
    a1 = select_greedy_action(net, s, cascade_iterations_1=1)
    a2 = select_greedy_action(net, s, cascade_iterations_1=1)
    assert a1.shape == (1, 1)
    assert 0 <= int(a1) < 6
    assert torch.equal(a1, a2)


def test_world_dynamics_warmup_is_random_policy():
    """t < replay_start_size → uniform random action, valid transition tuple."""
    from minatar import Environment

    set_all_seeds(42)
    env = Environment("breakout")
    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()
    net = SarlQNetworkV1(in_channels, num_actions).eval()

    from maps.domains.sarl.data import get_state

    s = get_state(env.state())
    s_prime, action, reward, is_terminal = world_dynamics(
        t=0,
        replay_start_size=5000,
        num_actions=num_actions,
        s=s,
        env=env,
        policy_net=net,
        cascade_iterations_1=1,
    )
    assert s_prime.shape == (1, in_channels, 10, 10)
    assert action.shape == (1, 1) and 0 <= int(action) < num_actions
    assert reward.shape == (1, 1)
    assert is_terminal.shape == (1, 1)


def test_world_dynamics_greedy_step_runs():
    """t ≥ replay_start_size with ε forced to 0 → greedy step runs end-to-end."""
    from minatar import Environment

    from maps.domains.sarl.data import get_state

    set_all_seeds(1)
    env = Environment("breakout")
    net = SarlQNetworkV1(env.state_shape()[2], env.num_actions()).eval()
    s = get_state(env.state())
    # end_epsilon=0 + past anneal → always greedy.
    out = world_dynamics(
        t=200_000,
        replay_start_size=5000,
        num_actions=env.num_actions(),
        s=s,
        env=env,
        policy_net=net,
        cascade_iterations_1=1,
        end_epsilon=0.0,
        start_epsilon=0.0,
    )
    assert out[0].shape == (1, env.state_shape()[2], 10, 10)
