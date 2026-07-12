"""Tier-4-light: SARL DQN training loop + greedy evaluation (Sprint 14.G).

The full loop uses an **unseeded** MinAtar env (D14.3, faithful) so returns are
not deterministic run-to-run; this is an orchestration smoke — the loop runs
end-to-end, the buffer warms, updates + target syncs happen, metrics are
recorded — plus greedy evaluation. (Uses a real MinAtar env: first import ~60s.)
"""

from __future__ import annotations

import pytest

from maps.domains.sarl.evaluate import evaluate_greedy
from maps.domains.sarl.trainer import SETTINGS_REGISTRY
from maps.domains.sarl.training_loop import run_training
from maps.utils.config import load_config
from maps.utils.seeding import set_all_seeds


def _tiny_cfg():
    return load_config(
        "domains/sarl/training",
        overrides=[
            "training.replay_start_size=20",
            "training.batch_size=8",
            "training.target_update_freq=5",
            "training.replay_buffer_size=500",
        ],
    )


@pytest.fixture(scope="module")
def env():
    from minatar import Environment

    return Environment("breakout")


def test_baseline_loop_runs_and_updates(env):
    set_all_seeds(42)
    cfg = _tiny_cfg()
    _policy_net, second_order_net, metrics = run_training(
        SETTINGS_REGISTRY["setting-1-baseline"], cfg, env, num_frames=150
    )
    assert metrics.total_frames >= 150
    assert metrics.total_updates > 0  # buffer warmed past replay_start_size
    assert len(metrics.episode_returns) >= 1
    assert second_order_net is None  # baseline: no 2nd-order


def test_meta_loop_builds_second_order(env):
    set_all_seeds(1)
    cfg = _tiny_cfg()
    _policy_net, second_order_net, metrics = run_training(
        SETTINGS_REGISTRY["setting-3-second-order-only"], cfg, env, num_frames=120
    )
    assert second_order_net is not None
    assert metrics.total_updates > 0
    # meta path records a second-order loss
    assert any(x != 0.0 for x in metrics.episode_losses_second)


def test_evaluate_greedy_returns_metrics(env):
    set_all_seeds(2)
    cfg = _tiny_cfg()
    policy_net, _, _ = run_training(
        SETTINGS_REGISTRY["setting-1-baseline"], cfg, env, num_frames=60
    )
    result = evaluate_greedy(policy_net, env, n_episodes=2, cascade_iterations_1=1)
    assert set(result) >= {"mean_return", "std_return", "n_episodes"}
    assert result["n_episodes"] == 2.0
