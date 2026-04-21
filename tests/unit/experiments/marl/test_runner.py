"""Unit tests for :class:`MeltingpotRunner` + helpers (E.9b).

Uses a minimal fake MeltingPot-like env (``_FakeEnv``) so we can exercise
``warmup → collect → insert → compute → train`` end-to-end without dmlab2d.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from maps.experiments.marl import MarlSetting, MeltingpotRunner, RunnerConfig
from maps.experiments.marl.runner import compute_wager_objective
from maps.utils import load_config


HIDDEN = 32
OBS_SHAPE = (11, 11, 3)
N_AGENTS = 2
EPISODE_LEN = 4
N_THREADS = 2
N_ACTIONS = 8


# ──────────────────────────────────────────────────────────────
# compute_wager_objective (paper eq.13-14)
# ──────────────────────────────────────────────────────────────


def test_compute_wager_objective_shape_and_one_hot():
    reward = np.array([1.0, -0.5, 0.2], dtype=np.float32)
    ema = np.zeros(3, dtype=np.float32)
    ema_new, wager = compute_wager_objective(reward, ema, alpha=0.45, condition="r_t_gt_ema")
    assert ema_new.shape == (3,)
    assert wager.shape == (3, 2)
    # Each row is a one-hot.
    assert np.all(wager.sum(axis=1) == 1.0)


def test_compute_wager_objective_r_t_gt_ema_semantics():
    reward = np.array([1.0, 0.0], dtype=np.float32)
    ema_prev = np.array([0.0, 0.5], dtype=np.float32)
    _ema_new, wager = compute_wager_objective(reward, ema_prev, alpha=0.45, condition="r_t_gt_ema")
    # agent 0 : r > ema (new = 0.45*1 + 0.55*0 = 0.45 ; 1 > 0.45) → high (1, 0)
    # agent 1 : r < ema (new = 0.275 ; 0 < 0.275) → low (0, 1)
    assert np.allclose(wager[0], [1.0, 0.0])
    assert np.allclose(wager[1], [0.0, 1.0])


def test_compute_wager_objective_ema_gt_zero_condition():
    reward = np.array([1.0, -1.0], dtype=np.float32)
    ema_prev = np.array([0.0, 0.0], dtype=np.float32)
    _ema_new, wager = compute_wager_objective(reward, ema_prev, alpha=0.5, condition="ema_gt_zero")
    # ema_new : [0.5, -0.5] → high, low.
    assert np.allclose(wager[0], [1.0, 0.0])
    assert np.allclose(wager[1], [0.0, 1.0])


def test_compute_wager_objective_rejects_unknown_condition():
    with pytest.raises(ValueError, match="Unknown wager"):
        compute_wager_objective(
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            alpha=0.45,
            condition="bogus",
        )


def test_compute_wager_objective_ema_update_equation():
    reward = np.array([10.0], dtype=np.float32)
    ema = np.array([2.0], dtype=np.float32)
    ema_new, _ = compute_wager_objective(reward, ema, alpha=0.45)
    expected = 0.45 * 10.0 + 0.55 * 2.0
    assert np.isclose(ema_new[0], expected)


# ──────────────────────────────────────────────────────────────
# _FakeEnv : MeltingPot-like multi-agent env
# ──────────────────────────────────────────────────────────────


class _FakeEnv:
    """Minimal multi-agent env matching the runner's contract."""

    def __init__(self, num_agents: int, n_threads: int, obs_shape, n_actions: int):
        self.num_agents = num_agents
        self.n_threads = n_threads
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self._player_keys = [f"player_{i}" for i in range(num_agents)]

    def _obs_dict(self):
        return {
            k: {
                "RGB": np.random.randint(0, 256, (self.n_threads, *self.obs_shape), dtype=np.uint8).astype(np.float32),
                "WORLD.RGB": np.random.randint(0, 256, (self.n_threads, *self.obs_shape), dtype=np.uint8).astype(np.float32),
            }
            for k in self._player_keys
        }

    def reset(self):
        return self._obs_dict(), {}

    def step(self, action_dict):
        obs = self._obs_dict()
        rewards = {k: np.random.randn(self.n_threads).astype(np.float32) for k in self._player_keys}
        dones = {k: np.zeros(self.n_threads, dtype=bool) for k in self._player_keys}
        info = {}
        return obs, rewards, dones, info

    def close(self):
        pass


# ──────────────────────────────────────────────────────────────
# Runner construction + warmup
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def cfg():
    return load_config(
        "training/marl",
        overrides=[
            f"model.hidden_size={HIDDEN}",
            "model.second_order_dropout=0.0",
            f"training.episode_length={EPISODE_LEN}",
            f"training.n_rollout_threads={N_THREADS}",
            "training.num_env_steps=100",
            "ppo.ppo_epoch=2",
            "ppo.num_mini_batch=1",
            "ppo.data_chunk_length=2",
        ],
    )


@pytest.fixture
def runner_cfg(cfg):
    setting = MarlSetting(
        id="baseline", label="baseline",
        meta=False, cascade_iterations1=1, cascade_iterations2=1,
    )
    return RunnerConfig(
        cfg=cfg,
        setting=setting,
        num_agents=N_AGENTS,
        obs_shape=OBS_SHAPE,
        share_obs_shape=OBS_SHAPE,
        action_space=spaces.Discrete(N_ACTIONS),
        device="cpu",
    )


@pytest.fixture
def meta_runner_cfg(cfg):
    setting = MarlSetting(
        id="maps", label="maps",
        meta=True, cascade_iterations1=50, cascade_iterations2=1,
    )
    return RunnerConfig(
        cfg=cfg,
        setting=setting,
        num_agents=N_AGENTS,
        obs_shape=OBS_SHAPE,
        share_obs_shape=OBS_SHAPE,
        action_space=spaces.Discrete(N_ACTIONS),
        device="cpu",
    )


def test_runner_constructs_one_policy_per_agent(runner_cfg):
    env = _FakeEnv(N_AGENTS, N_THREADS, OBS_SHAPE, N_ACTIONS)
    runner = MeltingpotRunner(runner_cfg, env)
    assert len(runner.policies) == N_AGENTS
    assert len(runner.trainers) == N_AGENTS
    assert len(runner.buffers) == N_AGENTS
    # EMA state shape matches (num_agents, n_rollout_threads).
    assert runner.ema_reward.shape == (N_AGENTS, N_THREADS)


def test_runner_warmup_fills_buffer_step_zero(runner_cfg):
    env = _FakeEnv(N_AGENTS, N_THREADS, OBS_SHAPE, N_ACTIONS)
    runner = MeltingpotRunner(runner_cfg, env)
    runner.warmup()
    for agent_id in range(N_AGENTS):
        buf = runner.buffers[agent_id]
        # step 0 must be filled (not all zeros, since env uses random obs).
        assert buf.obs[0].shape == (N_THREADS, *OBS_SHAPE)
        assert buf.share_obs[0].shape == (N_THREADS, *OBS_SHAPE)
        assert np.any(buf.obs[0] != 0)


# ──────────────────────────────────────────────────────────────
# Full rollout + train (baseline)
# ──────────────────────────────────────────────────────────────


def test_runner_run_one_episode_baseline(runner_cfg):
    env = _FakeEnv(N_AGENTS, N_THREADS, OBS_SHAPE, N_ACTIONS)
    runner = MeltingpotRunner(runner_cfg, env)
    infos = runner.run(num_episodes=1)
    assert len(infos) == 1
    ep = infos[0]
    assert ep["episode"] == 0
    assert len(ep["per_agent"]) == N_AGENTS
    for per_agent_info in ep["per_agent"]:
        assert "value_loss" in per_agent_info
        assert "policy_loss" in per_agent_info
        # Baseline : wager losses are exactly zero (no meta path).
        assert per_agent_info["wager_loss_actor"] == 0.0
        assert per_agent_info["wager_loss_critic"] == 0.0
        assert np.isfinite(per_agent_info["value_loss"])


def test_runner_run_one_episode_meta_produces_wager_loss(meta_runner_cfg):
    """Meta setting : wager_loss should be > 0 because BCE on one-hot targets."""
    env = _FakeEnv(N_AGENTS, N_THREADS, OBS_SHAPE, N_ACTIONS)
    runner = MeltingpotRunner(meta_runner_cfg, env)
    infos = runner.run(num_episodes=1)
    assert len(infos) == 1
    for per_agent_info in infos[0]["per_agent"]:
        assert per_agent_info["wager_loss_actor"] > 0
        assert per_agent_info["wager_loss_critic"] > 0


def test_runner_collect_returns_expected_keys(runner_cfg):
    env = _FakeEnv(N_AGENTS, N_THREADS, OBS_SHAPE, N_ACTIONS)
    runner = MeltingpotRunner(runner_cfg, env)
    runner.warmup()
    rollout = runner.collect(step=0)
    assert len(rollout) == N_AGENTS
    expected_keys = {"actions", "action_log_probs", "values", "rnn_states", "rnn_states_critic"}
    for r in rollout:
        assert expected_keys.issubset(r.keys())
        assert r["actions"].shape == (N_THREADS, 1)
        assert r["values"].shape == (N_THREADS, 1)
        assert r["rnn_states"].shape[0] == N_THREADS


# ──────────────────────────────────────────────────────────────
# Episode reward logging (E.17b2) — required to compare to paper Table 7.
# ──────────────────────────────────────────────────────────────


class _ConstRewardEnv(_FakeEnv):
    """Env that always returns reward = ``const`` per agent per step.

    Useful for exact-arithmetic reward-logging test : with constant rewards,
    the per-episode return is deterministic and we can check the aggregator.
    """

    def __init__(self, num_agents, n_threads, obs_shape, n_actions, const):
        super().__init__(num_agents, n_threads, obs_shape, n_actions)
        self.const = float(const)

    def step(self, action_dict):
        obs = self._obs_dict()
        rewards = {k: np.full(self.n_threads, self.const, dtype=np.float32) for k in self._player_keys}
        dones = {k: np.zeros(self.n_threads, dtype=bool) for k in self._player_keys}
        return obs, rewards, dones, {}


def test_runner_logs_per_episode_return(runner_cfg):
    """Constant reward per step × episode_length steps = deterministic return."""
    const = 0.5
    env = _ConstRewardEnv(N_AGENTS, N_THREADS, OBS_SHAPE, N_ACTIONS, const=const)
    runner = MeltingpotRunner(runner_cfg, env)
    infos = runner.run(num_episodes=2)
    assert len(infos) == 2
    for ep_info in infos:
        assert "episode_return_mean" in ep_info
        assert "episode_return_per_agent" in ep_info
        # Each agent accumulates ``const`` for every one of ``EPISODE_LEN`` steps.
        expected_return = const * EPISODE_LEN
        assert len(ep_info["episode_return_per_agent"]) == N_AGENTS
        for per_agent in ep_info["episode_return_per_agent"]:
            assert abs(per_agent - expected_return) < 1e-5
        assert abs(ep_info["episode_return_mean"] - expected_return) < 1e-5
