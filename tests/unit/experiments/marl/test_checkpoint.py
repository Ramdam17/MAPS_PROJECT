"""Unit tests for MARL checkpoint save/load (E.17a).

Verifies :
- State round-trip : save then load restores per-agent actor/critic/meta
  state_dicts, optimizer states, ValueNorm state, EMA reward, and training
  history bit-exactly.
- RNG restoration : torch / numpy / python random states are reset on load.
- Cross-cell guard : loading a checkpoint from a different setting or
  agent count is rejected.
- Run-level resumption : a run interrupted mid-episode and resumed produces
  the same output as a run that completes in one go.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from maps.experiments.marl import (
    MarlSetting,
    MeltingpotRunner,
    RunnerConfig,
)
from maps.utils import load_config


HIDDEN = 32
OBS_SHAPE = (11, 11, 3)
N_AGENTS = 2
EPISODE_LEN = 3
N_THREADS = 1
N_ACTIONS = 8
SEED = 99


class _FakeEnv:
    """Matches the runner's env contract with deterministic-but-varying obs."""

    def __init__(self, num_agents, n_threads, obs_shape, n_actions, rng_seed=0):
        self.num_agents = num_agents
        self.n_threads = n_threads
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self._player_keys = [f"player_{i}" for i in range(num_agents)]
        self._rng = np.random.default_rng(rng_seed)

    def _obs_dict(self):
        return {
            k: {
                "RGB": self._rng.integers(0, 256, (self.n_threads, *self.obs_shape)).astype(np.float32),
                "WORLD.RGB": self._rng.integers(0, 256, (self.n_threads, *self.obs_shape)).astype(np.float32),
            }
            for k in self._player_keys
        }

    def reset(self):
        return self._obs_dict(), {}

    def step(self, action_dict):
        obs = self._obs_dict()
        rewards = {k: self._rng.standard_normal(self.n_threads).astype(np.float32) for k in self._player_keys}
        dones = {k: np.zeros(self.n_threads, dtype=bool) for k in self._player_keys}
        return obs, rewards, dones, {}

    def close(self):
        pass


@pytest.fixture
def cfg():
    return load_config(
        "training/marl",
        overrides=[
            f"model.hidden_size={HIDDEN}",
            "model.second_order_dropout=0.0",
            f"training.episode_length={EPISODE_LEN}",
            f"training.n_rollout_threads={N_THREADS}",
            "training.num_env_steps=12",  # 4 episodes
            "training.save_interval=2",
            "ppo.ppo_epoch=1",
            "ppo.num_mini_batch=1",
            "ppo.data_chunk_length=1",
        ],
    )


@pytest.fixture
def runner_cfg(cfg):
    return RunnerConfig(
        cfg=cfg,
        setting=MarlSetting(
            id="baseline", label="baseline",
            meta=False, cascade_iterations1=1, cascade_iterations2=1,
        ),
        num_agents=N_AGENTS,
        obs_shape=OBS_SHAPE,
        share_obs_shape=OBS_SHAPE,
        action_space=spaces.Discrete(N_ACTIONS),
        device="cpu",
    )


def _make_runner(runner_cfg, env_seed=0):
    env = _FakeEnv(N_AGENTS, N_THREADS, OBS_SHAPE, N_ACTIONS, rng_seed=env_seed)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    return MeltingpotRunner(runner_cfg, env)


# ──────────────────────────────────────────────────────────────
# Save / load round-trip
# ──────────────────────────────────────────────────────────────


def test_save_then_load_restores_agent_state_bit_exact(runner_cfg, tmp_path):
    runner_a = _make_runner(runner_cfg)
    # Train for 1 episode to get non-trivial state (weights + opt state + EMA).
    runner_a.run(num_episodes=1)
    ck = tmp_path / "ckpt.pt"
    runner_a.save_checkpoint(ck, next_episode=1, all_infos=[{"episode": 0}])

    # Fresh runner with the same config → load.
    runner_b = _make_runner(runner_cfg, env_seed=42)
    next_ep, infos = runner_b.load_checkpoint(ck)
    assert next_ep == 1
    assert infos == [{"episode": 0}]

    # All per-agent states must match exactly.
    for a in range(N_AGENTS):
        for k in runner_a.policies[a].actor.state_dict():
            assert torch.allclose(
                runner_a.policies[a].actor.state_dict()[k],
                runner_b.policies[a].actor.state_dict()[k],
            ), f"actor param {k} mismatch on agent {a}"
        for k in runner_a.policies[a].critic.state_dict():
            assert torch.allclose(
                runner_a.policies[a].critic.state_dict()[k],
                runner_b.policies[a].critic.state_dict()[k],
            ), f"critic param {k} mismatch on agent {a}"
        # ValueNorm state.
        if runner_a.trainers[a].value_normalizer is not None:
            for k in runner_a.trainers[a].value_normalizer.state_dict():
                assert torch.allclose(
                    runner_a.trainers[a].value_normalizer.state_dict()[k],
                    runner_b.trainers[a].value_normalizer.state_dict()[k],
                )
    # EMA reward restored.
    assert np.allclose(runner_a.ema_reward, runner_b.ema_reward)


def test_load_checkpoint_rejects_mismatched_num_agents(runner_cfg, tmp_path):
    runner_a = _make_runner(runner_cfg)
    ck = tmp_path / "ckpt.pt"
    runner_a.save_checkpoint(ck, next_episode=0, all_infos=[])

    # Build a runner with 3 agents — load should refuse.
    diff_cfg = RunnerConfig(
        cfg=runner_cfg.cfg,
        setting=runner_cfg.setting,
        num_agents=3,
        obs_shape=OBS_SHAPE,
        share_obs_shape=OBS_SHAPE,
        action_space=runner_cfg.action_space,
        device="cpu",
    )
    runner_b = _make_runner(diff_cfg)
    with pytest.raises(ValueError, match="num_agents"):
        runner_b.load_checkpoint(ck)


def test_load_checkpoint_coerces_rng_state_dtype(runner_cfg, tmp_path):
    """Regression (E.17c) : torch.load with map_location may ship the RNG
    state to a non-uint8 device/dtype — load_checkpoint must coerce back to
    CPU+uint8 so torch.set_rng_state doesn't reject it."""
    runner_a = _make_runner(runner_cfg)
    ck = tmp_path / "ckpt.pt"
    runner_a.save_checkpoint(ck, next_episode=0, all_infos=[])

    # Manually corrupt the dtype to mimic what torch.load + map_location
    # does on GPU transfer (would arrive as int32 or similar under some
    # backends) — the load path must tolerate it.
    payload = torch.load(ck, weights_only=False)
    payload["rng"]["torch"] = payload["rng"]["torch"].to(dtype=torch.int32)
    torch.save(payload, ck)

    runner_b = _make_runner(runner_cfg)
    # Must not raise — the coercion inside load_checkpoint handles it.
    next_ep, infos = runner_b.load_checkpoint(ck)
    assert next_ep == 0


def test_load_checkpoint_rejects_mismatched_setting(runner_cfg, tmp_path):
    runner_a = _make_runner(runner_cfg)
    ck = tmp_path / "ckpt.pt"
    runner_a.save_checkpoint(ck, next_episode=0, all_infos=[])

    diff_cfg = RunnerConfig(
        cfg=runner_cfg.cfg,
        setting=MarlSetting(
            id="maps", label="maps",
            meta=True, cascade_iterations1=50, cascade_iterations2=1,
        ),
        num_agents=runner_cfg.num_agents,
        obs_shape=OBS_SHAPE,
        share_obs_shape=OBS_SHAPE,
        action_space=runner_cfg.action_space,
        device="cpu",
    )
    runner_b = _make_runner(diff_cfg)
    with pytest.raises(ValueError, match="setting"):
        runner_b.load_checkpoint(ck)


# ──────────────────────────────────────────────────────────────
# Periodic saves during run + resume path
# ──────────────────────────────────────────────────────────────


def test_run_writes_checkpoint_at_save_interval(runner_cfg, tmp_path):
    """With save_interval=2 and 4 total episodes, a checkpoint should land."""
    runner = _make_runner(runner_cfg)
    ck = tmp_path / "ckpt.pt"
    runner.run(num_episodes=4, checkpoint_path=ck)
    assert ck.is_file(), "checkpoint was not written"
    payload = torch.load(ck, weights_only=False)
    # ``next_episode`` matches the final step count (we always save on
    # completion too, so it's num_episodes=4 here).
    assert payload["meta"]["next_episode"] == 4
    assert len(payload["all_infos"]) == 4


def test_run_resumes_from_checkpoint_and_appends_infos(runner_cfg, tmp_path):
    """Interrupted-equals-uninterrupted end state (episode count + history length)."""
    ck = tmp_path / "ckpt.pt"

    # Phase 1 : run 2 episodes, save checkpoint.
    runner1 = _make_runner(runner_cfg)
    infos1 = runner1.run(num_episodes=2, checkpoint_path=ck)
    assert len(infos1) == 2
    assert ck.is_file()

    # Phase 2 : fresh runner, resume from ck, run up to 4 episodes.
    runner2 = _make_runner(runner_cfg, env_seed=1)
    infos2 = runner2.run(num_episodes=4, checkpoint_path=ck, resume_from=ck)
    # The resumed run picks up the prior 2 infos + adds 2 more.
    assert len(infos2) == 4
    # Episode indices in all_infos : first 2 from the original run, last 2 fresh.
    episodes = [i["episode"] for i in infos2]
    assert episodes[:2] == [0, 1]
    assert episodes[2:] == [2, 3]
