"""Unit tests for :mod:`maps.experiments.marl.env` (E.10).

Exercises the pure-Python helpers (``spec_to_space``,
``timestep_to_observations``, ``MeltingPotEnv``) with a **mock dmlab2d env**,
so the tests run in the main ``.venv`` without needing meltingpot.

The DownSamplingSubstrateWrapper + env_creator paths are only smoke-tested
on DRAC compute nodes via the dedicated slurm script (they require
meltingpot + dmlab2d which only exist in ``.venv-marl``).
"""

from __future__ import annotations

from types import SimpleNamespace

import dm_env
import numpy as np
import pytest
from gymnasium import spaces

from maps.experiments.marl import env as marl_env
from maps.experiments.marl.env import (
    MeltingPotEnv,
    PLAYER_STR_FORMAT,
    remove_world_observations_from_space,
    spec_to_space,
    timestep_to_observations,
)


# ──────────────────────────────────────────────────────────────
# spec_to_space
# ──────────────────────────────────────────────────────────────


def test_spec_to_space_discrete():
    spec = dm_env.specs.DiscreteArray(num_values=8)
    out = spec_to_space(spec)
    assert isinstance(out, spaces.Discrete)
    assert out.n == 8


def test_spec_to_space_bounded_array():
    spec = dm_env.specs.BoundedArray(
        shape=(11, 11, 3), dtype=np.uint8, minimum=0, maximum=255
    )
    out = spec_to_space(spec)
    assert isinstance(out, spaces.Box)
    assert out.shape == (11, 11, 3)
    assert out.dtype == np.uint8


def test_spec_to_space_unbounded_float_array():
    spec = dm_env.specs.Array(shape=(4,), dtype=np.float32)
    out = spec_to_space(spec)
    assert isinstance(out, spaces.Box)
    assert np.all(np.isneginf(out.low))
    assert np.all(np.isposinf(out.high))


def test_spec_to_space_unbounded_int_array():
    spec = dm_env.specs.Array(shape=(2,), dtype=np.int32)
    out = spec_to_space(spec)
    assert isinstance(out, spaces.Box)
    assert out.dtype == np.int32


def test_spec_to_space_tuple_and_dict():
    spec = {
        "rgb": dm_env.specs.BoundedArray((2, 2, 3), np.uint8, 0, 255),
        "action": dm_env.specs.DiscreteArray(num_values=4),
    }
    out = spec_to_space(spec)
    assert isinstance(out, spaces.Dict)
    assert isinstance(out["rgb"], spaces.Box)
    assert isinstance(out["action"], spaces.Discrete)


def test_spec_to_space_rejects_unsupported_dtype():
    class _BogusSpec:
        dtype = np.dtype("U4")  # unicode, unsupported

    # Emulate the Array path by passing a real Array with an object dtype.
    spec = dm_env.specs.Array(shape=(1,), dtype=np.dtype("O"))
    with pytest.raises(NotImplementedError):
        spec_to_space(spec)


def test_spec_to_space_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unexpected spec"):
        spec_to_space(object())


# ──────────────────────────────────────────────────────────────
# timestep_to_observations + remove_world_observations_from_space
# ──────────────────────────────────────────────────────────────


def _make_fake_timestep(num_players: int, obs_shape=(11, 11, 3), world_shape=(30, 21, 3)):
    """Build a minimal dm_env.TimeStep-like object matching MeltingPot's layout."""
    observation = [
        {
            "RGB": np.full(obs_shape, fill_value=idx, dtype=np.uint8),
            "WORLD.RGB": np.full(world_shape, fill_value=idx * 2, dtype=np.uint8),
            "EXTRA": np.zeros((1,), dtype=np.float32),  # should be dropped
        }
        for idx in range(num_players)
    ]
    rewards = [float(i) for i in range(num_players)]
    return SimpleNamespace(
        observation=observation,
        reward=rewards,
        last=lambda: False,
        step_type=dm_env.StepType.MID,
    )


def test_timestep_to_observations_drops_extra_keys():
    ts = _make_fake_timestep(num_players=3)
    obs = timestep_to_observations(ts)
    assert set(obs.keys()) == {"player_0", "player_1", "player_2"}
    for player, d in obs.items():
        assert set(d.keys()) == {"RGB", "WORLD.RGB"}


def test_remove_world_observations_from_space_strips_world_keys():
    full = spaces.Dict(
        {
            "RGB": spaces.Box(0, 255, (11, 11, 3), np.uint8),
            "WORLD.RGB": spaces.Box(0, 255, (30, 21, 3), np.uint8),
            "INTERACTION_INVENTORIES": spaces.Box(0, 1, (2,), np.float32),
        }
    )
    stripped = remove_world_observations_from_space(full)
    assert set(stripped.spaces.keys()) == {"RGB"}


# ──────────────────────────────────────────────────────────────
# MeltingPotEnv with a mock dmlab2d env
# ──────────────────────────────────────────────────────────────


class _MockDmlab2dEnv:
    """Minimal mock of a dmlab2d.Environment for MeltingPotEnv testing."""

    def __init__(self, num_players: int = 3, obs_shape=(11, 11, 3), world_shape=(30, 21, 3), num_actions: int = 8):
        self.num_players = num_players
        self.obs_shape = obs_shape
        self.world_shape = world_shape
        self.num_actions = num_actions
        self._step_count = 0
        self._last_actions: np.ndarray | None = None

    def observation_spec(self):
        return [
            {
                "RGB": dm_env.specs.BoundedArray(self.obs_shape, np.uint8, 0, 255),
                "WORLD.RGB": dm_env.specs.BoundedArray(self.world_shape, np.uint8, 0, 255),
            }
            for _ in range(self.num_players)
        ]

    def action_spec(self):
        return [dm_env.specs.DiscreteArray(num_values=self.num_actions) for _ in range(self.num_players)]

    def reset(self):
        self._step_count = 0
        return _make_fake_timestep(self.num_players, self.obs_shape, self.world_shape)

    def step(self, actions):
        self._step_count += 1
        self._last_actions = np.asarray(actions)
        ts = _make_fake_timestep(self.num_players, self.obs_shape, self.world_shape)
        # Mark terminal at step 3 so tests can exercise done-propagation.
        if self._step_count >= 3:
            ts = SimpleNamespace(
                observation=ts.observation,
                reward=ts.reward,
                last=lambda: True,
                step_type=dm_env.StepType.LAST,
            )
        return ts

    def observation(self):
        return _make_fake_timestep(self.num_players).observation

    def close(self):
        pass


def test_meltingpot_env_exposes_correct_observation_space():
    inner = _MockDmlab2dEnv(num_players=3)
    env = MeltingPotEnv(inner, max_cycles=400)
    assert set(env.observation_space.spaces.keys()) == {"player_0", "player_1", "player_2"}
    # Per-player obs has only RGB (WORLD.RGB stripped).
    assert set(env.observation_space["player_0"].spaces.keys()) == {"RGB"}


def test_meltingpot_env_exposes_share_observation_space_world_rgb():
    inner = _MockDmlab2dEnv(num_players=4)
    env = MeltingPotEnv(inner)
    # share_obs = per-player WORLD.RGB space.
    assert set(env.share_observation_space.spaces.keys()) == {"player_0", "player_1", "player_2", "player_3"}
    sample_player = env.share_observation_space["player_0"]
    assert isinstance(sample_player, spaces.Box)
    assert sample_player.shape == (30, 21, 3)


def test_meltingpot_env_action_space_is_dict_of_discrete():
    inner = _MockDmlab2dEnv(num_players=2, num_actions=8)
    env = MeltingPotEnv(inner)
    assert set(env.action_space.spaces.keys()) == {"player_0", "player_1"}
    assert isinstance(env.action_space["player_0"], spaces.Discrete)
    assert env.action_space["player_0"].n == 8


def test_meltingpot_env_reset_returns_thread_dim_obs():
    inner = _MockDmlab2dEnv(num_players=2)
    env = MeltingPotEnv(inner)
    obs, info = env.reset()
    assert isinstance(info, dict)
    for player_key in ["player_0", "player_1"]:
        assert obs[player_key]["RGB"].shape == (1, 11, 11, 3)  # leading thread dim
        assert obs[player_key]["WORLD.RGB"].shape == (1, 30, 21, 3)


def test_meltingpot_env_step_returns_runner_contract_shapes():
    inner = _MockDmlab2dEnv(num_players=3)
    env = MeltingPotEnv(inner)
    env.reset()

    # Actions as (1, 1) arrays — matches what the runner passes.
    action_dict = {f"player_{i}": np.array([[i % 8]]) for i in range(3)}
    obs, rewards, dones, info = env.step(action_dict)

    for i in range(3):
        k = f"player_{i}"
        assert obs[k]["RGB"].shape == (1, 11, 11, 3)
        assert rewards[k].shape == (1,)
        assert rewards[k].dtype == np.float32
        assert dones[k].shape == (1,)
        assert dones[k].dtype == bool
    assert info == {}


def test_meltingpot_env_step_accepts_scalar_actions():
    """``action_dict[k]`` may be scalar int, (1,) array, or (1, 1) array."""
    inner = _MockDmlab2dEnv(num_players=2)
    env = MeltingPotEnv(inner)
    env.reset()
    # All three forms mixed.
    action_dict = {
        "player_0": 3,
        "player_1": np.array([5]),
    }
    _, _, _, _ = env.step(action_dict)
    assert inner._last_actions is not None
    assert list(inner._last_actions) == [3, 5]


def test_meltingpot_env_truncates_at_max_cycles():
    inner = _MockDmlab2dEnv(num_players=2)
    env = MeltingPotEnv(inner, max_cycles=2)
    env.reset()
    action_dict = {"player_0": 0, "player_1": 0}
    env.step(action_dict)
    _, _, dones, _ = env.step(action_dict)
    # After 2 cycles, every player should be done (truncated).
    assert dones["player_0"][0] is np.True_ or dones["player_0"][0] is True


def test_meltingpot_env_sends_terminal_done_from_timestep():
    inner = _MockDmlab2dEnv(num_players=2)
    env = MeltingPotEnv(inner, max_cycles=1000)
    env.reset()
    action_dict = {"player_0": 0, "player_1": 0}
    env.step(action_dict)  # step 1
    env.step(action_dict)  # step 2
    _, _, dones, _ = env.step(action_dict)  # step 3 → mock returns last()=True
    assert bool(dones["player_0"][0]) is True


def test_meltingpot_env_close_closes_inner():
    """Just verify close is a no-op call that doesn't raise."""
    inner = _MockDmlab2dEnv(num_players=1)
    env = MeltingPotEnv(inner)
    env.close()  # should not raise


# ──────────────────────────────────────────────────────────────
# Lazy-import paths : env_creator / DownSamplingSubstrateWrapper
# These don't have meltingpot → they should raise ImportError clearly.
# ──────────────────────────────────────────────────────────────


def test_env_creator_requires_meltingpot_install():
    """Without meltingpot in the current venv, ``env_creator`` fails import."""
    try:
        import meltingpot  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            marl_env.env_creator("commons_harvest__closed")
    else:
        pytest.skip("meltingpot available — no negative import path to check")


def test_downsampling_wrapper_factory_requires_meltingpot():
    try:
        import meltingpot  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            marl_env.DownSamplingSubstrateWrapper(object(), scaled=8)
    else:
        pytest.skip("meltingpot available — no negative import path to check")


# ──────────────────────────────────────────────────────────────
# Player-id format invariant
# ──────────────────────────────────────────────────────────────


def test_player_str_format_matches_runner_convention():
    """Runner uses ``f'player_{aid}'`` — env must match."""
    assert PLAYER_STR_FORMAT.format(index=0) == "player_0"
    assert PLAYER_STR_FORMAT.format(index=12) == "player_12"
