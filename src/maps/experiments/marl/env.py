"""MeltingPot env wrappers (E.10).

Ports student ``onpolicy/envs/meltingpot/MeltingPot_Env.py`` trimmed to :

- :func:`spec_to_space` : dm_env.specs → gymnasium.spaces converter.
- :func:`timestep_to_observations` : dm_env.TimeStep → per-player dict
  restricted to ``{"RGB", "WORLD.RGB"}`` per paper Fig.4.
- :class:`MeltingPotEnv` : gymnasium-style multi-agent wrapper over a
  dmlab2d environment. Exposes the env contract the E.9b runner expects :
  ``reset() → (obs_dict, info)`` / ``step(action_dict) → (obs, rew, done, info)``.
- :func:`downsample_observation` + :class:`DownSamplingSubstrateWrapper` :
  scales substrate RGB from 8x sprites to 1x (88×88 → 11×11 by default).
  See paper §A.4 ``obs_shape_agent = (11, 11, 3)``.
- :func:`env_creator` : substrate_id + roles → DownSample → MeltingPotEnv.

Simplifications vs student (E.5 scope) :
- Dropped ray.rllib.MultiAgentEnv inheritance — use plain gymnasium.Env
  protocol. Runner doesn't need RLLib integration.
- Dropped DataExtractor debug helper (plot/save RGB stills — dev-only).
- Dropped n_rollout_threads > 1 support (``step`` handles a single
  dm_env.step call per call — vectorization is a future concern).

Runtime
-------
``dmlab2d``, ``meltingpot``, and ``cv2`` are lazy-imported inside the
constructors/factories because they only exist in ``.venv-marl`` (Python
3.11) — the main 3.12 venv cannot install them. Tests in the main venv
can still import this module and exercise the pure-Python helpers
(``spec_to_space``, ``timestep_to_observations``, ``MeltingPotEnv``) with a
mock dm_env.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from gymnasium import spaces

__all__ = [
    "DownSamplingSubstrateWrapper",
    "MeltingPotEnv",
    "downsample_observation",
    "env_creator",
    "remove_world_observations_from_space",
    "spec_to_space",
    "timestep_to_observations",
]

log = logging.getLogger(__name__)

PLAYER_STR_FORMAT = "player_{index}"
MAX_CYCLES = 400
_OBSERVATION_PREFIX: tuple[str, ...] = ("WORLD.RGB", "RGB")
_WORLD_PREFIX: tuple[str, ...] = ("WORLD.RGB", "INTERACTION_INVENTORIES", "NUM_OTHERS_WHO_CLEANED_THIS_STEP")


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python helpers (no dmlab2d / meltingpot dependency)
# ─────────────────────────────────────────────────────────────────────────────


def spec_to_space(spec: Any) -> spaces.Space:
    """Convert a ``dm_env.specs`` node (or nested tuple/dict thereof) to a
    gymnasium ``Space``.

    Ports student L51-80. Handles :
    - :class:`dm_env.specs.DiscreteArray` → :class:`spaces.Discrete`.
    - :class:`dm_env.specs.BoundedArray` → :class:`spaces.Box`.
    - :class:`dm_env.specs.Array` → :class:`spaces.Box` with dtype-appropriate bounds.
    - ``list`` / ``tuple`` → :class:`spaces.Tuple` (recursive).
    - ``dict`` → :class:`spaces.Dict` (recursive).
    """
    import dm_env

    if isinstance(spec, dm_env.specs.DiscreteArray):
        return spaces.Discrete(int(spec.num_values))
    if isinstance(spec, dm_env.specs.BoundedArray):
        # dm_env stores minimum/maximum as 0-d arrays ; gymnasium's Box
        # requires either Python scalars (for broadcast) or arrays matching
        # ``shape``. Collapse 0-d arrays to ``.item()`` scalars.
        def _scalarize_bound(b):
            arr = np.asarray(b)
            return arr.item() if arr.ndim == 0 else arr

        low = _scalarize_bound(spec.minimum)
        high = _scalarize_bound(spec.maximum)
        return spaces.Box(low=low, high=high, shape=spec.shape, dtype=spec.dtype)
    if isinstance(spec, dm_env.specs.Array):
        if np.issubdtype(spec.dtype, np.floating):
            return spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
        if np.issubdtype(spec.dtype, np.integer):
            info = np.iinfo(spec.dtype)
            return spaces.Box(info.min, info.max, spec.shape, spec.dtype)
        raise NotImplementedError(f"Unsupported dtype {spec.dtype}")
    if isinstance(spec, (list, tuple)):
        return spaces.Tuple([spec_to_space(s) for s in spec])
    if isinstance(spec, dict):
        return spaces.Dict({key: spec_to_space(s) for key, s in spec.items()})
    raise ValueError(f"Unexpected spec of type {type(spec)}: {spec!r}")


def timestep_to_observations(timestep: Any) -> dict[str, dict[str, np.ndarray]]:
    """Convert a ``dm_env.TimeStep`` to ``{player_i: {"RGB": ..., "WORLD.RGB": ...}}``.

    Drops all per-player observations other than those listed in
    :data:`_OBSERVATION_PREFIX`. Student L32-40.
    """
    gym_observations: dict[str, dict[str, np.ndarray]] = {}
    for index, observation in enumerate(timestep.observation):
        gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
            key: value for key, value in observation.items() if key in _OBSERVATION_PREFIX
        }
    return gym_observations


def remove_world_observations_from_space(observation: spaces.Dict) -> spaces.Dict:
    """Strip the ``WORLD.*`` keys from a per-player :class:`spaces.Dict`."""
    return spaces.Dict(
        {key: observation[key] for key in observation if key not in _WORLD_PREFIX}
    )


def downsample_observation(array: np.ndarray, scaled: int) -> np.ndarray:
    """Downsample an RGB frame by ``scaled`` (spatial integer divisor).

    Uses ``cv2.INTER_AREA`` — the standard anti-aliased downsampling
    filter (student L303-315). Lazy-imports ``cv2`` so the main ``.venv``
    can still import this module for tests that don't call this path.
    """
    import cv2

    new_w = array.shape[0] // int(scaled)
    new_h = array.shape[1] // int(scaled)
    return cv2.resize(array, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _downsample_multi_timestep(timestep: Any, scaled: int) -> Any:
    """Apply :func:`downsample_observation` to each per-player RGB / WORLD.RGB
    field of a ``dm_env.TimeStep``. Student L318-321."""
    return timestep._replace(
        observation=[
            {
                k: downsample_observation(v, scaled) if k in _OBSERVATION_PREFIX else v
                for k, v in observation.items()
            }
            for observation in timestep.observation
        ]
    )


def _downsample_multi_spec(spec: Any, scaled: int) -> Any:
    """Shrink a dm_env Array spec by ``scaled`` along the first two axes."""
    import dm_env

    return dm_env.specs.Array(
        shape=(spec.shape[0] // int(scaled), spec.shape[1] // int(scaled), spec.shape[2]),
        dtype=spec.dtype,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MeltingPotEnv
# ─────────────────────────────────────────────────────────────────────────────


class MeltingPotEnv:
    """Gymnasium-style multi-agent wrapper over a dmlab2d MeltingPot env.

    Parameters
    ----------
    env : dmlab2d.Environment
        Underlying substrate (or downsampled wrapper). Will be closed by
        :meth:`close`.
    max_cycles : int
        Truncation limit (student default 400 = MAX_CYCLES).

    Contract (matches the runner's env interface from E.9b) :
    - ``reset() → (obs_dict, info)`` with
      ``obs_dict[player_i][{"RGB","WORLD.RGB"}]`` of shape ``(1, H, W, C)``.
    - ``step(action_dict) → (obs_dict, reward_dict, done_dict, info)``.
      ``action_dict[player_i]`` may be a scalar int, a ``(1,)`` array, or
      a ``(1, 1)`` array. Rewards and dones are ``(1,)`` arrays.
    - ``observation_space`` : per-player :class:`spaces.Dict` with ``RGB``
      only (WORLD.* removed).
    - ``share_observation_space`` : per-player :class:`spaces.Dict` with
      ``WORLD.RGB`` only (centralized obs for MAPPO critic).
    - ``action_space`` : per-player :class:`spaces.Dict` of Discrete.

    Concurrency : this class is NOT vectorized. For ``n_rollout_threads > 1``
    wrap multiple instances via a VectorEnv at the runner level (future scope).
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, env: Any, max_cycles: int = MAX_CYCLES):
        self._env = env
        self._num_players = len(self._env.observation_spec())
        self._ordered_agent_ids = [
            PLAYER_STR_FORMAT.format(index=index) for index in range(self._num_players)
        ]
        self._agent_ids = set(self._ordered_agent_ids)

        obs_tuple_space = spec_to_space(self._env.observation_spec())
        self.observation_space = self._convert_spaces_tuple_to_dict(
            obs_tuple_space, remove_world_observations=True
        )
        self.action_space = self._convert_spaces_tuple_to_dict(
            spec_to_space(self._env.action_spec())
        )
        self.share_observation_space = self._create_world_rgb_observation_space(
            self._env.observation_spec()
        )

        self.max_cycles = int(max_cycles)
        self.num_cycles = 0

    # ────────────────────────────────────────────────────────────
    # Gym-style API
    # ────────────────────────────────────────────────────────────

    def reset(self, *args: Any, **kwargs: Any) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
        """Reset the substrate + wrap obs to ``(1, H, W, C)`` per-thread convention."""
        timestep = self._env.reset()
        self.num_cycles = 0
        obs = timestep_to_observations(timestep)
        return self._expand_obs_to_thread_dim(obs), {}

    def step(
        self, action_dict: Mapping[str, Any]
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict,
    ]:
        """One dm_env step. Accepts per-agent action in scalar or (1,) form."""
        # Build a flat per-player int vector (N_players,) for dmlab2d.
        actions = np.asarray(
            [int(self._scalarize_action(action_dict[agent_id])) for agent_id in self._ordered_agent_ids]
        )

        timestep = self._env.step(actions)
        self.num_cycles += 1

        truncated = self.num_cycles >= self.max_cycles
        term = bool(timestep.last())

        obs = timestep_to_observations(timestep)
        obs = self._expand_obs_to_thread_dim(obs)

        rewards: dict[str, np.ndarray] = {}
        dones: dict[str, np.ndarray] = {}
        for idx, agent_id in enumerate(self._ordered_agent_ids):
            rewards[agent_id] = np.asarray([float(timestep.reward[idx])], dtype=np.float32)
            dones[agent_id] = np.asarray([term or truncated], dtype=bool)

        info: dict = {}
        return obs, rewards, dones, info

    def close(self) -> None:
        self._env.close()

    def render(self) -> np.ndarray:
        """Render the current world RGB — suitable for video recording.

        Student L240-255.
        """
        observation = self._env.observation()
        return observation[0]["WORLD.RGB"]

    def get_dmlab2d_env(self) -> Any:
        """Expose the underlying dmlab2d env (e.g. for frame extraction)."""
        return self._env

    # ────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _scalarize_action(a: Any) -> int:
        """Accept scalar / (1,) / (1, 1) shapes and return a single int."""
        arr = np.asarray(a)
        return int(arr.reshape(-1)[0])

    @staticmethod
    def _expand_obs_to_thread_dim(
        obs: Mapping[str, Mapping[str, np.ndarray]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Add a leading (n_rollout_threads=1) dim to each per-player RGB/WORLD.RGB."""
        out: dict[str, dict[str, np.ndarray]] = {}
        for player, d in obs.items():
            out[player] = {k: np.asarray(v)[None, ...] for k, v in d.items()}
        return out

    def _convert_spaces_tuple_to_dict(
        self,
        input_tuple: spaces.Tuple,
        remove_world_observations: bool = False,
    ) -> spaces.Dict:
        """Tuple → Dict keyed by ``player_i``. Student L257-271."""
        return spaces.Dict(
            {
                agent_id: (
                    remove_world_observations_from_space(input_tuple[i])
                    if remove_world_observations
                    else input_tuple[i]
                )
                for i, agent_id in enumerate(self._ordered_agent_ids)
            }
        )

    def _create_world_rgb_observation_space(self, observation_spec: Sequence[Mapping[str, Any]]) -> spaces.Dict:
        """Build the centralized-obs Dict space (WORLD.RGB only). Student L273-299."""
        world_rgb_spec = [player_obs_spec["WORLD.RGB"] for player_obs_spec in observation_spec]
        world_rgb_space = spaces.Tuple([spec_to_space(spec) for spec in world_rgb_spec])
        return spaces.Dict(
            {agent_id: world_rgb_space[i] for i, agent_id in enumerate(self._ordered_agent_ids)}
        )


# ─────────────────────────────────────────────────────────────────────────────
# DownSamplingSubstrateWrapper — subclass of meltingpot's observables wrapper
# ─────────────────────────────────────────────────────────────────────────────


def _load_observables_wrapper_base():
    """Lazy-import ``meltingpot.utils.substrates.wrappers.observables.ObservableLab2dWrapper``
    — available only in the MARL venv. Used to subclass at runtime."""
    from meltingpot.utils.substrates.wrappers import observables

    return observables.ObservableLab2dWrapper


def _make_downsampling_wrapper_class():
    """Factory for :class:`DownSamplingSubstrateWrapper` deferred until the
    meltingpot import is available (avoids import at module load on login node).

    Returns a class subclassing ``observables.ObservableLab2dWrapper`` that
    downsamples RGB / WORLD.RGB through :func:`downsample_observation`.
    """
    Base = _load_observables_wrapper_base()

    class _DownSamplingSubstrateWrapper(Base):
        """Downsamples per-step & per-spec RGB observations by an integer factor.

        Paper §A.4 uses an 11×11 RGB window → scaled=8 (88×88 → 11×11).
        Student L326-350. This subclasses the meltingpot ObservableLab2dWrapper
        so the downstream env sees a standard dmlab2d.Environment interface.
        """

        def __init__(self, substrate_instance: Any, scaled: int):
            super().__init__(substrate_instance)
            self._scaled = int(scaled)

        def reset(self):
            timestep = super().reset()
            return _downsample_multi_timestep(timestep, self._scaled)

        def step(self, actions):
            timestep = super().step(actions)
            return _downsample_multi_timestep(timestep, self._scaled)

        def observation_spec(self):
            spec = super().observation_spec()
            return [
                {
                    k: _downsample_multi_spec(v, self._scaled)
                    if k in _OBSERVATION_PREFIX
                    else v
                    for k, v in s.items()
                }
                for s in spec
            ]

    return _DownSamplingSubstrateWrapper


class DownSamplingSubstrateWrapper:
    """Thin factory that returns an instance of the lazily-built wrapper class.

    Usage ::

        wrapped = DownSamplingSubstrateWrapper(substrate, scaled=8)

    Equivalent to constructing the dynamically-built class.
    """

    def __new__(cls, substrate_instance: Any, scaled: int = 8):
        cls_real = _make_downsampling_wrapper_class()
        return cls_real(substrate_instance, scaled)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level factory
# ─────────────────────────────────────────────────────────────────────────────


def env_creator(
    substrate_id: str,
    roles: Sequence[str] | None = None,
    scaled: int = 8,
    max_cycles: int = MAX_CYCLES,
) -> MeltingPotEnv:
    """Build a fully-wrapped MeltingPot environment.

    Parameters
    ----------
    substrate_id : str
        E.g. ``"commons_harvest__closed"``, ``"chemistry__three_metabolic_cycles_with_plentiful_distractors"``.
    roles : sequence of str, optional
        Per-agent role strings. If ``None``, uses the substrate's
        ``default_player_roles`` (which defines paper-faithful num_agents).
    scaled : int
        RGB downsample factor (paper §A.4 = 8).
    max_cycles : int
        Step-count truncation (student default 400).

    Returns
    -------
    MeltingPotEnv
        Ready to ``.reset()`` / ``.step(action_dict)`` per the runner contract.
    """
    from meltingpot import substrate as meltingpot_substrate

    if roles is None:
        cfg = meltingpot_substrate.get_config(substrate_id)
        roles = list(cfg.default_player_roles)

    raw_env = meltingpot_substrate.build(substrate_id, roles=list(roles))
    downsampled = DownSamplingSubstrateWrapper(raw_env, scaled=scaled)
    return MeltingPotEnv(downsampled, max_cycles=max_cycles)
