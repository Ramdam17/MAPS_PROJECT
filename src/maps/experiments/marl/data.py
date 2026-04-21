"""Rollout buffer for MARL separated MAPPO (E.9b).

Ports student ``onpolicy/utils/separated_buffer.py:SeparatedReplayBuffer``
(L18-467), trimmed to what our MAPPO port actually uses.

One :class:`RolloutBuffer` is created per-agent (separated-MAPPO convention).
Each buffer holds one agent's rollout across ``episode_length`` steps and
``n_rollout_threads`` parallel envs.

Dropped from student (E.5 scope) :
- `rnn_cells` / `rnn_cells_critic` (LSTM path, we use GRU only).
- `naive_recurrent_generator` (uses per-env chunks, not required for our port).
- `chooseinsert` / `chooseafter_update` (SC2-specific).
- `store_action_and_rnn_state` (MI logging helper, off by default).
- `bad_masks` (proper_time_limits path, off by default).

The core contract preserved : ``insert(...)``, ``after_update()``,
``compute_returns(next_value, value_normalizer)``, and two mini-batch
generators (``feed_forward_generator``, ``recurrent_generator``) that yield
12-tuples matching :meth:`MAPPOTrainer.ppo_update`'s sample unpacking.
"""

from __future__ import annotations

import numpy as np
import torch

from maps.experiments.marl.util import get_shape_from_obs_space

__all__ = ["RolloutBuffer"]


def _flatten(T: int, N: int, x: np.ndarray) -> np.ndarray:
    return x.reshape(T * N, *x.shape[2:])


class RolloutBuffer:
    """Per-agent rollout buffer for separated MAPPO.

    Parameters
    ----------
    episode_length : int
        Number of env steps per episode.
    n_rollout_threads : int
        Number of parallel envs (each with its own rollout sequence).
    hidden_size : int
        RNN hidden dim (for ``rnn_states`` / ``rnn_states_critic``).
    recurrent_n : int
        GRU layer count (paper Fig.4 = 1).
    gamma : float
        Discount factor (standard PPO/MAPPO default = 0.99).
    gae_lambda : float
        GAE λ (student default = 0.95).
    obs_space : gymnasium.Space
        Per-agent observation space.
    share_obs_space : gymnasium.Space
        Centralized observation space (for critic).
    act_space : gymnasium.Space
        Discrete action space.
    use_valuenorm : bool
        Whether value predictions are normalized (denormalize before GAE).
    """

    def __init__(
        self,
        episode_length: int,
        n_rollout_threads: int,
        hidden_size: int,
        recurrent_n: int,
        gamma: float,
        gae_lambda: float,
        obs_space,
        share_obs_space,
        act_space,
        use_valuenorm: bool = True,
    ):
        self.episode_length = int(episode_length)
        self.n_rollout_threads = int(n_rollout_threads)
        self.hidden_size = int(hidden_size)
        self.recurrent_n = int(recurrent_n)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self._use_valuenorm = bool(use_valuenorm)

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        T, N = self.episode_length, self.n_rollout_threads
        self.share_obs = np.zeros((T + 1, N, *share_obs_shape), dtype=np.float32)
        self.obs = np.zeros((T + 1, N, *obs_shape), dtype=np.float32)
        self.rnn_states = np.zeros((T + 1, N, self.recurrent_n, self.hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)
        self.value_preds = np.zeros((T + 1, N, 1), dtype=np.float32)
        self.returns = np.zeros((T + 1, N, 1), dtype=np.float32)

        if act_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones((T + 1, N, act_space.n), dtype=np.float32)
            act_shape = 1
        else:
            raise NotImplementedError(
                f"RolloutBuffer supports Discrete action spaces only (E.5 scope). "
                f"Got {act_space.__class__.__name__}."
            )

        self.actions = np.zeros((T, N, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((T, N, act_shape), dtype=np.float32)
        self.rewards = np.zeros((T, N, 1), dtype=np.float32)

        self.masks = np.ones((T + 1, N, 1), dtype=np.float32)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    # ──────────────────────────────────────────────────────────────
    # Insertion
    # ──────────────────────────────────────────────────────────────

    def insert(
        self,
        share_obs: np.ndarray,
        obs: np.ndarray,
        rnn_states: np.ndarray,
        rnn_states_critic: np.ndarray,
        actions: np.ndarray,
        action_log_probs: np.ndarray,
        value_preds: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        active_masks: np.ndarray | None = None,
        available_actions: np.ndarray | None = None,
    ) -> None:
        """Store one step of rollout. ``step`` auto-increments mod episode_length."""
        self.share_obs[self.step + 1] = np.asarray(share_obs)
        self.obs[self.step + 1] = np.asarray(obs)
        self.rnn_states[self.step + 1] = np.asarray(rnn_states)
        self.rnn_states_critic[self.step + 1] = np.asarray(rnn_states_critic)
        self.actions[self.step] = np.asarray(actions)
        self.action_log_probs[self.step] = np.asarray(action_log_probs)
        self.value_preds[self.step] = np.asarray(value_preds)
        self.rewards[self.step] = np.asarray(rewards)
        self.masks[self.step + 1] = np.asarray(masks)
        if active_masks is not None:
            self.active_masks[self.step + 1] = np.asarray(active_masks)
        if available_actions is not None:
            self.available_actions[self.step + 1] = np.asarray(available_actions)

        self.step = (self.step + 1) % self.episode_length

    def after_update(self) -> None:
        """Copy step-T terminal state to step-0 for the next episode."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        self.available_actions[0] = self.available_actions[-1].copy()

    # ──────────────────────────────────────────────────────────────
    # GAE returns (student L163-213 minus proper_time_limits, popart)
    # ──────────────────────────────────────────────────────────────

    def compute_returns(self, next_value: np.ndarray, value_normalizer=None) -> None:
        """GAE advantage estimation. Sets ``self.returns`` in-place.

        Uses student's non-proper-time-limits + GAE path (L195-209).
        ``value_normalizer`` should be the trainer's ValueNorm (for denormalization)
        when ``use_valuenorm=True``.
        """
        self.value_preds[-1] = next_value
        gae = 0.0
        for step in reversed(range(self.rewards.shape[0])):
            if self._use_valuenorm and value_normalizer is not None:
                v_next = value_normalizer.denormalize(self.value_preds[step + 1])
                v_curr = value_normalizer.denormalize(self.value_preds[step])
                delta = self.rewards[step] + self.gamma * v_next * self.masks[step + 1] - v_curr
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + v_curr
            else:
                delta = (
                    self.rewards[step]
                    + self.gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]

    # ──────────────────────────────────────────────────────────────
    # Mini-batch generators (student L215-467 trimmed)
    # ──────────────────────────────────────────────────────────────

    def feed_forward_generator(self, advantages: np.ndarray, num_mini_batch: int):
        """Random permutation over flattened (T * N) items.

        Yields 12-tuples matching MAPPOTrainer.ppo_update's sample signature.
        """
        T, N = self.rewards.shape[0:2]
        batch_size = N * T
        assert batch_size >= num_mini_batch, (
            f"batch_size ({batch_size}) must be >= num_mini_batch ({num_mini_batch})"
        )
        mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size : (i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        # Flatten all (T + 1, N, ...) → (T * N, ...) via [:-1] slice for T-length data.
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            yield (
                torch.from_numpy(share_obs[indices]).float(),
                torch.from_numpy(obs[indices]).float(),
                torch.from_numpy(rnn_states[indices]).float(),
                torch.from_numpy(rnn_states_critic[indices]).float(),
                torch.from_numpy(actions[indices]).long(),
                value_preds[indices],
                returns[indices],
                torch.from_numpy(masks[indices]).float(),
                active_masks[indices],
                action_log_probs[indices],
                advantages[indices],
                available_actions[indices],
            )

    def recurrent_generator(self, advantages: np.ndarray, num_mini_batch: int, data_chunk_length: int):
        """Chunk-based generator for recurrent policies.

        Splits (T * N) into contiguous chunks of ``data_chunk_length`` steps
        per env, shuffles chunks, yields 12-tuples.
        """
        T, N = self.rewards.shape[0:2]
        batch_size = N * T
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = max(data_chunks // num_mini_batch, 1)

        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size] for i in range(num_mini_batch)
        ]

        # Reshape (T, N, ...) → (T * N, ...) via transpose+flatten, keeping
        # temporal contiguity along axis 0.
        def _cast(x):
            # (T, N, ...) → (N, T, ...) → (N * T, ...)
            return x.transpose(1, 0, *range(2, x.ndim)).reshape(-1, *x.shape[2:])

        share_obs = _cast(self.share_obs[:-1])
        obs = _cast(self.obs[:-1])
        rnn_states = _cast(self.rnn_states[:-1])
        rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages_flat = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            action_log_probs_batch = []
            value_preds_batch = []
            returns_batch = []
            masks_batch = []
            active_masks_batch = []
            available_actions_batch = []
            advantages_batch = []

            for idx in indices:
                start = idx * data_chunk_length
                end = start + data_chunk_length
                share_obs_batch.append(share_obs[start:end])
                obs_batch.append(obs[start:end])
                actions_batch.append(actions[start:end])
                action_log_probs_batch.append(action_log_probs[start:end])
                advantages_batch.append(advantages_flat[start:end])
                value_preds_batch.append(value_preds[start:end])
                returns_batch.append(returns[start:end])
                masks_batch.append(masks[start:end])
                active_masks_batch.append(active_masks[start:end])
                available_actions_batch.append(available_actions[start:end])
                # RNN state only from the first step of each chunk.
                rnn_states_batch.append(rnn_states[start])
                rnn_states_critic_batch.append(rnn_states_critic[start])

            L = data_chunk_length
            # (mini_batch_size, L, ...) → (L * mini_batch_size, ...)
            share_obs_b = np.stack(share_obs_batch).reshape(-1, *share_obs.shape[1:])
            obs_b = np.stack(obs_batch).reshape(-1, *obs.shape[1:])
            actions_b = np.stack(actions_batch).reshape(-1, actions.shape[-1])
            alp_b = np.stack(action_log_probs_batch).reshape(-1, action_log_probs.shape[-1])
            adv_b = np.stack(advantages_batch).reshape(-1, 1)
            vp_b = np.stack(value_preds_batch).reshape(-1, 1)
            ret_b = np.stack(returns_batch).reshape(-1, 1)
            m_b = np.stack(masks_batch).reshape(-1, 1)
            am_b = np.stack(active_masks_batch).reshape(-1, 1)
            aa_b = np.stack(available_actions_batch).reshape(-1, available_actions.shape[-1])
            rnns_b = np.stack(rnn_states_batch)  # (mini_batch_size, recurrent_n, H)
            rnnsc_b = np.stack(rnn_states_critic_batch)

            yield (
                torch.from_numpy(share_obs_b).float(),
                torch.from_numpy(obs_b).float(),
                torch.from_numpy(rnns_b).float(),
                torch.from_numpy(rnnsc_b).float(),
                torch.from_numpy(actions_b).long(),
                vp_b,
                ret_b,
                torch.from_numpy(m_b).float(),
                am_b,
                alp_b,
                adv_b,
                aa_b,
            )
