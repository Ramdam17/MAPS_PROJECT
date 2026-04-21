"""Student ``SeparatedReplayBuffer`` — stripped to the path our port targets.

Dropped vs ``external/paper_reference/marl_tmlr/onpolicy/utils/separated_buffer.py`` :
- ``rnn_cells`` / ``rnn_cells_critic`` (LSTM ; we use GRU).
- ``bad_masks`` / ``_use_proper_time_limits`` path.
- ``naive_recurrent_generator``.
- ``chooseinsert`` / ``chooseafter_update`` (SC2).
- ``store_action_and_rnn_state`` (MI logging).
- ``_use_popart`` path (``use_popart=False`` per E.5).

The ``_use_gae = True`` + ``_use_valuenorm = True`` branch is the only path
we need for E.14 parity. Verbatim otherwise.
"""

from __future__ import annotations

import numpy as np
import torch


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    # Student's generic path : only called for (T, N, D1, ...)-shaped arrays.
    # For ndim == 3 (T, N, D), student uses `x.transpose(1, 0, 2).reshape(-1, *x.shape[2:])`.
    return x.transpose(1, 0, *range(2, x.ndim)).reshape(-1, *x.shape[2:])


class SeparatedReplayBuffer:
    """Stripped student SeparatedReplayBuffer (GAE + ValueNorm path only)."""

    def __init__(
        self,
        episode_length,
        n_rollout_threads,
        hidden_size,
        recurrent_N,
        gamma,
        gae_lambda,
        obs_shape,
        share_obs_shape,
        act_space,
    ):
        self.episode_length = int(episode_length)
        self.n_rollout_threads = int(n_rollout_threads)
        self.rnn_hidden_size = int(hidden_size)
        self.recurrent_N = int(recurrent_N)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self._use_valuenorm = True

        T = self.episode_length
        N = self.n_rollout_threads
        self.share_obs = np.zeros((T + 1, N, *share_obs_shape), dtype=np.float32)
        self.obs = np.zeros((T + 1, N, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros(
            (T + 1, N, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32
        )
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros((T + 1, N, 1), dtype=np.float32)
        self.returns = np.zeros((T + 1, N, 1), dtype=np.float32)

        assert act_space.__class__.__name__ == "Discrete", "parity ref is Discrete-only"
        self.available_actions = np.ones((T + 1, N, act_space.n), dtype=np.float32)

        self.actions = np.zeros((T, N, 1), dtype=np.float32)
        self.action_log_probs = np.zeros((T, N, 1), dtype=np.float32)
        self.rewards = np.zeros((T, N, 1), dtype=np.float32)

        self.masks = np.ones((T + 1, N, 1), dtype=np.float32)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(
        self,
        share_obs,
        obs,
        rnn_states,
        rnn_states_critic,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        active_masks=None,
        available_actions=None,
    ):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        self.available_actions[0] = self.available_actions[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """Student L195-209 — GAE + ValueNorm (no proper_time_limits)."""
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            if self._use_valuenorm and value_normalizer is not None:
                delta = (
                    self.rewards[step]
                    + self.gamma
                    * value_normalizer.denormalize(self.value_preds[step + 1])
                    * self.masks[step + 1]
                    - value_normalizer.denormalize(self.value_preds[step])
                )
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
            else:
                delta = (
                    self.rewards[step]
                    + self.gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self, advantages, num_mini_batch, mini_batch_size=None):
        """Student L215-270 — non-recurrent, random flat-index minibatches."""
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        if mini_batch_size is None:
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size] for i in range(num_mini_batch)
        ]

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
                share_obs[indices],
                obs[indices],
                rnn_states[indices],
                rnn_states_critic[indices],
                actions[indices],
                value_preds[indices],
                returns[indices],
                masks[indices],
                active_masks[indices],
                action_log_probs[indices],
                advantages[indices],
                available_actions[indices],
            )

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        """Student L352-468 — chunk-based recurrent minibatches."""
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size] for i in range(num_mini_batch)
        ]

        # Student's fork on obs ndim : (T, N, H, W, C) uses explicit transpose ;
        # smaller shapes use _cast. Both are numerically equivalent.
        if len(self.share_obs.shape) > 3:
            share_obs = self.share_obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.share_obs.shape[2:])
            obs = self.obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.obs.shape[2:])
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        available_actions = _cast(self.available_actions[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 0, 2, 3).reshape(
            -1, *self.rnn_states_critic.shape[2:]
        )

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                share_obs_batch.append(share_obs[ind : ind + data_chunk_length])
                obs_batch.append(obs[ind : ind + data_chunk_length])
                actions_batch.append(actions[ind : ind + data_chunk_length])
                available_actions_batch.append(available_actions[ind : ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                return_batch.append(returns[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind : ind + data_chunk_length])
                adv_targ.append(advantages[ind : ind + data_chunk_length])
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size
            share_obs_batch = np.stack(share_obs_batch)
            obs_batch = np.stack(obs_batch)
            actions_batch = np.stack(actions_batch)
            available_actions_batch = np.stack(available_actions_batch)
            value_preds_batch = np.stack(value_preds_batch)
            return_batch = np.stack(return_batch)
            masks_batch = np.stack(masks_batch)
            active_masks_batch = np.stack(active_masks_batch)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch)
            adv_targ = np.stack(adv_targ)

            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[2:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[2:]
            )

            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            available_actions_batch = _flatten(L, N, available_actions_batch)
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield (
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
            )
