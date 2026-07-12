"""MARL on-policy rollout buffer — MAPS §5 (MeltingPot MAPPO).

Ported from ``external/paper_reference/marl_tmlr/onpolicy/utils/separated_buffer.py``
(the ``separated`` variant, used by ``meltingpot_runner``), reduced to the path
the MeltingPot runner exercises: GRU recurrence (no LSTM ``rnn_cells``), GAE with
value normalisation, no ``proper_time_limits``/popart/attention/MI-logging.

**M-C1 (recurrent_generator ordering)** — the source stacks chunks with
``np.stack`` (default axis 0) then ``_flatten`` (a plain reshape, no transpose),
giving **chunk-major** flat order. This is faithful to the paper (D16.2); the
Sprint-F3 investigation confirmed the audit's "time-major" reading came from the
unused ``naive_recurrent_generator`` / ``shared_buffer``. We reproduce chunk-major.

References
----------
Vargas et al. (2025), MAPS, TMLR submission §5, Table 7.
Yu et al. (2022). The surprising effectiveness of PPO in cooperative MARL (MAPPO).
"""

from __future__ import annotations

import numpy as np
import torch


def _flatten(t: int, n: int, x: np.ndarray) -> np.ndarray:
    """Verbatim ``_flatten`` (separated_buffer.py:10-11) — plain reshape, no transpose."""
    return x.reshape(t * n, *x.shape[2:])


def _cast(x: np.ndarray) -> np.ndarray:
    """Verbatim ``_cast`` (separated_buffer.py:14-15) — (T,N,D) → (N,T,D) → (N*T,D)."""
    return x.transpose(1, 0, 2).reshape(-1, *x.shape[2:])


class SeparatedReplayBuffer:
    """One agent's on-policy rollout buffer (GRU path). Port of the source subset."""

    def __init__(
        self,
        *,
        episode_length: int,
        n_rollout_threads: int,
        hidden_size: int,
        recurrent_n: int,
        gamma: float,
        gae_lambda: float,
        obs_shape: tuple[int, ...],
        share_obs_shape: tuple[int, ...],
        num_actions: int,
        use_valuenorm: bool = True,
    ) -> None:
        self.episode_length = episode_length
        self.n_rollout_threads = n_rollout_threads
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._use_valuenorm = use_valuenorm

        el, n = episode_length, n_rollout_threads
        self.share_obs = np.zeros((el + 1, n, *share_obs_shape), dtype=np.float32)
        self.obs = np.zeros((el + 1, n, *obs_shape), dtype=np.float32)
        self.rnn_states = np.zeros((el + 1, n, recurrent_n, hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)
        self.value_preds = np.zeros((el + 1, n, 1), dtype=np.float32)
        self.returns = np.zeros((el + 1, n, 1), dtype=np.float32)
        self.available_actions = np.ones((el + 1, n, num_actions), dtype=np.float32)
        self.actions = np.zeros((el, n, 1), dtype=np.float32)
        self.action_log_probs = np.zeros((el, n, 1), dtype=np.float32)
        self.rewards = np.zeros((el, n, 1), dtype=np.float32)
        self.masks = np.ones((el + 1, n, 1), dtype=np.float32)
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
    ) -> None:
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
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        self.available_actions[0] = self.available_actions[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None) -> None:
        """GAE (no proper_time_limits). Verbatim source L195-209 (use_gae branch)."""
        self.value_preds[-1] = next_value
        gae = 0.0
        for step in reversed(range(self.rewards.shape[0])):
            if self._use_valuenorm:
                assert value_normalizer is not None
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

    def recurrent_generator(
        self, advantages: np.ndarray, num_mini_batch: int, data_chunk_length: int
    ):
        """Chunked minibatches — CHUNK-MAJOR (M-C1, faithful). Source L352-461 (GRU subset).

        ``np.stack`` (default axis 0) → ``_flatten`` (plain reshape) keeps chunk-major
        order — the paper's actual ordering (see module docstring).
        """
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size] for i in range(num_mini_batch)
        ]

        if len(self.share_obs.shape) > 3:
            share_obs = (
                self.share_obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.share_obs.shape[2:])
            )
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
        rnn_states = (
            self.rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        )
        rnn_states_critic = (
            self.rnn_states_critic[:-1]
            .transpose(1, 0, 2, 3)
            .reshape(-1, *self.rnn_states_critic.shape[2:])
        )

        for indices in sampler:
            share_obs_b, obs_b, rnn_b, rnn_c_b = [], [], [], []
            act_b, avail_b, vp_b, ret_b, mask_b, amask_b, alp_b, adv_b = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for index in indices:
                ind = index * data_chunk_length
                share_obs_b.append(share_obs[ind : ind + data_chunk_length])
                obs_b.append(obs[ind : ind + data_chunk_length])
                act_b.append(actions[ind : ind + data_chunk_length])
                avail_b.append(available_actions[ind : ind + data_chunk_length])
                vp_b.append(value_preds[ind : ind + data_chunk_length])
                ret_b.append(returns[ind : ind + data_chunk_length])
                mask_b.append(masks[ind : ind + data_chunk_length])
                amask_b.append(active_masks[ind : ind + data_chunk_length])
                alp_b.append(action_log_probs[ind : ind + data_chunk_length])
                adv_b.append(advantages[ind : ind + data_chunk_length])
                rnn_b.append(rnn_states[ind])  # first step of each chunk
                rnn_c_b.append(rnn_states_critic[ind])

            length, n = data_chunk_length, mini_batch_size
            # np.stack default axis=0 → (n_chunks, L, ...) → _flatten reshape → CHUNK-MAJOR.
            share_obs_b = _flatten(length, n, np.stack(share_obs_b))
            obs_b = _flatten(length, n, np.stack(obs_b))
            act_b = _flatten(length, n, np.stack(act_b))
            avail_b = _flatten(length, n, np.stack(avail_b))
            vp_b = _flatten(length, n, np.stack(vp_b))
            ret_b = _flatten(length, n, np.stack(ret_b))
            mask_b = _flatten(length, n, np.stack(mask_b))
            amask_b = _flatten(length, n, np.stack(amask_b))
            alp_b = _flatten(length, n, np.stack(alp_b))
            adv_b = _flatten(length, n, np.stack(adv_b))
            rnn_b = np.stack(rnn_b).reshape(n, *self.rnn_states.shape[2:])
            rnn_c_b = np.stack(rnn_c_b).reshape(n, *self.rnn_states_critic.shape[2:])

            yield (
                share_obs_b,
                obs_b,
                rnn_b,
                rnn_c_b,
                act_b,
                vp_b,
                ret_b,
                mask_b,
                amask_b,
                alp_b,
                adv_b,
                avail_b,
            )
