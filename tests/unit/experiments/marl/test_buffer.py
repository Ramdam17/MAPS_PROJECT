"""Unit tests for :class:`RolloutBuffer` (E.9b).

Checks insert / after_update / compute_returns (GAE) / feed_forward_generator /
recurrent_generator shapes and semantics.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from maps.experiments.marl.data import RolloutBuffer
from maps.experiments.marl.valuenorm import ValueNorm


T = 6  # episode length
N = 2  # n_rollout_threads
H = 32  # hidden size
RECURRENT_N = 1
OBS_SHAPE = (11, 11, 3)
N_ACTIONS = 8


def _make_buffer(use_valuenorm: bool = False) -> RolloutBuffer:
    obs_space = spaces.Box(low=0, high=255, shape=OBS_SHAPE, dtype=np.float32)
    share_obs_space = spaces.Box(low=0, high=255, shape=OBS_SHAPE, dtype=np.float32)
    act_space = spaces.Discrete(N_ACTIONS)
    return RolloutBuffer(
        episode_length=T,
        n_rollout_threads=N,
        hidden_size=H,
        recurrent_n=RECURRENT_N,
        gamma=0.99,
        gae_lambda=0.95,
        obs_space=obs_space,
        share_obs_space=share_obs_space,
        act_space=act_space,
        use_valuenorm=use_valuenorm,
    )


# ──────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────


def test_buffer_shapes_on_construction():
    buf = _make_buffer()
    assert buf.obs.shape == (T + 1, N, *OBS_SHAPE)
    assert buf.share_obs.shape == (T + 1, N, *OBS_SHAPE)
    assert buf.rnn_states.shape == (T + 1, N, RECURRENT_N, H)
    assert buf.rnn_states_critic.shape == (T + 1, N, RECURRENT_N, H)
    assert buf.value_preds.shape == (T + 1, N, 1)
    assert buf.returns.shape == (T + 1, N, 1)
    assert buf.actions.shape == (T, N, 1)
    assert buf.action_log_probs.shape == (T, N, 1)
    assert buf.rewards.shape == (T, N, 1)
    assert buf.masks.shape == (T + 1, N, 1)
    assert buf.active_masks.shape == (T + 1, N, 1)
    assert buf.available_actions.shape == (T + 1, N, N_ACTIONS)


def test_buffer_rejects_non_discrete_action_space():
    obs_space = spaces.Box(low=0, high=255, shape=OBS_SHAPE, dtype=np.float32)
    box_act = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    with pytest.raises(NotImplementedError, match="Discrete"):
        RolloutBuffer(
            episode_length=T,
            n_rollout_threads=N,
            hidden_size=H,
            recurrent_n=RECURRENT_N,
            gamma=0.99,
            gae_lambda=0.95,
            obs_space=obs_space,
            share_obs_space=obs_space,
            act_space=box_act,
        )


# ──────────────────────────────────────────────────────────────
# Insert / after_update
# ──────────────────────────────────────────────────────────────


def test_buffer_insert_advances_step_and_wraps():
    buf = _make_buffer()
    for _ in range(T):
        buf.insert(
            share_obs=np.zeros((N, *OBS_SHAPE), dtype=np.float32),
            obs=np.zeros((N, *OBS_SHAPE), dtype=np.float32),
            rnn_states=np.zeros((N, RECURRENT_N, H), dtype=np.float32),
            rnn_states_critic=np.zeros((N, RECURRENT_N, H), dtype=np.float32),
            actions=np.zeros((N, 1), dtype=np.float32),
            action_log_probs=np.zeros((N, 1), dtype=np.float32),
            value_preds=np.zeros((N, 1), dtype=np.float32),
            rewards=np.ones((N, 1), dtype=np.float32),
            masks=np.ones((N, 1), dtype=np.float32),
        )
    # T inserts must wrap step back to 0.
    assert buf.step == 0
    # rewards filled across the episode.
    assert np.all(buf.rewards == 1.0)


def test_buffer_after_update_copies_last_to_first():
    buf = _make_buffer()
    buf.obs[-1] = 7.0
    buf.share_obs[-1] = 3.0
    buf.masks[-1] = 0.5
    buf.rnn_states[-1] = 2.0
    buf.after_update()
    assert np.all(buf.obs[0] == 7.0)
    assert np.all(buf.share_obs[0] == 3.0)
    assert np.all(buf.masks[0] == 0.5)
    assert np.all(buf.rnn_states[0] == 2.0)


# ──────────────────────────────────────────────────────────────
# GAE returns
# ──────────────────────────────────────────────────────────────


def test_compute_returns_no_valuenorm_gae_equation():
    """Closed-form check : with rewards=0, value=0, mask=1 → returns=0."""
    buf = _make_buffer(use_valuenorm=False)
    # rewards, value_preds, masks are already zeros/ones by default.
    next_value = np.zeros((N, 1), dtype=np.float32)
    buf.compute_returns(next_value)
    # All returns should be 0 (no signal).
    assert np.allclose(buf.returns[:-1], 0.0)


def test_compute_returns_positive_signal():
    """With constant reward=1 and value=0, returns should be monotonically
    increasing going backwards (gamma^k accumulation)."""
    buf = _make_buffer(use_valuenorm=False)
    buf.rewards[:] = 1.0
    next_value = np.zeros((N, 1), dtype=np.float32)
    buf.compute_returns(next_value)
    # Earliest step's return is the largest accumulated signal.
    assert buf.returns[0, 0, 0] > buf.returns[T - 1, 0, 0]
    assert buf.returns[0, 0, 0] > 0


def test_compute_returns_with_valuenorm_denorm():
    """ValueNorm path executes without numerical failure."""
    buf = _make_buffer(use_valuenorm=True)
    vn = ValueNorm(1, beta=0.9)
    # Train vn on some fake data.
    vn.update(np.random.randn(100, 1).astype(np.float32))
    buf.rewards[:] = np.random.randn(*buf.rewards.shape).astype(np.float32)
    buf.value_preds[:] = np.random.randn(*buf.value_preds.shape).astype(np.float32)
    next_value = np.random.randn(N, 1).astype(np.float32)
    buf.compute_returns(next_value, value_normalizer=vn)
    assert np.all(np.isfinite(buf.returns))


# ──────────────────────────────────────────────────────────────
# Mini-batch generators
# ──────────────────────────────────────────────────────────────


def test_feed_forward_generator_yields_expected_shapes():
    buf = _make_buffer()
    advantages = np.random.randn(T, N, 1).astype(np.float32)
    num_mini_batch = 2
    samples = list(buf.feed_forward_generator(advantages, num_mini_batch))
    assert len(samples) == num_mini_batch
    mini = (T * N) // num_mini_batch
    for sample in samples:
        (share_obs, obs, rnns, rnnsc, actions, vp, ret, masks, am, alp, adv, aa) = sample
        assert share_obs.shape == (mini, *OBS_SHAPE)
        assert obs.shape == (mini, *OBS_SHAPE)
        assert rnns.shape == (mini, RECURRENT_N, H)
        assert rnnsc.shape == (mini, RECURRENT_N, H)
        assert actions.shape == (mini, 1)
        assert actions.dtype == torch.int64
        assert vp.shape == (mini, 1)
        assert ret.shape == (mini, 1)
        assert masks.shape == (mini, 1)
        assert am.shape == (mini, 1)
        assert alp.shape == (mini, 1)
        assert adv.shape == (mini, 1)
        assert aa.shape == (mini, N_ACTIONS)


def test_recurrent_generator_yields_expected_shapes():
    buf = _make_buffer()
    advantages = np.random.randn(T, N, 1).astype(np.float32)
    chunk_length = 3
    num_mini_batch = 2

    samples = list(buf.recurrent_generator(advantages, num_mini_batch, chunk_length))
    assert len(samples) == num_mini_batch

    total_chunks = (T * N) // chunk_length
    mini_chunks = max(total_chunks // num_mini_batch, 1)
    expected_batch = mini_chunks * chunk_length

    for sample in samples:
        (share_obs, obs, rnns, rnnsc, actions, vp, ret, masks, am, alp, adv, aa) = sample
        assert share_obs.shape == (expected_batch, *OBS_SHAPE)
        assert obs.shape == (expected_batch, *OBS_SHAPE)
        assert rnns.shape == (mini_chunks, RECURRENT_N, H)  # one state per chunk
        assert rnnsc.shape == (mini_chunks, RECURRENT_N, H)
        assert actions.shape == (expected_batch, 1)
        assert vp.shape == (expected_batch, 1)
        assert ret.shape == (expected_batch, 1)
        assert masks.shape == (expected_batch, 1)
        assert adv.shape == (expected_batch, 1)
        assert aa.shape == (expected_batch, N_ACTIONS)
