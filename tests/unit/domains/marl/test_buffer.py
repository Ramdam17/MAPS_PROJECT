"""MARL SeparatedReplayBuffer — GAE + chunk-major recurrent_generator (Sprint 16.C).

Verifies the M-C1 property faithfully: the recurrent_generator yields CHUNK-MAJOR
minibatches (the paper's ordering — np.stack + _flatten, no transpose), NOT
time-major. No MinAtar/MeltingPot dependency — pure tensor test.
"""

from __future__ import annotations

import numpy as np

from maps.domains.marl.data import SeparatedReplayBuffer

T = 6  # episode length
N = 2  # rollout threads
OBS = (3,)
NUM_ACTIONS = 4


def _make_buffer(use_valuenorm=False):
    return SeparatedReplayBuffer(
        episode_length=T,
        n_rollout_threads=N,
        hidden_size=8,
        recurrent_n=1,
        gamma=0.99,
        gae_lambda=0.95,
        obs_shape=OBS,
        share_obs_shape=OBS,
        num_actions=NUM_ACTIONS,
        use_valuenorm=use_valuenorm,
    )


def test_shapes_on_construction():
    b = _make_buffer()
    assert b.obs.shape == (T + 1, N, *OBS)
    assert b.actions.shape == (T, N, 1)
    assert b.rnn_states.shape == (T + 1, N, 1, 8)
    assert b.available_actions.shape == (T + 1, N, NUM_ACTIONS)


def test_compute_returns_gae_zero_signal():
    b = _make_buffer(use_valuenorm=False)
    b.compute_returns(np.zeros((N, 1), dtype=np.float32))
    assert np.allclose(b.returns[:-1], 0.0)


def test_compute_returns_gae_positive_signal_monotone():
    b = _make_buffer(use_valuenorm=False)
    b.rewards[:] = 1.0
    b.compute_returns(np.zeros((N, 1), dtype=np.float32))
    # earliest step accumulates the most discounted future reward
    assert b.returns[0, 0, 0] > b.returns[T - 1, 0, 0] > 0


def test_recurrent_generator_is_chunk_major():
    """M-C1 faithful: each yielded chunk is a contiguous time-slice of ONE env
    (chunk-major), reproducing the paper's np.stack+_flatten ordering."""
    b = _make_buffer()
    # marker obs[t, n, 0] = t*10 + n
    for t in range(T):
        for n in range(N):
            b.obs[t, n, 0] = t * 10 + n
    advantages = np.zeros((T, N, 1), dtype=np.float32)
    # chunk = T, one minibatch → 2 chunks (one per env), no shuffle ambiguity in content
    samples = list(b.recurrent_generator(advantages, num_mini_batch=1, data_chunk_length=T))
    assert len(samples) == 1
    obs_b = samples[0][1]  # (T*n_chunks, 3)
    n_chunks = (T * N) // T  # = N
    markers = obs_b[:, 0].reshape(n_chunks, T)  # CHUNK-major reshape (rows = chunks)
    for c in range(n_chunks):
        row = markers[c]
        envs = np.round(row % 10).astype(int)
        steps = np.round(row // 10).astype(int)
        assert (envs == envs[0]).all(), f"chunk {c} mixes envs: {envs}"
        assert np.array_equal(steps, np.arange(T)), f"chunk {c} not time-contiguous: {steps}"


def test_recurrent_generator_shapes():
    b = _make_buffer()
    advantages = np.random.randn(T, N, 1).astype(np.float32)
    samples = list(b.recurrent_generator(advantages, num_mini_batch=2, data_chunk_length=3))
    assert len(samples) == 2
    total_chunks = (T * N) // 3
    mini = total_chunks // 2
    expected = mini * 3
    for s in samples:
        obs, rnn, avail = s[1], s[2], s[11]
        assert obs.shape == (expected, *OBS)
        assert rnn.shape == (mini, 1, 8)  # one initial state per chunk
        assert avail.shape == (expected, NUM_ACTIONS)
