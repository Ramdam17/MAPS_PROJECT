"""MAPPO buffer / rollout bit-exact parity vs student (E.14, tier-2).

Verifies that our port's :class:`RolloutBuffer` produces numerically identical
outputs to the student ``SeparatedReplayBuffer`` (stripped, vendored at
``tests/parity/_student_ref/marl/buffer.py``) for :

1. :meth:`insert` + :meth:`after_update` — state parity after a full rollout.
2. :meth:`compute_returns` (GAE + ValueNorm) — returns bit-exact.
3. :meth:`feed_forward_generator` — mini-batch shapes + values bit-exact
   under a seeded ``torch.randperm``.
4. :meth:`recurrent_generator` — chunk-based mini-batches bit-exact.

atol = 1e-6 on all floating-point arrays.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent / "_student_ref"))

from marl.buffer import SeparatedReplayBuffer as RefBuffer  # noqa: E402

from maps.experiments.marl.data import RolloutBuffer  # noqa: E402
from maps.experiments.marl.valuenorm import ValueNorm  # noqa: E402


ATOL = 1e-6
T = 8            # episode_length
N = 3            # n_rollout_threads
H = 32           # hidden size
RECURRENT_N = 1
OBS_SHAPE = (11, 11, 3)
N_ACTIONS = 8
SEED = 7


# ──────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────


def _make_ours() -> RolloutBuffer:
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
        use_valuenorm=True,
    )


def _make_ref() -> RefBuffer:
    act_space = spaces.Discrete(N_ACTIONS)
    return RefBuffer(
        episode_length=T,
        n_rollout_threads=N,
        hidden_size=H,
        recurrent_N=RECURRENT_N,
        gamma=0.99,
        gae_lambda=0.95,
        obs_shape=OBS_SHAPE,
        share_obs_shape=OBS_SHAPE,
        act_space=act_space,
    )


def _fill_both(ours: RolloutBuffer, ref: RefBuffer) -> None:
    """Insert T identical random steps into both buffers."""
    rng = np.random.default_rng(SEED)
    for _ in range(T):
        so = rng.random((N, *OBS_SHAPE), dtype=np.float32)
        o = rng.random((N, *OBS_SHAPE), dtype=np.float32)
        rh = rng.random((N, RECURRENT_N, H), dtype=np.float32)
        rhc = rng.random((N, RECURRENT_N, H), dtype=np.float32)
        a = rng.integers(0, N_ACTIONS, (N, 1)).astype(np.float32)
        alp = rng.random((N, 1), dtype=np.float32) - 2.0
        vp = rng.random((N, 1), dtype=np.float32)
        r = rng.random((N, 1), dtype=np.float32) - 0.5
        m = np.ones((N, 1), dtype=np.float32)

        ours.insert(so, o, rh, rhc, a, alp, vp, r, m)
        ref.insert(so, o, rh, rhc, a, alp, vp, r, m)


# ──────────────────────────────────────────────────────────────
# Insert + after_update
# ──────────────────────────────────────────────────────────────


def test_buffer_insert_state_matches_student():
    ours = _make_ours()
    ref = _make_ref()
    _fill_both(ours, ref)

    assert np.allclose(ours.share_obs, ref.share_obs, atol=ATOL)
    assert np.allclose(ours.obs, ref.obs, atol=ATOL)
    assert np.allclose(ours.rnn_states, ref.rnn_states, atol=ATOL)
    assert np.allclose(ours.rnn_states_critic, ref.rnn_states_critic, atol=ATOL)
    assert np.allclose(ours.actions, ref.actions, atol=ATOL)
    assert np.allclose(ours.action_log_probs, ref.action_log_probs, atol=ATOL)
    assert np.allclose(ours.value_preds, ref.value_preds, atol=ATOL)
    assert np.allclose(ours.rewards, ref.rewards, atol=ATOL)
    assert np.allclose(ours.masks, ref.masks, atol=ATOL)


def test_after_update_copies_last_to_first_matches_student():
    ours = _make_ours()
    ref = _make_ref()
    _fill_both(ours, ref)
    ours.after_update()
    ref.after_update()
    assert np.allclose(ours.share_obs[0], ref.share_obs[0], atol=ATOL)
    assert np.allclose(ours.rnn_states[0], ref.rnn_states[0], atol=ATOL)
    assert np.allclose(ours.masks[0], ref.masks[0], atol=ATOL)


# ──────────────────────────────────────────────────────────────
# GAE compute_returns — with and without ValueNorm
# ──────────────────────────────────────────────────────────────


def test_compute_returns_gae_without_valuenorm_matches_student():
    """Disable valuenorm on both to isolate GAE math."""
    ours = _make_ours()
    ours._use_valuenorm = False
    ref = _make_ref()
    ref._use_valuenorm = False
    _fill_both(ours, ref)

    next_value = np.random.default_rng(99).random((N, 1), dtype=np.float32)
    ours.compute_returns(next_value, value_normalizer=None)
    ref.compute_returns(next_value, value_normalizer=None)
    assert np.allclose(ours.returns, ref.returns, atol=ATOL)


def test_compute_returns_gae_with_valuenorm_matches_student():
    """GAE + ValueNorm path — both buffers use the same ValueNorm instance."""
    vn = ValueNorm(1, beta=0.99)
    # Train vn on some fake data so denormalize is non-trivial.
    vn.update(torch.from_numpy(np.random.randn(128, 1).astype(np.float32)))

    ours = _make_ours()
    ref = _make_ref()
    _fill_both(ours, ref)

    next_value = np.random.default_rng(100).random((N, 1), dtype=np.float32)
    ours.compute_returns(next_value, value_normalizer=vn)
    ref.compute_returns(next_value, value_normalizer=vn)
    assert np.allclose(ours.returns, ref.returns, atol=ATOL)


# ──────────────────────────────────────────────────────────────
# feed_forward_generator
# ──────────────────────────────────────────────────────────────


def _as_np(x):
    """Convert a possibly-torch tensor to numpy for cross-yield comparison."""
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


@pytest.mark.parametrize("num_mini_batch", [1, 2, 4])
def test_feed_forward_generator_matches_student_seeded(num_mini_batch):
    ours = _make_ours()
    ref = _make_ref()
    _fill_both(ours, ref)

    next_value = np.zeros((N, 1), dtype=np.float32)
    ours.compute_returns(next_value, value_normalizer=None)
    ref.compute_returns(next_value, value_normalizer=None)
    # Recompute ours-advantages and ref-advantages identically for generator input.
    adv = ours.returns[:-1] - ours.value_preds[:-1]

    torch.manual_seed(SEED)
    ref_batches = list(ref.feed_forward_generator(adv, num_mini_batch))
    torch.manual_seed(SEED)
    our_batches = list(ours.feed_forward_generator(adv, num_mini_batch))

    assert len(ref_batches) == len(our_batches) == num_mini_batch
    # Both yield 12-tuples with identical layout.
    for rb, ob in zip(ref_batches, our_batches):
        for i, (ref_arr, our_arr) in enumerate(zip(rb, ob)):
            if ref_arr is None and our_arr is None:
                continue
            ref_np = _as_np(ref_arr)
            our_np = _as_np(our_arr)
            assert ref_np.shape == our_np.shape, f"shape mismatch at index {i}"
            assert np.allclose(
                ref_np.astype(np.float64), our_np.astype(np.float64), atol=ATOL
            ), f"feed-forward generator divergence at tuple-idx {i}"


# ──────────────────────────────────────────────────────────────
# recurrent_generator (chunked)
# ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("chunk_length", [2, 4])
@pytest.mark.parametrize("num_mini_batch", [1, 2])
def test_recurrent_generator_matches_student_seeded(chunk_length, num_mini_batch):
    ours = _make_ours()
    ref = _make_ref()
    _fill_both(ours, ref)

    next_value = np.zeros((N, 1), dtype=np.float32)
    ours.compute_returns(next_value, value_normalizer=None)
    ref.compute_returns(next_value, value_normalizer=None)
    adv = ours.returns[:-1] - ours.value_preds[:-1]

    torch.manual_seed(SEED)
    ref_batches = list(ref.recurrent_generator(adv, num_mini_batch, chunk_length))
    torch.manual_seed(SEED)
    our_batches = list(ours.recurrent_generator(adv, num_mini_batch, chunk_length))

    assert len(ref_batches) == len(our_batches) == num_mini_batch
    for rb, ob in zip(ref_batches, our_batches):
        for i, (ref_arr, our_arr) in enumerate(zip(rb, ob)):
            if ref_arr is None and our_arr is None:
                continue
            ref_np = _as_np(ref_arr)
            our_np = _as_np(our_arr)
            assert ref_np.shape == our_np.shape, f"shape mismatch at index {i}"
            assert np.allclose(
                ref_np.astype(np.float64), our_np.astype(np.float64), atol=ATOL
            ), f"recurrent generator divergence at tuple-idx {i}"
