"""Tier-1 parity: SARL replay buffer / get_state / target_wager (Sprint 14.C).

Bit-exact vs verbatim maps_v1.py extracts (D14.4 Tier 1).
"""

from __future__ import annotations

import random

import numpy as np
import torch

from maps.domains.sarl import data
from tests.parity.sarl import _student_extracts as ref


def _fake_transition(i):
    return (
        torch.full((1, 4, 10, 10), float(i)),
        torch.full((1, 4, 10, 10), float(i + 1)),
        torch.tensor([[i % 3]]),
        torch.tensor([[float(i)]]),
        torch.tensor([[0]]),
    )


def test_buffer_cyclic_overwrite_matches_reference():
    ours = data.SarlReplayBuffer(5)
    theirs = ref.replay_buffer(5)
    for i in range(8):  # 8 > 5 → wraps
        tr = _fake_transition(i)
        ours.add(*tr)
        theirs.add(*tr)
    assert len(ours) == len(theirs.buffer) == 5
    assert ours.location == theirs.location
    # Same stored rewards in the same slots after wrap.
    for o, t in zip(ours.buffer, theirs.buffer, strict=True):
        assert torch.equal(o.reward, t.reward)


def test_buffer_sample_same_rng_stream():
    ours = data.SarlReplayBuffer(50)
    theirs = ref.replay_buffer(50)
    for i in range(50):
        tr = _fake_transition(i)
        ours.add(*tr)
        theirs.add(*tr)
    random.seed(123)
    s_ours = ours.sample(8)
    random.seed(123)
    s_theirs = theirs.sample(8)
    # Same transitions sampled (compare the reward field as a fingerprint).
    assert [float(t.reward) for t in s_ours] == [float(t.reward) for t in s_theirs]


def test_get_state_shape_and_values():
    s = np.arange(10 * 10 * 4, dtype=np.float32).reshape(10, 10, 4)
    ours = data.get_state(s)
    theirs = ref.get_state(s)
    assert ours.shape == (1, 4, 10, 10)
    assert torch.equal(ours, theirs)


def test_target_wager_bit_exact():
    torch.manual_seed(0)
    rewards = torch.randn(32, 1)
    for alpha in (25, 45):  # v1 paper (25) and v2 (45)
        ours = data.target_wager(rewards, alpha)
        theirs = ref.target_wager(rewards, alpha)
        assert torch.equal(ours, theirs), f"alpha={alpha}"
        assert ours.shape == (32, 2)


def test_target_wager_bet_when_reward_beats_ema():
    # Monotonically increasing rewards → every G > EMA → all [1, 0].
    rewards = torch.arange(1, 6, dtype=torch.float32).view(-1, 1)
    out = data.target_wager(rewards, 25)
    assert torch.equal(out[:, 0], torch.ones(5))
    assert torch.equal(out[:, 1], torch.zeros(5))
