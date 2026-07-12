"""Tier-2: MARL CNN encoder + RNN layer (Sprint 16.E)."""

from __future__ import annotations

import torch

from maps.domains.marl.encoder import CNNBase
from maps.domains.marl.rnn import RNNLayer
from maps.utils.seeding import set_all_seeds

H = W = 11
C = 3
HIDDEN = 64


def test_cnn_forward_shape_and_determinism():
    set_all_seeds(42)
    net = CNNBase((H, W, C), HIDDEN).eval()  # eval → BatchNorm running stats → deterministic
    x = torch.rand(8, H, W, C) * 255.0
    out_a = net(x)
    out_b = net(x)
    assert out_a.shape == (8, HIDDEN)
    assert torch.equal(out_a, out_b)


def test_rnn_rollout_mode_shapes():
    set_all_seeds(1)
    rnn = RNNLayer(HIDDEN, HIDDEN, recurrent_n=1, use_orthogonal=True)
    batch = 8
    x = torch.rand(batch, HIDDEN)
    hxs = torch.rand(batch, 1, HIDDEN)
    masks = torch.ones(batch, 1)
    out, new_hxs, cascade = rnn(x, hxs, masks, None, 1.0)
    assert out.shape == (batch, HIDDEN)
    assert new_hxs.shape == (batch, 1, HIDDEN)
    assert cascade.shape == (batch, HIDDEN)


def test_rnn_minibatch_mode_shapes():
    set_all_seeds(2)
    rnn = RNNLayer(HIDDEN, HIDDEN, recurrent_n=1, use_orthogonal=True)
    episode_len, batch_num = 5, 3
    x = torch.rand(episode_len * batch_num, HIDDEN)
    hxs = torch.rand(batch_num, 1, HIDDEN)
    masks = torch.ones(episode_len * batch_num, 1)
    out, new_hxs, _cascade = rnn(x, hxs, masks, None, 1.0)
    assert out.shape == (episode_len * batch_num, HIDDEN)
    assert new_hxs.shape == (batch_num, 1, HIDDEN)


def test_rnn_cascade_blend_semantics():
    """output_cascade1 = rate*raw + (1-rate)*prev (pre-norm); returned x = norm(raw).

    With prev=None, output_cascade1 is the raw GRU output; with rate=1 and any prev
    it must equal that raw output (no prev contribution).
    """
    set_all_seeds(3)
    rnn = RNNLayer(HIDDEN, HIDDEN, recurrent_n=1, use_orthogonal=True).eval()
    batch = 4
    x = torch.rand(batch, HIDDEN)
    hxs = torch.rand(batch, 1, HIDDEN)
    masks = torch.ones(batch, 1)
    prev = torch.rand(batch, HIDDEN)

    out_none, _h1, cascade_raw = rnn(x, hxs, masks, None, 0.02)  # prev=None → cascade = raw
    _out_blend, _h2, cascade_blend = rnn(x, hxs, masks, prev, 0.02)  # blends prev
    _out1, _h3, cascade_rate1 = rnn(x, hxs, masks, prev, 1.0)  # rate=1 → raw

    # Returned x is the normalised output (differs from the raw cascade).
    assert not torch.allclose(out_none, cascade_raw)
    # Blend with a real prev differs from the raw; rate=1 recovers the raw.
    assert not torch.allclose(cascade_blend, cascade_raw)
    assert torch.allclose(cascade_rate1, cascade_raw, atol=1e-6)
