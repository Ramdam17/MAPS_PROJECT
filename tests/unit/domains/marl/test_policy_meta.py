"""Tier-2: MARL meta (wager) networks — Sprint 16.F (meta).

The meta path is a passive wager read-out (M-C2): these nets produce the (B, 2)
wager the trainer supervises with a BCE against ``reward>0``. They do NOT select
env actions. R_Critic_Meta is structural dead-weight (M-C3, never gets gradients
in the source). We test shapes and that the cascade loops run.
"""

from __future__ import annotations

import torch
from gymnasium import spaces

from maps.domains.marl.policy_meta import (
    R_Actor_Meta,
    R_Critic_Meta,
    RNNLayer_Meta,
    SecondOrderNetwork,
)
from maps.utils.seeding import set_all_seeds

H = W = 11
C = 3
HIDDEN = 64
RECN = 1
N_ACT = 6
B = 8


def _actor(cascade1=1, cascade2=1):
    return R_Actor_Meta(
        (H, W, C),
        spaces.Discrete(N_ACT),
        hidden_size=HIDDEN,
        recurrent_n=RECN,
        cascade_iterations_1=cascade1,
        cascade_iterations_2=cascade2,
    )


def test_second_order_wager_shape():
    """SecondOrderNetwork returns (wager (B,2), comparison_out (B,hidden))."""
    set_all_seeds(42)
    net = SecondOrderNetwork(HIDDEN).eval()
    comparison = torch.randn(B, HIDDEN)
    wager, comparison_out = net(comparison, None, 0.02)
    assert wager.shape == (B, 2)
    assert comparison_out.shape == (B, HIDDEN)


def test_second_order_cascade_blend():
    """With prev=None the comparison_out is the raw relu; a real prev blends in."""
    set_all_seeds(1)
    net = SecondOrderNetwork(HIDDEN).eval()
    comparison = torch.randn(B, HIDDEN)
    _w0, out_none = net(comparison, None, 0.02)
    prev = torch.randn(B, HIDDEN)
    _w1, out_blend = net(comparison, prev, 0.02)
    assert not torch.allclose(out_none, out_blend)


def test_rnn_meta_wager_false_path():
    """wager=False → returns (norm(cascade) (B,hidden), hxs, output_cascade1 (B,hidden))."""
    set_all_seeds(2)
    rnn = RNNLayer_Meta(HIDDEN, HIDDEN, RECN, use_orthogonal=True)
    x = torch.rand(B, HIDDEN)
    hxs = torch.rand(B, RECN, HIDDEN)
    masks = torch.ones(B, 1)
    out, new_hxs, cascade1 = rnn(x, hxs, masks, None, None, 0.02, 0.02, wager=False)
    assert out.shape == (B, HIDDEN)
    assert new_hxs.shape == (B, RECN, HIDDEN)
    assert cascade1.shape == (B, HIDDEN)


def test_rnn_meta_wager_true_path():
    """wager=True → returns (sigmoid wager (B,2), hxs, output_cascade2 (B,hidden))."""
    set_all_seeds(3)
    rnn = RNNLayer_Meta(HIDDEN, HIDDEN, RECN, use_orthogonal=True)
    x = torch.rand(B, HIDDEN)
    hxs = torch.rand(B, RECN, HIDDEN)
    masks = torch.ones(B, 1)
    wager, new_hxs, cascade2 = rnn(x, hxs, masks, None, None, 0.02, 0.02, wager=True)
    assert wager.shape == (B, 2)
    assert ((wager >= 0) & (wager <= 1)).all()  # sigmoid output
    assert new_hxs.shape == (B, RECN, HIDDEN)
    assert cascade2.shape == (B, HIDDEN)


def test_actor_meta_evaluate_actions_returns_wager():
    """R_Actor_Meta.evaluate_actions returns the (B, 2) wager."""
    set_all_seeds(4)
    actor = _actor().eval()
    obs = torch.rand(B, H, W, C) * 255
    rnn = torch.zeros(B, RECN, HIDDEN)
    masks = torch.ones(B, 1)
    wager = actor.evaluate_actions(obs, rnn, masks=masks)
    assert wager.shape == (B, 2)
    assert torch.isfinite(wager).all()


def test_actor_meta_cascade_setting_runs():
    """The cascade setting (50 iterations on both cascades) runs end-to-end."""
    set_all_seeds(5)
    actor = _actor(cascade1=50, cascade2=50).eval()
    obs = torch.rand(B, H, W, C) * 255
    rnn = torch.zeros(B, RECN, HIDDEN)
    wager = actor.evaluate_actions(obs, rnn, masks=torch.ones(B, 1))
    assert wager.shape == (B, 2)


def test_critic_meta_builds():
    """R_Critic_Meta is structural dead-weight (M-C3) — it must at least construct."""
    set_all_seeds(6)
    critic = R_Critic_Meta(
        (H, W, C),
        hidden_size=HIDDEN,
        recurrent_n=RECN,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
    )
    assert isinstance(critic.second_order, SecondOrderNetwork)
    assert critic.base.hidden_size == HIDDEN
