"""Tier-2: MARL baseline R_Actor / R_Critic (Sprint 16.F)."""

from __future__ import annotations

import torch
from gymnasium import spaces

from maps.domains.marl.policy import R_Actor, R_Critic
from maps.utils.seeding import set_all_seeds

H = W = 11
C = 3
HIDDEN = 64
RECN = 1
N_ACT = 6
B = 8


def _actor(cascade=1):
    return R_Actor(
        (H, W, C),
        spaces.Discrete(N_ACT),
        hidden_size=HIDDEN,
        recurrent_n=RECN,
        cascade_iterations_1=cascade,
    )


def _critic(cascade=1):
    return R_Critic((H, W, C), hidden_size=HIDDEN, recurrent_n=RECN, cascade_iterations_1=cascade)


def test_actor_forward_shapes():
    set_all_seeds(42)
    actor = _actor().eval()
    obs = torch.rand(B, H, W, C) * 255
    rnn = torch.zeros(B, RECN, HIDDEN)
    masks = torch.ones(B, 1)
    actions, _log_probs, new_rnn = actor(obs, rnn, masks)
    assert actions.shape == (B, 1)
    assert (actions >= 0).all() and (actions < N_ACT).all()
    assert new_rnn.shape == (RECN, B, HIDDEN)


def test_actor_evaluate_actions_shapes():
    set_all_seeds(1)
    actor = _actor()
    obs = torch.rand(B, H, W, C) * 255
    rnn = torch.zeros(B, RECN, HIDDEN)
    masks = torch.ones(B, 1)
    action = torch.randint(0, N_ACT, (B, 1)).float()
    log_probs, entropy, wager = actor.evaluate_actions(obs, rnn, action, masks)
    assert log_probs.shape == (B, 1)
    assert entropy.dim() == 0
    assert wager.shape == (B, 2)  # fix (a): the acting actor also emits the co-training wager


def test_critic_forward_shapes():
    set_all_seeds(2)
    critic = _critic().eval()
    obs = torch.rand(B, H, W, C) * 255
    rnn = torch.zeros(B, RECN, HIDDEN)
    masks = torch.ones(B, 1)
    values, new_rnn = critic(obs, rnn, masks)
    assert values.shape == (B, 1)
    assert new_rnn.shape == (RECN, B, HIDDEN)


def test_cascade_setting_runs():
    """A cascade setting (50 iterations) runs end-to-end."""
    set_all_seeds(3)
    actor = _actor(cascade=50).eval()
    obs = torch.rand(B, H, W, C) * 255
    rnn = torch.zeros(B, RECN, HIDDEN)
    actions, _lp, _r = actor(obs, rnn, torch.ones(B, 1))
    assert actions.shape == (B, 1)
