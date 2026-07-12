"""R_MAPPOPolicy after fix (a): actor + critic only (no ghost meta nets); the confidence
judge lives inside the acting actor and is covered by ``actor_optimizer`` (co-training)."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from maps.domains.marl.policy import R_Actor, R_Critic
from maps.domains.marl.policy_meta import SecondOrderNetwork
from maps.domains.marl.rmappo_policy import R_MAPPOPolicy
from maps.utils.seeding import set_all_seeds

H = W = 11
C = 3
HIDDEN = 64
RECN = 1
N_ACT = 6
B = 8


def _policy(cascade1=1, cascade2=1, optimizer="ADAM"):
    return R_MAPPOPolicy(
        obs_space=spaces.Box(low=0, high=255, shape=(H, W, C), dtype=np.float32),
        cent_obs_space=spaces.Box(low=0, high=255, shape=(H, W, C), dtype=np.float32),
        act_space=spaces.Discrete(N_ACT),
        hidden_size=HIDDEN,
        recurrent_n=RECN,
        cascade_iterations_1=cascade1,
        cascade_iterations_2=cascade2,
        lr=7e-5,
        critic_lr=7e-5,
        opti_eps=1e-5,
        weight_decay=1e-5,
        optimizer=optimizer,
    )


def test_builds_actor_critic_no_ghosts():
    set_all_seeds(42)
    policy = _policy()
    assert isinstance(policy.actor, R_Actor)
    assert isinstance(policy.critic, R_Critic)
    # Fix (a): the judge lives INSIDE the acting actor.
    assert isinstance(policy.actor.second_order, SecondOrderNetwork)
    assert isinstance(policy.actor_optimizer, torch.optim.Adam)
    assert isinstance(policy.critic_optimizer, torch.optim.Adam)
    # Ghost meta networks / optimizers / read-out are gone.
    assert not hasattr(policy, "actor_meta")
    assert not hasattr(policy, "critic_meta")
    assert not hasattr(policy, "actor_meta_optimizer")
    assert not hasattr(policy, "evaluate_actions_meta")


def test_actor_optimizer_covers_second_order():
    """Co-training requires the judge to be trainable via ``actor_optimizer``."""
    set_all_seeds(0)
    policy = _policy()
    opt_param_ids = {id(p) for g in policy.actor_optimizer.param_groups for p in g["params"]}
    assert all(id(p) in opt_param_ids for p in policy.actor.second_order.parameters())


def test_optimizer_dispatch_adamax():
    set_all_seeds(1)
    policy = _policy(optimizer="ADAMAX")
    assert isinstance(policy.actor_optimizer, torch.optim.Adamax)


def test_evaluate_actions_returns_wager():
    set_all_seeds(2)
    policy = _policy()
    obs = (np.random.rand(B, H, W, C) * 255).astype(np.float32)
    rnn = np.zeros((B, RECN, HIDDEN), dtype=np.float32)
    masks = np.ones((B, 1), dtype=np.float32)
    action = np.random.randint(0, N_ACT, (B, 1)).astype(np.float32)
    values, log_probs, entropy, wager = policy.evaluate_actions(obs, obs, rnn, rnn, action, masks)
    assert values.shape == (B, 1)
    assert log_probs.shape == (B, 1)
    assert entropy.dim() == 0
    assert wager.shape == (B, 2)
    assert torch.isfinite(wager).all()


def test_get_actions_rollout_unchanged():
    """Monitoring only: rollout still uses the baseline actor (wager never gates actions)."""
    set_all_seeds(4)
    policy = _policy()
    obs = (np.random.rand(B, H, W, C) * 255).astype(np.float32)
    rnn = np.zeros((B, RECN, HIDDEN), dtype=np.float32)
    masks = np.ones((B, 1), dtype=np.float32)
    values, actions, _log_probs, _rnn_a, _rnn_c = policy.get_actions(obs, obs, rnn, rnn, masks)
    assert actions.shape == (B, 1)
    assert (actions >= 0).all() and (actions < N_ACT).all()
    assert values.shape == (B, 1)
