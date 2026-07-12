"""Tier-3: MARL MAPPO trainer (Sprint 16.G).

``cal_value_loss`` and construction run on CPU. ``ppo_update`` hardcodes ``.cuda()``
(faithful to the source), so its smoke test is gated on CUDA availability.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from maps.domains.marl.rmappo_policy import R_MAPPOPolicy
from maps.domains.marl.trainer import R_MAPPO, FocalLoss
from maps.domains.marl.valuenorm import ValueNorm
from maps.utils.seeding import set_all_seeds

H = W = 11
C = 3
HIDDEN = 64
RECN = 1
N_ACT = 6
B = 8


def _policy():
    return R_MAPPOPolicy(
        obs_space=spaces.Box(low=0, high=255, shape=(H, W, C), dtype=np.float32),
        cent_obs_space=spaces.Box(low=0, high=255, shape=(H, W, C), dtype=np.float32),
        act_space=spaces.Discrete(N_ACT),
        hidden_size=HIDDEN,
        recurrent_n=RECN,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        lr=7e-5,
        critic_lr=7e-5,
        opti_eps=1e-5,
        weight_decay=1e-5,
    )


def _trainer(device="cpu"):
    return R_MAPPO(_policy(), hidden_size=HIDDEN, device=device)


def test_trainer_builds_with_valuenorm():
    set_all_seeds(42)
    trainer = _trainer()
    assert isinstance(trainer.value_normalizer, ValueNorm)
    # Attention off → both recurrent flags forced True (faithful override).
    assert trainer._use_recurrent_policy is True
    assert trainer._use_naive_recurrent_policy is True


def test_cal_value_loss_scalar():
    set_all_seeds(1)
    trainer = _trainer()
    values = torch.randn(B, 1)
    value_preds = torch.randn(B, 1)
    returns = torch.randn(B, 1)
    active_masks = torch.ones(B, 1)
    loss = trainer.cal_value_loss(values, value_preds, returns, active_masks)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_cal_value_loss_respects_active_masks():
    """A fully-inactive batch would divide by zero; a partially-active one stays finite."""
    set_all_seeds(2)
    trainer = _trainer()
    values = torch.randn(B, 1)
    value_preds = torch.randn(B, 1)
    returns = torch.randn(B, 1)
    active_masks = torch.ones(B, 1)
    active_masks[B // 2 :] = 0.0
    loss = trainer.cal_value_loss(values, value_preds, returns, active_masks)
    assert torch.isfinite(loss)


def test_focal_loss_is_scalar():
    set_all_seeds(3)
    fl = FocalLoss()
    inputs = torch.randn(B, 2)
    targets = torch.randint(0, 2, (B, 2)).float()
    out = fl(inputs, targets)
    assert out.dim() == 0
    assert torch.isfinite(out)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ppo_update hardcodes .cuda() (source-faithful)"
)
def test_ppo_update_smoke_gpu():
    """Smoke test of one PPO update — GPU only (the source pins wager tensors to CUDA)."""
    set_all_seeds(4)
    trainer = _trainer(device="cuda")

    def arr(*shape):
        return (np.random.rand(*shape)).astype(np.float32)

    sample = (
        arr(B, H, W, C),  # share_obs
        arr(B, H, W, C),  # obs
        np.zeros((B, RECN, HIDDEN), np.float32),  # rnn_states
        np.zeros((B, RECN, HIDDEN), np.float32),  # rnn_states_critic
        np.random.randint(0, N_ACT, (B, 1)).astype(np.float32),  # actions
        arr(B, 1),  # value_preds
        arr(B, 1),  # returns
        np.ones((B, 1), np.float32),  # masks
        np.ones((B, 1), np.float32),  # active_masks
        arr(B, 1) - 0.5,  # old_action_log_probs
        arr(B, 1) - 0.5,  # adv_targ
        np.ones((B, N_ACT), np.float32),  # available_actions
    )
    wager_objective = np.random.randint(0, 2, (B, 2)).astype(np.float32)
    out = trainer.ppo_update(sample, update_actor=True, wager_objective=wager_objective, meta=True)
    value_loss, _critic_grad_norm, policy_loss, _dist_entropy, _actor_grad_norm, imp_weights = out
    assert torch.isfinite(value_loss)
    assert torch.isfinite(policy_loss)
    assert imp_weights.shape == (B, 1)
