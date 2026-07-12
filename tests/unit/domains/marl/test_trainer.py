"""R_MAPPO after fix (a): device-aware ``ppo_update`` (runs on CPU); the confidence wager is
co-trained INTO the acting actor (BCE vs advantage>0). These tests LOCK the fix's key property:
the judge's gradient reaches the acting network — the whole point of the corrected MARL run.
"""

from __future__ import annotations

import numpy as np
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


def _sample():
    def arr(*shape):
        return np.random.rand(*shape).astype(np.float32)

    return (
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
        arr(B, 1) - 0.5,  # adv_targ (mix of + and -)
        np.ones((B, N_ACT), np.float32),  # available_actions
    )


def test_trainer_builds_with_valuenorm():
    set_all_seeds(42)
    trainer = _trainer()
    assert isinstance(trainer.value_normalizer, ValueNorm)
    assert trainer._use_recurrent_policy is True


def test_cal_value_loss_scalar():
    set_all_seeds(1)
    trainer = _trainer()
    loss = trainer.cal_value_loss(
        torch.randn(B, 1), torch.randn(B, 1), torch.randn(B, 1), torch.ones(B, 1)
    )
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_focal_loss_is_scalar():
    set_all_seeds(3)
    fl = FocalLoss()
    out = fl(torch.randn(B, 2), torch.randint(0, 2, (B, 2)).float())
    assert out.dim() == 0 and torch.isfinite(out)


def test_ppo_update_runs_on_cpu():
    """Fix (a): ``ppo_update`` is device-aware (no hardcoded .cuda()) and returns a 7-tuple."""
    set_all_seeds(4)
    trainer = _trainer(device="cpu")
    out = trainer.ppo_update(_sample(), update_actor=True, meta=True)
    assert len(out) == 7
    value_loss, _cgn, policy_loss, _de, _agn, imp_weights, wager_loss = out
    assert torch.isfinite(value_loss) and torch.isfinite(policy_loss)
    assert torch.isfinite(wager_loss) and wager_loss.item() > 0.0  # BCE against advantage>0
    assert imp_weights.shape == (B, 1)


def test_meta_cotraining_reaches_actor():
    """KEY LOCK (fix a): the wager gradient reaches BOTH the judge and the shared encoder."""
    set_all_seeds(5)
    trainer = _trainer(device="cpu")
    trainer.ppo_update(_sample(), update_actor=True, meta=True)
    # The judge is trainable ONLY through the wager term:
    judge_grad = trainer.policy.actor.second_order.wager.weight.grad
    assert judge_grad is not None and float(judge_grad.abs().sum()) > 0.0
    # The shared CNN encoder also carries gradient (co-training path exists):
    base_param = next(iter(trainer.policy.actor.base.parameters()))
    assert base_param.grad is not None


def test_baseline_has_no_wager_gradient():
    """meta=False: the judge receives no gradient (wager term absent) → wager_loss == 0."""
    set_all_seeds(6)
    trainer = _trainer(device="cpu")
    out = trainer.ppo_update(_sample(), update_actor=True, meta=False)
    assert float(out[6]) == 0.0
    assert trainer.policy.actor.second_order.wager.weight.grad is None


def test_wager_changes_acting_gradient():
    """GOLD LOCK: identical weights + inputs, meta=True vs meta=False produce DIFFERENT gradients
    on the acting encoder → the judge genuinely shapes the acting network (not a passive probe)."""
    set_all_seeds(7)
    t_off = _trainer(device="cpu")
    t_on = _trainer(device="cpu")
    t_on.policy.actor.load_state_dict(t_off.policy.actor.state_dict())
    t_on.policy.critic.load_state_dict(t_off.policy.critic.state_dict())
    sample = _sample()
    t_off.ppo_update(sample, update_actor=True, meta=False)
    t_on.ppo_update(sample, update_actor=True, meta=True)
    g_off = next(iter(t_off.policy.actor.base.parameters())).grad
    g_on = next(iter(t_on.policy.actor.base.parameters())).grad
    assert g_off is not None and g_on is not None
    assert not torch.allclose(g_off, g_on)


def test_prep_modes_toggle_without_ghost_nets():
    """Regression: prep_training/prep_rollout must not reference the removed ghost nets
    (the runner calls these every rollout/update — the unit tests now exercise them too)."""
    set_all_seeds(8)
    trainer = _trainer(device="cpu")
    trainer.prep_rollout()
    assert not trainer.policy.actor.training and not trainer.policy.critic.training
    trainer.prep_training()
    assert trainer.policy.actor.training and trainer.policy.critic.training
