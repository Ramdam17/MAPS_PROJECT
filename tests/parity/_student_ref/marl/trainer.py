"""Student ``R_MAPPO`` trainer — stripped to the baseline path only (E.15 ref).

Dropped vs ``external/paper_reference/marl_tmlr/onpolicy/algorithms/r_mappo/r_mappo.py`` :
- ``FocalLoss`` (dead — never called).
- Unconditional ``values_meta = self.policy.evaluate_actions_meta(...)`` +
  ``binary_cross_entropy_with_logits(values_meta, wager_objective)`` branch
  on the baseline (meta=False) path. Student runs the actor_meta forward
  even when wager is not used, which our port skips per E.5 scope lock.
  **Parity ref matches the port — this is a "port fidelity" parity test,
  not a carbon copy of the student's dead-code path.**
- ``_use_popart`` branch (``use_popart=False`` per E.5).
- ``evaluate_actions_meta`` + ``actor_meta_optimizer`` / ``critic_meta_optimizer``
  (meta=True path ; covered separately by our unit tests).

Ops otherwise verbatim from r_mappo.py L74-244 with ``meta=False`` fixed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .util import check


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e**2 / 2


def get_grad_norm(it):
    sum_grad = 0.0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return sum_grad**0.5


class R_MAPPO:
    """Stripped student R_MAPPO trainer — baseline PPO update only."""

    def __init__(self, args, policy, value_normalizer=None, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.clip_param = args.clip_param
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_max_grad_norm = True
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        self.value_normalizer = value_normalizer

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """Student L74-111 verbatim."""
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()
        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """Single baseline PPO mini-batch update — student L113-244, meta=False stripped."""
        (
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Forward : evaluate_actions on actor + critic.
        action_log_probs, dist_entropy = self.policy.actor.evaluate_actions(
            obs=obs_batch,
            rnn_states=rnn_states_batch,
            action=actions_batch,
            masks=masks_batch,
            available_actions=available_actions_batch,
            active_masks=active_masks_batch if self._use_policy_active_masks else None,
        )
        values, _ = self.policy.critic(share_obs_batch, rnn_states_critic_batch, masks_batch)

        # PPO policy loss.
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        # Actor update.
        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            total_loss = policy_loss - dist_entropy * self.entropy_coef
            total_loss.backward()
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()

        # Critic update — student re-evaluates values via critic (r_mappo.py L227).
        self.policy.critic_optimizer.zero_grad()
        values_fresh, _ = self.policy.critic(share_obs_batch, rnn_states_critic_batch, masks_batch)
        value_loss = self.cal_value_loss(
            values_fresh, value_preds_batch, return_batch, active_masks_batch
        )
        total_critic_loss = value_loss * self.value_loss_coef
        total_critic_loss.backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            imp_weights,
        )
