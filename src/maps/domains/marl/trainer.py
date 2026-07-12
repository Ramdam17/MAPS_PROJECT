"""MARL MAPPO trainer — MAPS §5 (MeltingPot MAPPO).

Ported from ``onpolicy/algorithms/r_mappo/r_mappo.py`` (``R_MAPPO``), the GRU path.
Performs the PPO update over minibatches from the rollout buffer and supervises the
metacognitive **wager** with a BCE against ``reward>0`` (the ``wager_objective``).

Faithful properties of the paper's code (see :mod:`maps.domains.marl.rmappo_policy`):

- **M-C2 / M-C3**: both the actor-side and critic-side wager losses call
  ``policy.evaluate_actions_meta``, which reads the wager from ``actor_meta``. So
  ``critic_meta`` receives no gradients and its ``.step()`` is a no-op — reproduced.
- **Hardcoded ``.cuda()``**: the wager tensors are forced onto CUDA
  (``r_mappo.py:162-163,220``) *after* ``.to(**self.tpdv)``. On GPU (where MARL runs)
  this is a redundant no-op; on CPU it raises. We reproduce it **verbatim** — so
  :meth:`ppo_update` (and :meth:`train`) require a GPU. Everything else
  (:meth:`cal_value_loss`, construction) runs on CPU.
- **rnn_cells (LSTM) dropped**: the sample tuple is the 12-element GRU tuple our
  :class:`~maps.domains.marl.data.SeparatedReplayBuffer` yields (no ``rnn_cells``),
  consistent end-to-end (D16).
- ``naive_recurrent_generator`` / ``feed_forward_generator`` are not reached with the
  MeltingPot config (``use_recurrent_policy`` forced True when attention is off) and
  are not ported.

References
----------
Vargas et al. (2025), MAPS, TMLR submission §5, Table 7.
Yu et al. (2022). The surprising effectiveness of PPO in cooperative MARL (MAPPO).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from maps.domains.marl.util import check, get_grad_norm, huber_loss, mse_loss
from maps.domains.marl.valuenorm import ValueNorm

# Source-faithful default device (r_mappo.py:31 evaluates torch.device(...) once, at
# definition time); a module-level singleton keeps that semantics and is lint-clean.
_DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    """Verbatim ``FocalLoss`` (r_mappo.py:8-18). Defined in the source but not used by
    :meth:`R_MAPPO.ppo_update` (which uses ``binary_cross_entropy_with_logits``)."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return torch.mean(f_loss)


class R_MAPPO:  # noqa: N801 (source class name)
    """Trainer that updates the MAPPO policy. Port of R_MAPPO (GRU path)."""

    def __init__(
        self,
        policy,
        *,
        hidden_size: int,
        clip_param: float = 0.2,
        ppo_epoch: int = 15,
        num_mini_batch: int = 1,
        data_chunk_length: int = 10,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.01,
        huber_delta: float = 5.0,
        use_attention: bool = False,
        use_max_grad_norm: bool = True,
        use_clipped_value_loss: bool = True,
        use_huber_loss: bool = True,
        use_popart: bool = False,
        use_valuenorm: bool = True,
        use_value_active_masks: bool = True,
        use_policy_active_masks: bool = True,
        device: torch.device | str = _DEFAULT_DEVICE,
    ) -> None:
        self.device = torch.device(device)
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.policy = policy
        self.use_attention = use_attention
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.data_chunk_length = data_chunk_length
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.huber_delta = huber_delta

        self._use_max_grad_norm = use_max_grad_norm
        self._use_clipped_value_loss = use_clipped_value_loss
        self._use_huber_loss = use_huber_loss
        self._use_popart = use_popart
        self._use_valuenorm = use_valuenorm
        self._use_value_active_masks = use_value_active_masks
        self._use_policy_active_masks = use_policy_active_masks

        # Dead weight in the source (r_mappo.py:56) — never used in ppo_update.
        self.layer_output_2nd = nn.Linear(hidden_size, 1)

        # Attention off → the source forces both recurrent flags True (r_mappo.py:58-63).
        if self.use_attention:
            self._use_naive_recurrent_policy = False
            self._use_recurrent_policy = False
        else:
            self._use_naive_recurrent_policy = True
            self._use_recurrent_policy = True

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """Clipped value-function loss (Huber or MSE). Verbatim r_mappo.py:74-111."""
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if self._use_popart or self._use_valuenorm:
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

    def ppo_update(self, sample, update_actor=True, wager_objective=None, meta=False):
        """One PPO minibatch update (actor then critic). Verbatim r_mappo.py:113-244.

        ``sample`` is the 12-element GRU tuple from
        :meth:`~maps.domains.marl.data.SeparatedReplayBuffer.recurrent_generator`.
        Requires CUDA (the source hardcodes ``.cuda()`` on the wager tensors).
        """
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

        # Reshape to do in a single forward pass for all steps.
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        values_meta = self.policy.evaluate_actions_meta(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # ################## 2ND ORDER NETWORK (wager BCE) ##################
        wager_objective = (
            torch.tensor(wager_objective, dtype=torch.float32).unsqueeze(-1).unsqueeze(0)
        )
        values_meta = check(values_meta).to(**self.tpdv).squeeze(-1).squeeze(0).cuda()
        wager_objective = check(wager_objective).to(**self.tpdv).squeeze(-1).squeeze(0).cuda()
        loss_2 = torch.nn.functional.binary_cross_entropy_with_logits(values_meta, wager_objective)
        loss_2_values = loss_2 * self.value_loss_coef

        self.policy.actor_meta_optimizer.zero_grad()
        self.policy.actor_optimizer.zero_grad()

        # ################## 1ST ORDER NETWORK (actor update) ##################
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

        if update_actor:
            total_loss = policy_loss - dist_entropy * self.entropy_coef
            if meta:
                loss_2_values.backward(retain_graph=True)
                self.policy.actor_meta_optimizer.step()
            total_loss.backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()

        # ################## CRITIC UPDATE ##################
        # M-C3: evaluate_actions_meta reads actor_meta again → critic_meta gets no
        # gradients, so critic_meta_optimizer.step() is a no-op (faithful dead weight).
        values_meta_critic = self.policy.evaluate_actions_meta(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )
        values_meta_critic = check(values_meta_critic).to(**self.tpdv).squeeze(-1).squeeze(0).cuda()
        loss_2_critic = torch.nn.functional.binary_cross_entropy_with_logits(
            values_meta_critic, wager_objective
        )
        loss_2_values_critic = loss_2_critic * self.value_loss_coef

        self.policy.critic_meta_optimizer.zero_grad()
        self.policy.critic_optimizer.zero_grad()

        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch
        )
        total_critic_loss = value_loss * self.value_loss_coef

        if meta:
            loss_2_values_critic.backward(retain_graph=True)
            self.policy.critic_meta_optimizer.step()
        total_critic_loss.backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, wager_objective=None, update_actor=True, meta=False):
        """Run ``ppo_epoch`` passes of minibatch PPO. Verbatim r_mappo.py:247-304.

        Advantage computation/normalisation runs on CPU (numpy); :meth:`ppo_update`
        requires CUDA.
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(
                buffer.value_preds[:-1]
            )
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {
            "value_loss": 0,
            "policy_loss": 0,
            "dist_entropy": 0,
            "actor_grad_norm": 0,
            "critic_grad_norm": 0,
            "ratio": 0,
        }

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy or self.use_attention:
                data_generator = buffer.recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length
                )
            else:  # naive_recurrent / feed_forward not reached with the MeltingPot config.
                raise NotImplementedError(
                    "only recurrent_generator is ported (GRU path); "
                    "naive/feed-forward generators are unused by the MeltingPot runner"
                )

            for sample in data_generator:
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                ) = self.ppo_update(sample, update_actor, wager_objective, meta)

                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["critic_grad_norm"] += critic_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info:
            train_info[k] /= num_updates

        return train_info

    def prep_training(self) -> None:
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.actor_meta.train()
        self.policy.critic_meta.train()

    def prep_rollout(self) -> None:
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.actor_meta.eval()
        self.policy.critic_meta.eval()
