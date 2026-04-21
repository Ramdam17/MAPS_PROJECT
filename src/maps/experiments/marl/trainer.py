"""MAPPO trainer with MAPS wager integration (E.9a).

Ports student ``r_mappo/r_mappo.py`` (316 L). Core responsibilities :

1. **``cal_value_loss``** — value loss with PPO clip + optional Huber and
   optional ValueNorm. Student L74-111.
2. **``ppo_update``** — one PPO mini-batch step : clip policy loss, value
   loss, entropy bonus, + optional MAPS wager loss (paper eq.5,
   ``binary_cross_entropy_with_logits`` on ``values_meta`` vs
   ``wager_objective``). Student L113-244.
3. **``train``** — outer loop : compute GAE advantages, iterate ``ppo_epoch``
   times over mini-batches, aggregate train infos. Student L247-304.
4. **``prep_training`` / ``prep_rollout``** — set modules to train/eval mode.

Dropped from student (E.5 scope) :
- ``FocalLoss`` class (L8-18) : defined but never called (dead code).
- ``PopArt`` integration : `use_popart=False` in our config by default.

Note : student's ppo_update calls ``self.policy.evaluate_actions_meta`` twice
(once for actor wager, once for critic wager) — but the wager signal is
identical (same (obs, rnn_states, masks)). Port factors this to ONE call.

References
----------
- Paper §2.2 eq.5 + standard PPO (Yu et al. 2022).
- ``docs/reviews/marl-architecture.md §(g)`` + ``marl-maps-additions.md §(g)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from maps.experiments.marl.policy import MAPPOPolicy
from maps.experiments.marl.util import check, get_grad_norm, huber_loss, mse_loss
from maps.experiments.marl.valuenorm import ValueNorm

__all__ = ["MAPPOTrainer", "TrainInfo"]

log = logging.getLogger(__name__)


@dataclass
class TrainInfo:
    """Per-PPO-update metrics, averaged across mini-batches (student L273-297)."""

    value_loss: float = 0.0
    policy_loss: float = 0.0
    dist_entropy: float = 0.0
    actor_grad_norm: float = 0.0
    critic_grad_norm: float = 0.0
    ratio: float = 0.0
    wager_loss_actor: float = 0.0
    wager_loss_critic: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


class MAPPOTrainer:
    """MAPPO trainer with optional MAPS wager loss.

    Parameters
    ----------
    cfg : DictConfig
        Composed config (typically ``load_config('training/marl')``).
    policy : MAPPOPolicy
        The 4-network wrapper (actor, critic, + optional actor_meta, critic_meta).
    device : torch.device | str

    Usage
    -----
    ```python
    trainer = MAPPOTrainer(cfg, policy, device="cuda")
    trainer.prep_training()
    info = trainer.train(buffer, wager_objective=ema_signal, meta=setting.meta)
    trainer.prep_rollout()
    ```
    """

    def __init__(self, cfg: DictConfig, policy: MAPPOPolicy, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.policy = policy
        self.device = torch.device(device)
        self.tpdv = dict(dtype=torch.float32, device=self.device)

        ppo_cfg = cfg.ppo
        self.clip_param = float(ppo_cfg.clip_param)
        self.ppo_epoch = int(ppo_cfg.ppo_epoch)
        self.num_mini_batch = int(ppo_cfg.num_mini_batch)
        self.data_chunk_length = int(ppo_cfg.data_chunk_length)
        self.value_loss_coef = float(ppo_cfg.value_loss_coef)
        self.entropy_coef = float(ppo_cfg.entropy_coef)
        self.max_grad_norm = float(ppo_cfg.max_grad_norm)
        self.huber_delta = float(ppo_cfg.huber_delta)

        self._use_max_grad_norm = True  # student default
        self._use_clipped_value_loss = bool(ppo_cfg.use_clipped_value_loss)
        self._use_huber_loss = bool(ppo_cfg.use_huber_loss)
        self._use_popart = bool(ppo_cfg.use_popart)
        self._use_valuenorm = bool(ppo_cfg.use_valuenorm) and not self._use_popart
        self._use_value_active_masks = bool(ppo_cfg.use_value_active_masks)
        self._use_policy_active_masks = bool(ppo_cfg.use_policy_active_masks)

        if self._use_popart:
            # Not supported in port scope ; student wires self.value_normalizer =
            # self.policy.critic.v_out. We explicitly raise (E.5 scope).
            raise NotImplementedError(
                "use_popart=True is OMITTED per E.5 scope lock (docs/reviews/marl-scope-decisions.md §c)."
            )

        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None

    # ──────────────────────────────────────────────────────────────
    # Value loss (student L74-111)
    # ──────────────────────────────────────────────────────────────

    def cal_value_loss(
        self,
        values: torch.Tensor,
        value_preds_batch: torch.Tensor,
        return_batch: torch.Tensor,
        active_masks_batch: torch.Tensor,
    ) -> torch.Tensor:
        """PPO clipped value loss with optional Huber + optional ValueNorm."""
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

    # ──────────────────────────────────────────────────────────────
    # PPO mini-batch update (student L113-244, simplified)
    # ──────────────────────────────────────────────────────────────

    def ppo_update(
        self,
        sample: tuple,
        update_actor: bool = True,
        wager_objective: torch.Tensor | np.ndarray | None = None,
        meta: bool = False,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        """Single mini-batch PPO update.

        Parameters
        ----------
        sample : tuple
            ``(share_obs, obs, rnn_states, rnn_states_critic, actions,
               value_preds, returns, masks, active_masks,
               old_action_log_probs, advantages, available_actions)``.
            This is 12 fields ; student uses 14 (with rnn_cells_actor/critic
            for attention paths we dropped per E.5).
        update_actor : bool
            Whether to step the actor optimizer.
        wager_objective : Tensor | ndarray, optional
            EMA-based wager signal (paper eq.14). Required when ``meta=True``.
            Expected shape broadcasts against ``values_meta`` (B, 2).
        meta : bool
            If True, compute wager loss (paper eq.5) and step actor_meta /
            critic_meta optimizers.

        Returns
        -------
        (value_loss, critic_grad_norm, policy_loss, dist_entropy,
         actor_grad_norm, imp_weights_mean, wager_loss_actor, wager_loss_critic)
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

        # ─ Baseline forward : evaluate actions + values through actor/critic.
        action_log_probs, dist_entropy = self.policy.actor.evaluate_actions(
            obs=obs_batch,
            rnn_states=rnn_states_batch,
            action=actions_batch,
            masks=masks_batch,
            available_actions=available_actions_batch,
            active_masks=active_masks_batch if self._use_policy_active_masks else None,
        )
        values, _ = self.policy.critic(share_obs_batch, rnn_states_critic_batch, masks_batch)

        # ─ Meta wager — actor side forward only (student L149-166).
        # Critic wager is re-forwarded AFTER actor.step() to avoid stale graphs.
        wager_loss_actor = torch.zeros((), device=self.device)
        wager_loss_critic = torch.zeros((), device=self.device)
        if meta:
            if wager_objective is None:
                raise ValueError("meta=True requires wager_objective to be provided.")
            if self.policy.actor_meta is None or self.policy.critic_meta is None:
                raise RuntimeError(
                    "meta=True but policy.actor_meta/critic_meta is None. "
                    "Build MAPPOPolicy(..., meta=True)."
                )
            wager_objective_t = check(wager_objective).to(**self.tpdv)

            values_meta_actor = self.policy.actor_meta.evaluate_actions(
                obs=obs_batch,
                rnn_states=rnn_states_batch,
                action=actions_batch,
                masks=masks_batch,
            )
            wager_loss_actor = F.binary_cross_entropy_with_logits(values_meta_actor, wager_objective_t)

        # ─ PPO policy loss (student L174-189).
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

        # ─ Actor update (student L193-206).
        self.policy.optimizers.actor.zero_grad()
        if meta and self.policy.optimizers.actor_meta is not None:
            self.policy.optimizers.actor_meta.zero_grad()

        if update_actor:
            if meta:
                # Wager loss back-propagates into actor_meta.
                (wager_loss_actor * self.value_loss_coef).backward(retain_graph=True)
                self.policy.optimizers.actor_meta.step()

            total_loss = policy_loss - dist_entropy * self.entropy_coef
            total_loss.backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm
            ).item()
        else:
            actor_grad_norm = get_grad_norm(self.policy.actor.parameters())
        self.policy.optimizers.actor.step()

        # ─ Critic update (student L211-242).
        self.policy.optimizers.critic.zero_grad()
        if meta and self.policy.optimizers.critic_meta is not None:
            self.policy.optimizers.critic_meta.zero_grad()

        # Re-forward meta for critic-side wager (student L211-218) — this creates
        # a FRESH graph after actor_meta.step() invalidated the previous one.
        if meta:
            assert wager_objective is not None
            wager_objective_t = check(wager_objective).to(**self.tpdv)
            values_meta_critic = self.policy.actor_meta.evaluate_actions(
                obs=share_obs_batch,
                rnn_states=rnn_states_critic_batch,
                action=actions_batch,
                masks=masks_batch,
            )
            wager_loss_critic = F.binary_cross_entropy_with_logits(values_meta_critic, wager_objective_t)

        # Compute values AGAIN for value loss — student L227 re-evaluates via critic.
        values_fresh, _ = self.policy.critic(share_obs_batch, rnn_states_critic_batch, masks_batch)
        value_loss = self.cal_value_loss(values_fresh, value_preds_batch, return_batch, active_masks_batch)
        total_critic_loss = value_loss * self.value_loss_coef

        if meta:
            (wager_loss_critic * self.value_loss_coef).backward(retain_graph=True)
            self.policy.optimizers.critic_meta.step()

        total_critic_loss.backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm
            ).item()
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())
        self.policy.optimizers.critic.step()

        return (
            value_loss.item(),
            float(critic_grad_norm),
            policy_loss.item(),
            dist_entropy.item(),
            float(actor_grad_norm),
            float(imp_weights.mean().item()),
            float(wager_loss_actor.item()),
            float(wager_loss_critic.item()),
        )

    # ──────────────────────────────────────────────────────────────
    # Outer train loop (student L247-304)
    # ──────────────────────────────────────────────────────────────

    def train(
        self,
        buffer,
        wager_objective: torch.Tensor | np.ndarray | None = None,
        update_actor: bool = True,
        meta: bool = False,
    ) -> dict[str, float]:
        """Perform a training update using minibatch GD.

        Parameters
        ----------
        buffer : object with attributes ``returns``, ``value_preds``, ``active_masks``,
            and methods ``recurrent_generator`` / ``feed_forward_generator``.
            Matches student's ``SeparatedReplayBuffer`` / our port's rollout buffer (E.9b).
        wager_objective : optional
            EMA signal per-agent. Required when ``meta=True``.
        update_actor : bool
            Whether to step actor optimizer(s).
        meta : bool
            If True, also supervise actor_meta / critic_meta.

        Returns
        -------
        dict[str, float]
            Averaged per-metric train info (value_loss, policy_loss,
            dist_entropy, actor_grad_norm, critic_grad_norm, ratio,
            wager_loss_actor, wager_loss_critic).
        """
        if self.value_normalizer is not None:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(
                buffer.value_preds[:-1]
            )
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_adv = np.nanmean(advantages_copy)
        std_adv = np.nanstd(advantages_copy)
        advantages = (advantages - mean_adv) / (std_adv + 1e-5)

        info = TrainInfo()
        n_updates = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.recurrent_generator(
                advantages, self.num_mini_batch, self.data_chunk_length
            )

            for sample in data_generator:
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    ratio,
                    wl_actor,
                    wl_critic,
                ) = self.ppo_update(sample, update_actor, wager_objective, meta)

                info.value_loss += value_loss
                info.policy_loss += policy_loss
                info.dist_entropy += dist_entropy
                info.actor_grad_norm += actor_grad_norm
                info.critic_grad_norm += critic_grad_norm
                info.ratio += ratio
                info.wager_loss_actor += wl_actor
                info.wager_loss_critic += wl_critic
                n_updates += 1

        # Average over the total number of mini-batch updates performed.
        if n_updates > 0:
            for k, v in info.__dict__.items():
                info.__dict__[k] = v / n_updates

        return info.as_dict()

    # ──────────────────────────────────────────────────────────────
    # Mode switches (student L306-316)
    # ──────────────────────────────────────────────────────────────

    def prep_training(self) -> None:
        self.policy.actor.train()
        self.policy.critic.train()
        if self.policy.actor_meta is not None:
            self.policy.actor_meta.train()
        if self.policy.critic_meta is not None:
            self.policy.critic_meta.train()

    def prep_rollout(self) -> None:
        self.policy.actor.eval()
        self.policy.critic.eval()
        if self.policy.actor_meta is not None:
            self.policy.actor_meta.eval()
        if self.policy.critic_meta is not None:
            self.policy.critic_meta.eval()
