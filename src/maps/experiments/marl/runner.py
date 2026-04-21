"""MARL rollout + training orchestrator (E.9b).

Ports student ``onpolicy/runner/separated/meltingpot_runner.py`` trimmed to
the essential MAPPO + MAPS training loop. Student's 808-line runner contains
a lot of MeltingPot-specific env handling (action-env dict flattening, dead
get_episode_parameters helpers, etc.) which we simplify here — the real
MeltingPot env wrapper is E.10's scope.

The Runner operates in **separated MAPPO** mode : one
(:class:`MAPPOPolicy`, :class:`MAPPOTrainer`, :class:`RolloutBuffer`) tuple
per agent. Agents share the env but train independently.

EMA wager (paper eq.13-14) is implemented per our port's config :
- ``alpha = 0.45`` (paper ; student uses 0.25 — D-marl-ema-alpha fix).
- ``y_wager = (1, 0) if r_t > EMA_t else (0, 1)`` (paper eq.14 ; student
  uses ``grad_rewards > 0`` — D-marl-wager-condition fix).

Env interface contract (E.10 will wire MeltingPotEnv to this) :
- ``env.reset() → (obs_dict, info)``
- ``env.step(action_dict) → (obs_dict, reward_dict, done_dict, info)``
- ``env.observation_space`` (Dict with "player_i" keys)
- ``env.share_observation_space`` (Dict with "player_i" keys)
- ``env.action_space`` (Dict with "player_i" keys, each Discrete)
- ``env.close()``
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from maps.experiments.marl.data import RolloutBuffer
from maps.experiments.marl.policy import MAPPOPolicy
from maps.experiments.marl.setting import MarlSetting
from maps.experiments.marl.trainer import MAPPOTrainer
from maps.experiments.marl.valuenorm import ValueNorm

__all__ = ["MeltingpotRunner", "RunnerConfig", "compute_wager_objective"]

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# E.16 diagnostic helpers (remove once smoke is green).
# ──────────────────────────────────────────────────────────────


def _buffer_finite_check(agent_id: int, buf, stage: str) -> None:
    """Fail fast if any of the buffer's training-relevant arrays went non-finite."""
    for name in ("rewards", "value_preds", "returns", "masks", "active_masks"):
        arr = getattr(buf, name, None)
        if arr is None:
            continue
        if not np.isfinite(arr).all():
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            raise RuntimeError(
                f"[NaN-guard] buffer.{name} not finite at agent_id={agent_id} "
                f"stage={stage} : NaN={n_nan}, Inf={n_inf}, shape={arr.shape}, "
                f"min={arr.min():.3e}, max={arr.max():.3e}"
            )


def _next_values_finite_check(agent_id: int, next_values: np.ndarray) -> None:
    if not np.isfinite(next_values).all():
        n_nan = int(np.isnan(next_values).sum())
        raise RuntimeError(
            f"[NaN-guard] critic next_values not finite at agent_id={agent_id} : "
            f"NaN={n_nan}/{next_values.size}, shape={next_values.shape}"
        )


def _valuenorm_finite_check(agent_id: int, vn, stage: str) -> None:
    if vn is None:
        return
    for name in ("running_mean", "running_mean_sq", "debiasing_term"):
        p = getattr(vn, name)
        if not torch.isfinite(p).all():
            raise RuntimeError(
                f"[NaN-guard] value_normalizer.{name} not finite at agent_id={agent_id} "
                f"stage={stage} : value={p.tolist() if p.numel() <= 5 else p}"
            )


@dataclass
class RunnerConfig:
    """Runtime config bundle passed to :class:`MeltingpotRunner`.

    Keeping this as a dataclass (rather than argparse Namespace like student)
    makes the runner interface testable + type-checkable.
    """

    cfg: DictConfig  # the composed training/marl.yaml config
    setting: MarlSetting
    num_agents: int
    obs_shape: tuple[int, ...]
    share_obs_shape: tuple[int, ...]
    action_space: object  # gymnasium.spaces.Discrete
    device: torch.device | str = "cpu"


def compute_wager_objective(
    reward_t: np.ndarray,
    ema_prev: np.ndarray,
    alpha: float,
    condition: str = "r_t_gt_ema",
) -> tuple[np.ndarray, np.ndarray]:
    """Paper eq.13-14 EMA wager signal.

    Parameters
    ----------
    reward_t : shape (num_agents,)
    ema_prev : shape (num_agents,)
    alpha : float
        Paper eq.13 = 0.45 (port default). Student uses 0.25 (D-marl-ema-alpha).
    condition : "r_t_gt_ema" | "ema_gt_zero"
        Paper eq.14 = "r_t_gt_ema" ; student = "ema_gt_zero" (D-marl-wager-condition).

    Returns
    -------
    (ema_new, wager_target)
        ``ema_new`` shape (num_agents,) ; ``wager_target`` shape (num_agents, 2)
        one-hot per paper eq.14.
    """
    ema_new = alpha * reward_t + (1.0 - alpha) * ema_prev

    if condition == "r_t_gt_ema":
        is_high = reward_t > ema_new
    elif condition == "ema_gt_zero":
        is_high = ema_new > 0
    else:
        raise ValueError(f"Unknown wager condition {condition!r}")

    # One-hot (high_idx=0, low_idx=1) per paper eq.14.
    wager_target = np.zeros((reward_t.shape[0], 2), dtype=np.float32)
    wager_target[is_high, 0] = 1.0
    wager_target[~is_high, 1] = 1.0
    return ema_new, wager_target


class MeltingpotRunner:
    """Separated-MAPPO rollout + train orchestrator.

    Parameters
    ----------
    runner_cfg : RunnerConfig
        Bundle of dataclass + OmegaConf config.
    env : object
        Multi-agent env following the contract described in the module docstring.

    Usage
    -----
    ```python
    runner = MeltingpotRunner(runner_cfg, env)
    runner.run(num_episodes=N)
    ```
    """

    def __init__(self, runner_cfg: RunnerConfig, env: object):
        self.runner_cfg = runner_cfg
        self.cfg = runner_cfg.cfg
        self.setting = runner_cfg.setting
        self.env = env
        self.num_agents = int(runner_cfg.num_agents)
        self.device = torch.device(runner_cfg.device)

        # Training knobs (from cfg.training + cfg.maps).
        self.episode_length = int(self.cfg.training.episode_length)
        self.n_rollout_threads = int(self.cfg.training.n_rollout_threads)
        self.num_env_steps = int(self.cfg.training.num_env_steps)
        self.log_interval = int(self.cfg.training.log_interval)
        self.save_interval = int(self.cfg.training.save_interval)
        self.gamma = float(self.cfg.get("gamma", 0.99))
        self.gae_lambda = float(self.cfg.get("gae_lambda", 0.95))
        self.ema_alpha = float(self.cfg.maps.ema_alpha)
        self.wager_condition = str(self.cfg.maps.wager_condition)

        # Per-agent policy + trainer + buffer.
        self.policies: list[MAPPOPolicy] = []
        self.trainers: list[MAPPOTrainer] = []
        self.buffers: list[RolloutBuffer] = []
        for _ in range(self.num_agents):
            policy = MAPPOPolicy(
                self.cfg,
                obs_shape=runner_cfg.obs_shape,
                cent_obs_shape=runner_cfg.share_obs_shape,
                action_space=runner_cfg.action_space,
                meta=self.setting.meta,
                cascade_iterations1=self.setting.cascade_iterations1,
                cascade_iterations2=self.setting.cascade_iterations2,
                device=self.device,
            )
            trainer = MAPPOTrainer(self.cfg, policy, device=self.device)
            buffer = RolloutBuffer(
                episode_length=self.episode_length,
                n_rollout_threads=self.n_rollout_threads,
                hidden_size=int(self.cfg.model.hidden_size),
                recurrent_n=int(self.cfg.model.recurrent_n),
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                obs_space=self._make_obs_box(runner_cfg.obs_shape),
                share_obs_space=self._make_obs_box(runner_cfg.share_obs_shape),
                act_space=runner_cfg.action_space,
                use_valuenorm=trainer.value_normalizer is not None,
            )
            self.policies.append(policy)
            self.trainers.append(trainer)
            self.buffers.append(buffer)

        # EMA state for wager signal (per agent, per rollout thread).
        self.ema_reward = np.zeros((self.num_agents, self.n_rollout_threads), dtype=np.float32)

    @staticmethod
    def _make_obs_box(shape: tuple[int, ...]):
        """Build a dummy Box space for the buffer constructor."""
        from gymnasium import spaces

        return spaces.Box(low=0, high=255, shape=shape, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────
    # Rollout / buffer / train primitives
    # ──────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Reset the env and seed each agent's buffer at step 0."""
        obs_dict, _info = self.env.reset()
        for agent_id in range(self.num_agents):
            player_key = f"player_{agent_id}"
            obs = obs_dict[player_key]["RGB"]  # (n_rollout_threads, H, W, C) per env contract
            share_obs = obs_dict[player_key]["WORLD.RGB"]
            # Buffer expects (N, ...), not (T+1, N, ...) — so we index [0].
            self.buffers[agent_id].obs[0] = np.asarray(obs)
            self.buffers[agent_id].share_obs[0] = np.asarray(share_obs)

    @torch.no_grad()
    def collect(self, step: int) -> list[dict]:
        """Roll out one env step per agent. Returns per-agent dict of actions + RNN state."""
        results: list[dict] = []
        for agent_id in range(self.num_agents):
            policy = self.policies[agent_id]
            buf = self.buffers[agent_id]
            self.trainers[agent_id].prep_rollout()

            obs = torch.from_numpy(buf.obs[step]).float().to(self.device)
            share_obs = torch.from_numpy(buf.share_obs[step]).float().to(self.device)
            rnn_states = torch.from_numpy(buf.rnn_states[step]).float().to(self.device)
            rnn_states_critic = torch.from_numpy(buf.rnn_states_critic[step]).float().to(self.device)
            masks = torch.from_numpy(buf.masks[step]).float().to(self.device)

            actions, action_log_probs, new_rnn_states = policy.actor(
                obs=obs, rnn_states=rnn_states, masks=masks
            )
            values, new_rnn_states_critic = policy.critic(
                cent_obs=share_obs, rnn_states=rnn_states_critic, masks=masks
            )

            # E.16 diagnostic : catch rollout-level NaN before it poisons the
            # buffer. Helps localise whether blow-up is rollout-side or train-side.
            for name, t in (
                ("rollout actions", actions),
                ("rollout action_log_probs", action_log_probs),
                ("rollout values", values),
                ("rollout rnn_states", new_rnn_states),
                ("rollout rnn_states_critic", new_rnn_states_critic),
            ):
                if not torch.isfinite(t).all():
                    n_nan = int(torch.isnan(t).sum().item())
                    raise RuntimeError(
                        f"[NaN-guard] {name} not finite at step={step} agent_id={agent_id} "
                        f"(NaN={n_nan}/{t.numel()}, shape={tuple(t.shape)})"
                    )

            # ACTLayer returns log_prob with shape (N,) ; buffer expects (N, 1).
            alp_np = action_log_probs.detach().cpu().numpy()
            if alp_np.ndim == 1:
                alp_np = alp_np[:, None]
            results.append(
                {
                    "actions": actions.cpu().numpy(),
                    "action_log_probs": alp_np,
                    "values": values.cpu().numpy(),
                    "rnn_states": new_rnn_states.cpu().numpy(),
                    "rnn_states_critic": new_rnn_states_critic.cpu().numpy(),
                }
            )
        return results

    def insert(self, step: int, rollout: list[dict], env_step_result: tuple) -> np.ndarray:
        """Insert rollout into each agent's buffer. Returns per-agent reward vector.

        ``env_step_result`` is the tuple ``(obs_dict, reward_dict, done_dict, info)``
        from ``env.step(actions_dict)``.
        """
        obs_dict, reward_dict, done_dict, _info = env_step_result
        rewards_per_agent = np.zeros((self.num_agents, self.n_rollout_threads), dtype=np.float32)

        for agent_id in range(self.num_agents):
            player_key = f"player_{agent_id}"
            buf = self.buffers[agent_id]
            rdata = rollout[agent_id]

            # Extract per-agent env outputs.
            next_obs = np.asarray(obs_dict[player_key]["RGB"])
            next_share_obs = np.asarray(obs_dict[player_key]["WORLD.RGB"])
            reward = np.asarray(reward_dict[player_key])
            done = np.asarray(done_dict[player_key])

            # Reshape reward → (n_rollout_threads, 1) for buffer.
            reward_col = reward.reshape(self.n_rollout_threads, 1).astype(np.float32)
            if not np.isfinite(reward_col).all():
                raise RuntimeError(
                    f"[NaN-guard] env-side reward not finite at step={step} "
                    f"agent_id={agent_id} : {reward_col}"
                )
            rewards_per_agent[agent_id] = reward.reshape(-1)

            # Masks : 0 if done, 1 otherwise (paper eq.6 convention).
            mask = (~done.astype(bool)).astype(np.float32).reshape(self.n_rollout_threads, 1)
            active_mask = mask.copy()

            buf.insert(
                share_obs=next_share_obs,
                obs=next_obs,
                rnn_states=rdata["rnn_states"],
                rnn_states_critic=rdata["rnn_states_critic"],
                actions=rdata["actions"],
                action_log_probs=rdata["action_log_probs"],
                value_preds=rdata["values"],
                rewards=reward_col,
                masks=mask,
                active_masks=active_mask,
            )
        return rewards_per_agent

    @torch.no_grad()
    def compute(self) -> None:
        """Compute GAE returns for every agent's buffer."""
        for agent_id in range(self.num_agents):
            policy = self.policies[agent_id]
            buf = self.buffers[agent_id]
            self.trainers[agent_id].prep_rollout()

            share_obs = torch.from_numpy(buf.share_obs[-1]).float().to(self.device)
            rnn_states_critic = (
                torch.from_numpy(buf.rnn_states_critic[-1]).float().to(self.device)
            )
            masks = torch.from_numpy(buf.masks[-1]).float().to(self.device)

            next_values, _ = policy.critic(share_obs, rnn_states_critic, masks)
            next_values_np = next_values.cpu().numpy()

            # E.16 diagnostic : trace state of buffer + valuenorm + next_values
            # entering GAE so we can locate where NaN first appears.
            _buffer_finite_check(agent_id, buf, "pre-GAE")
            _next_values_finite_check(agent_id, next_values_np)
            _valuenorm_finite_check(agent_id, self.trainers[agent_id].value_normalizer, "pre-GAE")

            buf.compute_returns(next_values_np, self.trainers[agent_id].value_normalizer)

            _buffer_finite_check(agent_id, buf, "post-GAE")

    def train_agents(self, wager_targets_per_agent: np.ndarray | None) -> list[dict]:
        """Run one PPO update per agent. Returns a list of train_info dicts.

        ``wager_targets_per_agent`` is a (num_agents, ?, 2) array when meta=True, else None.
        """
        infos = []
        for agent_id in range(self.num_agents):
            self.trainers[agent_id].prep_training()
            wager = wager_targets_per_agent[agent_id] if wager_targets_per_agent is not None else None
            try:
                info = self.trainers[agent_id].train(
                    self.buffers[agent_id],
                    wager_objective=wager,
                    update_actor=True,
                    meta=self.setting.meta,
                )
            except RuntimeError as e:
                raise RuntimeError(f"train_agents failed at agent_id={agent_id} : {e}") from e
            infos.append(info)
            self.buffers[agent_id].after_update()
        return infos

    # ──────────────────────────────────────────────────────────────
    # Outer loop (simplified from student L117-295)
    # ──────────────────────────────────────────────────────────────

    def run(
        self,
        num_episodes: int | None = None,
        checkpoint_path: str | Path | None = None,
        resume_from: str | Path | None = None,
    ) -> list[dict]:
        """Main rollout + train loop.

        Parameters
        ----------
        num_episodes : int, optional
            Override the default (``num_env_steps // episode_length``).
        checkpoint_path : str or Path, optional
            If set, save the runner state at this path every
            ``cfg.training.save_interval`` episodes (and on completion).
            Atomic write via tmp file + rename.
        resume_from : str or Path, optional
            Load runner state from this checkpoint file and resume from
            the next episode. Replaces fresh init — every module's state
            (weights, optimizers, ValueNorm, RNG states, EMA, training
            history) is restored.

        Returns
        -------
        list of dict
            Per-episode aggregated train info. When resuming, infos
            from the prior run are prepended.
        """
        if num_episodes is None:
            num_episodes = max(1, self.num_env_steps // (self.episode_length * self.n_rollout_threads))

        start_episode = 0
        all_infos: list[dict] = []
        if resume_from is not None:
            start_episode, all_infos = self.load_checkpoint(resume_from)
            log.info(
                "resumed from %s at episode %d/%d (%d infos carried over)",
                resume_from,
                start_episode,
                num_episodes,
                len(all_infos),
            )

        start = time.time()
        for episode in range(start_episode, num_episodes):
            # Reset env + seed buffer step 0 at the start of each rollout.
            # Without this, ``num_cycles`` in the MeltingPotEnv truncation
            # counter accumulates across episodes — by ep2 ``num_cycles >=
            # max_cycles`` always holds, env returns done=True for every step,
            # active_masks fill with zeros and np.nanmean of all-NaN masks
            # cascades NaN through advantages. See E.16 diagnosis notes.
            self.warmup()

            # EMA per-agent reset at episode start (student L161).
            # Actually student keeps grad_rewards across episodes ; we preserve that.
            # Just capture per-step wagers to feed train_agents.
            episode_wagers: list[np.ndarray] = []  # len = episode_length, each (num_agents, 2)

            for step in range(self.episode_length):
                rollout = self.collect(step)

                # Stack per-agent actions into env's expected dict.
                # Env contract : action_dict[player_i] = actions.shape = (n_rollout_threads, 1)
                action_dict = {
                    f"player_{aid}": rollout[aid]["actions"] for aid in range(self.num_agents)
                }

                env_step_result = self.env.step(action_dict)
                rewards_per_agent = self.insert(step, rollout, env_step_result)

                # Update EMA + compute wager target (paper eq.13-14).
                if self.setting.meta:
                    # Reduce reward over rollout threads (mean) for EMA signal.
                    reward_t = rewards_per_agent.mean(axis=1)
                    ema_prev = self.ema_reward.mean(axis=1)
                    ema_new, wager_t = compute_wager_objective(
                        reward_t, ema_prev, alpha=self.ema_alpha, condition=self.wager_condition
                    )
                    self.ema_reward = alpha_broadcast(self.ema_reward, rewards_per_agent, self.ema_alpha)
                    episode_wagers.append(wager_t)

            # GAE returns
            self.compute()

            # Aggregate wager targets per-agent across steps for train call.
            wager_per_agent: np.ndarray | None = None
            if self.setting.meta and episode_wagers:
                # shape : (episode_length, num_agents, 2) → (num_agents, episode_length * batch, 2)
                stacked = np.stack(episode_wagers, axis=0)  # (T, A, 2)
                # Broadcast to match buffer batch_size = T * n_rollout_threads.
                # Each agent's trainer expects (B, 2) where B = T * n_rollout_threads.
                A = self.num_agents
                T = self.episode_length
                N = self.n_rollout_threads
                # Broadcast wager target across rollout threads (assumes same EMA across threads).
                wager_per_agent = np.broadcast_to(
                    stacked[:, :, None, :], (T, A, N, 2)
                ).transpose(1, 0, 2, 3).reshape(A, T * N, 2).astype(np.float32)

            # Train
            infos = self.train_agents(wager_per_agent)

            total_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            if episode % self.log_interval == 0:
                mean_value_loss = np.mean([i["value_loss"] for i in infos])
                mean_policy_loss = np.mean([i["policy_loss"] for i in infos])
                elapsed = time.time() - start
                log.info(
                    "episode %d/%d (steps=%d) | mean value=%.4f policy=%.4f | elapsed=%.1fs",
                    episode + 1,
                    num_episodes,
                    total_steps,
                    float(mean_value_loss),
                    float(mean_policy_loss),
                    elapsed,
                )

            all_infos.append(
                {
                    "episode": episode,
                    "per_agent": infos,
                    "total_steps": total_steps,
                }
            )

            if (
                checkpoint_path is not None
                and self.save_interval > 0
                and (episode + 1) % self.save_interval == 0
            ):
                self.save_checkpoint(checkpoint_path, next_episode=episode + 1, all_infos=all_infos)

        # Final checkpoint save (even if we didn't land on save_interval).
        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path, next_episode=num_episodes, all_infos=all_infos)

        return all_infos

    # ──────────────────────────────────────────────────────────────
    # Checkpoint save / load (E.17a)
    # ──────────────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        path: str | Path,
        next_episode: int,
        all_infos: list[dict],
    ) -> None:
        """Atomically serialize the runner state to ``path`` (tmp + rename).

        Contents : per-agent actor/critic/optional-meta state_dicts +
        their optimizer state_dicts + ValueNorm state_dict ; plus EMA
        reward, RNG states (torch/numpy/python/cuda), training history,
        and meta (substrate, setting, seed, next_episode).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        agents_payload = []
        for agent_id in range(self.num_agents):
            policy = self.policies[agent_id]
            trainer = self.trainers[agent_id]
            entry = {
                "actor": policy.actor.state_dict(),
                "critic": policy.critic.state_dict(),
                "actor_meta": policy.actor_meta.state_dict() if policy.actor_meta is not None else None,
                "critic_meta": policy.critic_meta.state_dict() if policy.critic_meta is not None else None,
                "actor_opt": policy.optimizers.actor.state_dict(),
                "critic_opt": policy.optimizers.critic.state_dict(),
                "actor_meta_opt": policy.optimizers.actor_meta.state_dict()
                if policy.optimizers.actor_meta is not None
                else None,
                "critic_meta_opt": policy.optimizers.critic_meta.state_dict()
                if policy.optimizers.critic_meta is not None
                else None,
                "value_normalizer": trainer.value_normalizer.state_dict()
                if trainer.value_normalizer is not None
                else None,
            }
            agents_payload.append(entry)

        cuda_rng = (
            [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else None
        )

        payload = {
            "meta": {
                "substrate": str(self.setting.id),  # setting id is a slug ; substrate comes from env_cfg separately
                "setting_id": self.setting.id,
                "setting": {
                    "id": self.setting.id,
                    "label": self.setting.label,
                    "meta": self.setting.meta,
                    "cascade_iterations1": self.setting.cascade_iterations1,
                    "cascade_iterations2": self.setting.cascade_iterations2,
                },
                "num_agents": self.num_agents,
                "num_env_steps": self.num_env_steps,
                "episode_length": self.episode_length,
                "next_episode": int(next_episode),
            },
            "agents": agents_payload,
            "ema_reward": self.ema_reward,
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
                "cuda": cuda_rng,
            },
            "all_infos": all_infos,
        }

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        log.info("checkpoint saved : %s (next_episode=%d, %d infos)", path, next_episode, len(all_infos))

    def load_checkpoint(self, path: str | Path) -> tuple[int, list[dict]]:
        """Restore the runner state from ``path`` — inverse of
        :meth:`save_checkpoint`. Returns ``(next_episode, all_infos)`` so
        :meth:`run` can resume its outer loop.

        Raises if the checkpoint's setting/num_agents don't match the current
        runner — refuses to cross-load incompatible runs.
        """
        path = Path(path)
        payload = torch.load(path, map_location=self.device, weights_only=False)

        # Sanity : reject cross-setting / cross-agent-count loads.
        ck_meta = payload["meta"]
        if ck_meta["num_agents"] != self.num_agents:
            raise ValueError(
                f"checkpoint num_agents={ck_meta['num_agents']} != runner num_agents={self.num_agents}"
            )
        if ck_meta["setting"]["id"] != self.setting.id:
            raise ValueError(
                f"checkpoint setting.id={ck_meta['setting']['id']!r} != runner setting.id={self.setting.id!r}"
            )

        for agent_id, entry in enumerate(payload["agents"]):
            policy = self.policies[agent_id]
            trainer = self.trainers[agent_id]
            policy.actor.load_state_dict(entry["actor"])
            policy.critic.load_state_dict(entry["critic"])
            if entry["actor_meta"] is not None and policy.actor_meta is not None:
                policy.actor_meta.load_state_dict(entry["actor_meta"])
            if entry["critic_meta"] is not None and policy.critic_meta is not None:
                policy.critic_meta.load_state_dict(entry["critic_meta"])
            policy.optimizers.actor.load_state_dict(entry["actor_opt"])
            policy.optimizers.critic.load_state_dict(entry["critic_opt"])
            if entry["actor_meta_opt"] is not None and policy.optimizers.actor_meta is not None:
                policy.optimizers.actor_meta.load_state_dict(entry["actor_meta_opt"])
            if entry["critic_meta_opt"] is not None and policy.optimizers.critic_meta is not None:
                policy.optimizers.critic_meta.load_state_dict(entry["critic_meta_opt"])
            if entry["value_normalizer"] is not None and trainer.value_normalizer is not None:
                trainer.value_normalizer.load_state_dict(entry["value_normalizer"])

        self.ema_reward = np.asarray(payload["ema_reward"], dtype=np.float32)

        rng = payload["rng"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            for i, state in enumerate(rng["cuda"]):
                if i < torch.cuda.device_count():
                    torch.cuda.set_rng_state(state, device=i)

        next_episode = int(ck_meta["next_episode"])
        all_infos = list(payload["all_infos"])
        return next_episode, all_infos


def alpha_broadcast(ema_per_thread: np.ndarray, reward_per_thread: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorized EMA update across threads + agents (paper eq.13).

    Parameters
    ----------
    ema_per_thread : shape (num_agents, n_rollout_threads)
    reward_per_thread : same shape
    alpha : float

    Returns
    -------
    Updated EMA, same shape.
    """
    return alpha * reward_per_thread + (1.0 - alpha) * ema_per_thread
