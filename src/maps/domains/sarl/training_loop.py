"""SARL DQN training loop — MAPS §3, Table 6.

Ported from the ``dqn`` frame loop in
``external/paper_reference/sarl/maps_v1.py`` L1480-1600 (Mnih et al. 2015
episodic DQN), **non-continual-learning path only** (curriculum / teacher /
adapnet / checkpoint-resume branches are SARL+CL infra — out of scope for the
faithful SARL reproduction; see Sprint-14 closeout).

Structure per the source:
- outer loop over episodes until the frame budget is spent;
- per step: ``world_dynamics`` → add to the replay buffer → once warmed
  (``t > replay_start_size`` and buffer ≥ batch), sample and ``sarl_train_step``
  every ``training_freq`` frames;
- hard target-network sync every ``target_update_freq`` policy updates.

References
----------
Vargas et al. (2025), MAPS, TMLR submission §3, Table 6.
Mnih et al. (2015). Human-level control through deep reinforcement learning.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import torch
from omegaconf import DictConfig
from torch import optim
from torch.optim.lr_scheduler import StepLR

from maps.domains.sarl.data import SarlReplayBuffer, get_state
from maps.domains.sarl.model_v1 import SarlQNetworkV1, SarlSecondOrderNetworkV1
from maps.domains.sarl.rollout import world_dynamics
from maps.domains.sarl.trainer import SarlSetting, sarl_train_step

logger = logging.getLogger(__name__)


@dataclass
class SarlMetrics:
    """Per-episode training metrics."""

    episode_returns: list[float] = field(default_factory=list)
    episode_losses_first: list[float] = field(default_factory=list)
    episode_losses_second: list[float] = field(default_factory=list)
    total_frames: int = 0
    total_updates: int = 0


def run_training(
    setting: SarlSetting,
    cfg: DictConfig,
    env,
    *,
    num_frames: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[SarlQNetworkV1, SarlSecondOrderNetworkV1 | None, SarlMetrics]:
    """Run the episodic DQN loop. Returns (policy_net, second_order_net, metrics).

    Networks/optimizers/schedulers are built here from ``cfg`` (Adam with the
    configured betas/eps; StepLR). ``env`` is a live MinAtar ``Environment``.
    """
    device = torch.device(device) if isinstance(device, str) else device
    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()

    num_frames = num_frames if num_frames is not None else int(cfg.training.num_frames)
    replay_start_size = int(cfg.training.replay_start_size)
    batch_size = int(cfg.training.batch_size)
    training_freq = int(cfg.training.training_freq)
    target_update_freq = int(cfg.training.target_update_freq)
    gamma = float(cfg.training.gamma)
    alpha = float(cfg.alpha)
    cae_lambda = float(cfg.losses.cae_lambda)
    meta = setting.meta

    # Networks: target initialised from policy (source syncs at build).
    policy_net = SarlQNetworkV1(in_channels, num_actions).to(device)
    target_net = SarlQNetworkV1(in_channels, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    second_order_net = SarlSecondOrderNetworkV1().to(device) if meta else None

    betas = tuple(cfg.optimizer.betas)
    eps = float(cfg.optimizer.eps)
    optimizer = optim.Adam(
        policy_net.parameters(), lr=float(cfg.optimizer.lr_first_order), betas=betas, eps=eps
    )
    scheduler1 = StepLR(optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)
    optimizer2 = scheduler2 = None
    if meta:
        optimizer2 = optim.Adam(
            second_order_net.parameters(),
            lr=float(cfg.optimizer.lr_second_order),
            betas=betas,
            eps=eps,
        )
        scheduler2 = StepLR(
            optimizer2, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma
        )

    buffer = SarlReplayBuffer(int(cfg.training.replay_buffer_size))
    metrics = SarlMetrics()

    t = 0
    update_counter = 0
    logger.info(
        "SARL training: setting=%s frames=%d meta=%s cascade=(%d,%d)",
        setting.id,
        num_frames,
        meta,
        setting.cascade_iterations_1,
        setting.cascade_iterations_2,
    )

    while t < num_frames:
        env.reset()
        s = get_state(env.state(), device=device)
        is_terminated = False
        episode_return = 0.0
        loss_one = 0.0
        loss_two = 0.0

        while (not is_terminated) and t < num_frames:
            s_prime, action, reward, is_terminated_t = world_dynamics(
                t,
                replay_start_size,
                num_actions,
                s,
                env,
                policy_net,
                setting.cascade_iterations_1,
                device=device,
            )
            is_terminated = bool(is_terminated_t.item())
            buffer.add(s, s_prime, action, reward, is_terminated_t)

            sample = None
            if t > replay_start_size and len(buffer) >= batch_size:
                sample = buffer.sample(batch_size)

            if t % training_freq == 0 and sample is not None:
                update_counter += 1
                out = sarl_train_step(
                    sample,
                    policy_net,
                    target_net,
                    second_order_net,
                    optimizer,
                    optimizer2,
                    scheduler1,
                    scheduler2,
                    meta=meta,
                    cascade_iterations_1=setting.cascade_iterations_1,
                    cascade_iterations_2=setting.cascade_iterations_2,
                    alpha=alpha,
                    gamma=gamma,
                    cae_lambda=cae_lambda,
                    device=device,
                )
                if meta:
                    loss_first, loss_second = out
                    loss_one += loss_first.item()
                    loss_two += loss_second.item()
                else:
                    loss_one += out.item()

                # Hard target sync every target_update_freq updates.
                if update_counter % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            episode_return += reward.item()
            t += 1
            s = s_prime

        metrics.episode_returns.append(episode_return)
        metrics.episode_losses_first.append(loss_one)
        metrics.episode_losses_second.append(loss_two)

    metrics.total_frames = t
    metrics.total_updates = update_counter
    return (
        policy_net,
        copy.deepcopy(second_order_net) if second_order_net is not None else None,
        metrics,
    )
