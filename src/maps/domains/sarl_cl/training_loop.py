"""SARL+CL curriculum training loop — MAPS §4 (CL mechanics, D15.6).

One curriculum stage = an episodic DQN loop (like SARL) that calls
:func:`sarl_cl_update_step` with an optional frozen teacher. A curriculum chains
stages over a list of games on a single channel-adaptive
:class:`AdaptiveQNetwork`: after each stage the current networks are **frozen and
deep-copied** to become the next stage's teacher (the anti-forgetting anchor).

Scope (D15.6): the CL mechanics + a small multi-game curriculum. The full
``curriculum_evaluation`` / cross-game forgetting evaluation (Figure 7) is
deferred pending Guillaume's decision D4. MinAtar env RNG is unseeded (faithful).

References
----------
Vargas et al. (2025), MAPS, TMLR submission §4, Figure 7.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

from maps.domains.sarl_cl.data import SarlReplayBuffer, get_state
from maps.domains.sarl_cl.loss_weighting import DynamicLossWeighter, LossMixingWeights
from maps.domains.sarl_cl.model import AdaptiveQNetwork, SarlCLSecondOrderNetwork
from maps.domains.sarl_cl.rollout import world_dynamics
from maps.domains.sarl_cl.trainer import sarl_cl_update_step

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    game: str
    episode_returns: list[float] = field(default_factory=list)
    total_frames: int = 0
    total_updates: int = 0


def _build_optims(policy_net, second_order_net, cfg, meta):
    betas = tuple(cfg.optimizer.betas)
    eps = float(cfg.optimizer.eps)
    opt = optim.Adam(
        policy_net.parameters(), lr=float(cfg.optimizer.lr_first_order), betas=betas, eps=eps
    )
    sch = StepLR(opt, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)
    opt2 = sch2 = None
    if meta:
        opt2 = optim.Adam(
            second_order_net.parameters(),
            lr=float(cfg.optimizer.lr_second_order),
            betas=betas,
            eps=eps,
        )
        sch2 = StepLR(opt2, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)
    return opt, opt2, sch, sch2


def run_stage(
    policy_net: AdaptiveQNetwork,
    target_net: AdaptiveQNetwork,
    second_order_net: nn.Module | None,
    env,
    cfg: DictConfig,
    setting,
    *,
    num_frames: int,
    teacher_first_net: nn.Module | None = None,
    teacher_second_net: nn.Module | None = None,
    mixing: LossMixingWeights | None = None,
    device: torch.device | str = "cpu",
) -> StageMetrics:
    """Run one curriculum stage (episodic DQN on ``env`` with optional teacher)."""
    device = torch.device(device) if isinstance(device, str) else device
    num_actions = env.num_actions()
    replay_start_size = int(cfg.training.replay_start_size)
    batch_size = int(cfg.training.batch_size)
    training_freq = int(cfg.training.training_freq)
    target_update_freq = int(cfg.training.target_update_freq)
    gamma = float(cfg.training.gamma)
    alpha = float(cfg.alpha)
    cae_lambda = float(cfg.losses.cae_lambda)
    meta = setting.meta

    opt, opt2, sch, sch2 = _build_optims(policy_net, second_order_net, cfg, meta)
    loss_weighter = DynamicLossWeighter() if teacher_first_net is not None else None
    loss_weighter_second = (
        DynamicLossWeighter() if (teacher_first_net is not None and meta) else None
    )

    buffer = SarlReplayBuffer(int(cfg.training.replay_buffer_size))
    metrics = StageMetrics(game=getattr(env, "game_name", "unknown"))
    t = 0
    update_counter = 0

    while t < num_frames:
        env.reset()
        s = get_state(env.state(), device=device)
        is_terminated = False
        episode_return = 0.0
        while (not is_terminated) and t < num_frames:
            s_prime, action, reward, term_t = world_dynamics(
                t,
                replay_start_size,
                num_actions,
                s,
                env,
                policy_net,
                setting.cascade_iterations_1,
                device=device,
            )
            is_terminated = bool(term_t.item())
            buffer.add(s, s_prime, action, reward, term_t)

            if t > replay_start_size and len(buffer) >= batch_size and t % training_freq == 0:
                update_counter += 1
                sarl_cl_update_step(
                    buffer.sample(batch_size),
                    policy_net,
                    target_net,
                    second_order_net,
                    opt,
                    opt2,
                    sch,
                    sch2,
                    teacher_first_net=teacher_first_net,
                    teacher_second_net=teacher_second_net,
                    loss_weighter=loss_weighter,
                    loss_weighter_second=loss_weighter_second,
                    mixing=mixing,
                    meta=meta,
                    cascade_iterations_1=setting.cascade_iterations_1,
                    cascade_iterations_2=setting.cascade_iterations_2,
                    alpha=alpha,
                    gamma=gamma,
                    cae_lambda=cae_lambda,
                    device=device,
                )
                if update_counter % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            episode_return += reward.item()
            t += 1
            s = s_prime
        metrics.episode_returns.append(episode_return)

    metrics.total_frames = t
    metrics.total_updates = update_counter
    return metrics


def run_curriculum(
    games: list[str],
    cfg: DictConfig,
    setting,
    *,
    frames_per_stage: int,
    max_input_channels: int = 10,
    device: torch.device | str = "cpu",
) -> tuple[AdaptiveQNetwork, list[StageMetrics]]:
    """Chain stages over ``games`` on one AdaptiveQNetwork; freeze→teacher between stages."""
    from minatar import Environment

    num_actions = max(Environment(g).num_actions() for g in games)
    policy_net = AdaptiveQNetwork(max_input_channels, num_actions).to(device)
    target_net = AdaptiveQNetwork(max_input_channels, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    second_order_net = SarlCLSecondOrderNetwork().to(device) if setting.meta else None
    mixing = LossMixingWeights()

    teacher_first = teacher_second = None
    all_metrics: list[StageMetrics] = []
    for game in games:
        env = Environment(game)
        env.game_name = game  # for metrics label
        logger.info("CL stage: game=%s (teacher=%s)", game, teacher_first is not None)
        m = run_stage(
            policy_net,
            target_net,
            second_order_net,
            env,
            cfg,
            setting,
            num_frames=frames_per_stage,
            teacher_first_net=teacher_first,
            teacher_second_net=teacher_second,
            mixing=mixing,
            device=device,
        )
        all_metrics.append(m)
        # Freeze current nets → next stage's teacher (anti-forgetting anchor).
        teacher_first = copy.deepcopy(policy_net).requires_grad_(False)
        if setting.meta:
            teacher_second = copy.deepcopy(second_order_net).requires_grad_(False)

    return policy_net, all_metrics
