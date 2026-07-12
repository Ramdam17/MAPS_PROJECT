"""SARL+CL DQN update step — MAPS §4 (continual learning).

Ported from ``external/paper_reference/sarl_cl_maps.py`` ``train`` L638-859. Adds
the three continual-learning terms to the standard SARL DQN update when a frozen
teacher is present:

- **task**         — current-task loss (CAE-Huber for FO, BCEWithLogits for SO);
- **distillation** — L2 parameter anchor vs the teacher
  (:func:`maps.core.losses.weight_regularization`, D15.4 — *not* Hinton KL);
- **feature**      — MSE between student and teacher intermediate activations
  (``h1`` for FO, ``comparison_out`` for SO).

Each component is normalised by a :class:`DynamicLossWeighter` (per-key running
max) then combined with fixed :class:`LossMixingWeights` (0.4/0.4/0.2, D15.5).
With ``teacher_*_net=None`` the update degenerates to the plain SARL DQN step
(the first task in a curriculum, no teacher yet).

Two-loss gradient pattern identical to SARL (loss_second.backward(retain_graph)
→ opt2.step → loss.backward → opt.step → schedulers).

References
----------
Vargas et al. (2025), MAPS, TMLR submission §4, Figure 7.
Kirkpatrick et al. (2017). Overcoming catastrophic forgetting (EWC).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import StepLR

from maps.core.losses import cae_loss, weight_regularization
from maps.domains.sarl_cl.data import Transition, target_wager
from maps.domains.sarl_cl.loss_weighting import DynamicLossWeighter, LossMixingWeights

GAMMA = 0.99
CAE_LAMBDA = 1e-4


@dataclass(frozen=True)
class SarlCLSetting:
    """One row of the SARL+CL factorial (same 6-cell schema as SARL)."""

    id: str
    label: str
    meta: bool
    cascade_iterations_1: int
    cascade_iterations_2: int


def _make_settings(n_iter: int = 50) -> dict[str, SarlCLSetting]:
    return {
        "setting-1-baseline": SarlCLSetting("setting-1-baseline", "baseline", False, 1, 1),
        "setting-2-cascade-1st": SarlCLSetting(
            "setting-2-cascade-1st", "cascade 1st", False, n_iter, 1
        ),
        "setting-3-second-order-only": SarlCLSetting(
            "setting-3-second-order-only", "2nd-order only", True, 1, 1
        ),
        "setting-4-maps-1st": SarlCLSetting(
            "setting-4-maps-1st", "MAPS (cascade 1st + 2nd)", True, n_iter, 1
        ),
        "setting-5-cascade-2nd": SarlCLSetting(
            "setting-5-cascade-2nd", "cascade 2nd", True, 1, n_iter
        ),
        "setting-6-full-maps": SarlCLSetting(
            "setting-6-full-maps", "Full MAPS (cascade both + 2nd)", True, n_iter, n_iter
        ),
    }


SETTINGS_REGISTRY: dict[str, SarlCLSetting] = _make_settings()


@dataclass
class ComponentLosses:
    """task / distillation / feature scalars (for logging)."""

    task: float = 0.0
    distillation: float = 0.0
    feature: float = 0.0


def _combine(
    weighter: DynamicLossWeighter, mixing: LossMixingWeights, task, distill, feat
) -> Tensor:
    """Normalise the 3 components by their running max, then mix (source L734-742)."""
    current = {"task": task, "distillation": distill, "feature": feat}
    weighter.update(current)
    w = weighter.weight_losses(current)
    return (
        mixing.distillation * w["distillation"]
        + mixing.task * w["task"]
        + mixing.feature * w["feature"]
    )


def sarl_cl_update_step(
    sample: list[Transition],
    policy_net: nn.Module,
    target_net: nn.Module,
    second_order_net: nn.Module | None,
    optimizer: optim.Optimizer,
    optimizer2: optim.Optimizer | None,
    scheduler1: StepLR,
    scheduler2: StepLR | None,
    *,
    teacher_first_net: nn.Module | None = None,
    teacher_second_net: nn.Module | None = None,
    loss_weighter: DynamicLossWeighter | None = None,
    loss_weighter_second: DynamicLossWeighter | None = None,
    mixing: LossMixingWeights | None = None,
    meta: bool,
    cascade_iterations_1: int,
    cascade_iterations_2: int,
    alpha: float,
    gamma: float = GAMMA,
    cae_lambda: float = CAE_LAMBDA,
    device: torch.device | str = "cpu",
    train: bool = True,
):
    """One CL DQN update. Returns (loss, loss_second, comp_first, comp_second)."""
    mixing = mixing or LossMixingWeights()
    has_teacher = teacher_first_net is not None
    cascade_rate_1 = float(1.0 / cascade_iterations_1)
    cascade_rate_2 = float(1.0 / cascade_iterations_2)
    main_task_out = target_task_out = comparison_out = None
    main_task_out_teacher = None

    optimizer.zero_grad()
    if meta and optimizer2 is not None:
        optimizer2.zero_grad()

    batch = Transition(*zip(*sample, strict=True))
    states = torch.cat(batch.state)
    next_states = torch.cat(batch.next_state)
    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)
    is_terminal = torch.cat(batch.is_terminal)
    targets_wagering = target_wager(rewards, alpha)

    for _ in range(cascade_iterations_1):
        q_policy, h1, comparison_1, main_task_out = policy_net(
            states, main_task_out, cascade_rate_1
        )
    q_s_a = q_policy.gather(1, actions)

    h1_teacher = None
    if has_teacher:
        with torch.no_grad():
            for _ in range(cascade_iterations_1):
                _, h1_teacher, _c1t, main_task_out_teacher = teacher_first_net(
                    states, main_task_out_teacher, cascade_rate_1
                )

    nonterm_idx = torch.tensor(
        [i for i, term in enumerate(is_terminal) if term == 0], dtype=torch.int64, device=device
    )
    nonterm_next = next_states.index_select(0, nonterm_idx)
    q_next = torch.zeros(len(sample), 1, device=device)
    if len(nonterm_next) != 0:
        for _ in range(cascade_iterations_1):
            q_target, _, _, target_task_out = target_net(
                nonterm_next, target_task_out, cascade_rate_1
            )
        q_next[nonterm_idx] = q_target.detach().max(1)[0].unsqueeze(1)
    target = rewards + gamma * q_next

    loss_task = cae_loss(
        weight=policy_net.fc_hidden.weight,
        x=target,
        recons_x=q_s_a,
        hidden=h1,
        lam=cae_lambda,
        recon="huber",
    )
    comp_first = ComponentLosses(task=float(loss_task.item()))
    if has_teacher:
        assert loss_weighter is not None
        loss_distill = weight_regularization(policy_net, teacher_first_net)
        loss_feature = F.mse_loss(h1, h1_teacher)
        comp_first = ComponentLosses(
            float(loss_task.item()), float(loss_distill.item()), float(loss_feature.item())
        )
        loss = _combine(loss_weighter, mixing, loss_task, loss_distill, loss_feature)
    else:
        loss = loss_task

    loss_second = torch.zeros((), device=device)
    comp_second = ComponentLosses()
    if meta:
        assert second_order_net is not None and optimizer2 is not None and scheduler2 is not None
        for _ in range(cascade_iterations_2):
            output_second, comparison_out = second_order_net(
                comparison_1, comparison_out, cascade_rate_2
            )
        task_second = F.binary_cross_entropy_with_logits(output_second, targets_wagering)
        comp_second = ComponentLosses(task=float(task_second.item()))
        if has_teacher and teacher_second_net is not None:
            assert loss_weighter_second is not None
            comp_out_teacher = None
            with torch.no_grad():
                for _ in range(cascade_iterations_2):
                    _, comp_out_teacher = teacher_second_net(
                        comparison_1.detach(), comp_out_teacher, cascade_rate_2
                    )
            distill_second = weight_regularization(second_order_net, teacher_second_net)
            feature_second = F.mse_loss(comparison_out, comp_out_teacher)
            comp_second = ComponentLosses(
                float(task_second.item()),
                float(distill_second.item()),
                float(feature_second.item()),
            )
            loss_second = _combine(
                loss_weighter_second, mixing, task_second, distill_second, feature_second
            )
        else:
            loss_second = task_second

        if train:
            loss_second.backward(retain_graph=True)
            optimizer2.step()
            loss.backward()
            optimizer.step()
            scheduler1.step()
            scheduler2.step()
        return loss, loss_second, comp_first, comp_second

    if train:
        loss.backward()
        optimizer.step()
        scheduler1.step()
    return loss, loss_second, comp_first, comp_second
