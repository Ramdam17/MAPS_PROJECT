"""SARL DQN update step (+ two-loss meta) — MAPS §3, Table 6.

Ported from ``external/paper_reference/sarl/maps_v1.py`` ``train`` L638-859,
**non-continual-learning path only** (the ``previous_loss``/teacher branches are
SARL+CL — Sprint 15, a separate domain/branch).

One call = one DQN update on a sampled batch:
- cascade the policy net → Q-values, hidden ``h1``, comparison (Input−Output);
- TD target from the (detached, cascaded) target net on non-terminal next states;
- first-order loss = CAE with a **Huber** reconstruction term (``recon="huber"``,
  the v1-active term) on ``(target, Q_s_a)`` + the ``h(1-h)`` Jacobian on ``h1``;
- if meta: cascade the second-order net on ``comparison_1`` → wager logits,
  ``BCEWithLogits`` vs the EMA wager target.

**Two-loss gradient pattern** (source L830-841): ``optimizer.zero_grad()`` +
``optimizer2.zero_grad()`` up front, then ``loss_second.backward(retain_graph=True)``
(deposits grads on BOTH the second-order net and — via ``comparison_1`` — the
policy net), ``optimizer2.step()``, then ``loss.backward()`` accumulates on the
policy net, ``optimizer.step()`` uses the sum. Schedulers step last.

References
----------
Vargas et al. (2025), MAPS, TMLR submission §3, Table 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

from maps.core.losses import cae_loss
from maps.domains.sarl.data import Transition, target_wager

GAMMA = 0.99  # source L86
CAE_LAMBDA = 1e-4  # source train() lam


@dataclass(frozen=True)
class SarlSetting:
    """One row of the paper Table 6 factorial.

    ``cascade_iterations_{1,2}`` are 50 when the cascade is on for that pathway,
    else 1 (the student ``type_cascade`` dispatch; ``cascade_rate = 1/iters``).
    """

    id: str
    label: str
    meta: bool
    cascade_iterations_1: int
    cascade_iterations_2: int


def _make_settings(n_iter: int = 50) -> dict[str, SarlSetting]:
    return {
        "setting-1-baseline": SarlSetting("setting-1-baseline", "baseline", False, 1, 1),
        "setting-2-cascade-1st": SarlSetting(
            "setting-2-cascade-1st", "cascade 1st", False, n_iter, 1
        ),
        "setting-3-second-order-only": SarlSetting(
            "setting-3-second-order-only", "2nd-order only", True, 1, 1
        ),
        "setting-4-maps-1st": SarlSetting(
            "setting-4-maps-1st", "MAPS (cascade 1st + 2nd)", True, n_iter, 1
        ),
        "setting-5-cascade-2nd": SarlSetting(
            "setting-5-cascade-2nd", "cascade 2nd", True, 1, n_iter
        ),
        "setting-6-full-maps": SarlSetting(
            "setting-6-full-maps", "Full MAPS (cascade both + 2nd)", True, n_iter, n_iter
        ),
    }


SETTINGS_REGISTRY: dict[str, SarlSetting] = _make_settings()


def sarl_train_step(
    sample: list[Transition],
    policy_net: nn.Module,
    target_net: nn.Module,
    second_order_net: nn.Module | None,
    optimizer: optim.Optimizer,
    optimizer2: optim.Optimizer | None,
    scheduler1: StepLR,
    scheduler2: StepLR | None,
    *,
    meta: bool,
    cascade_iterations_1: int,
    cascade_iterations_2: int,
    alpha: float,
    gamma: float = GAMMA,
    cae_lambda: float = CAE_LAMBDA,
    device: torch.device | str = "cpu",
    train: bool = True,
):
    """One DQN update on ``sample``. Returns ``loss`` (or ``(loss, loss_second)`` if meta)."""
    cascade_rate_1 = float(1.0 / cascade_iterations_1)
    cascade_rate_2 = float(1.0 / cascade_iterations_2)
    main_task_out = None
    target_task_out = None
    comparison_out = None

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

    # Cascade the policy net.
    for _ in range(cascade_iterations_1):
        q_policy, h1, comparison_1, main_task_out = policy_net(
            states, main_task_out, cascade_rate_1
        )
    q_s_a = q_policy.gather(1, actions)

    # TD target from the target net on non-terminal next states (detached).
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

    # First-order loss: CAE with Huber reconstruction (v1) on (target, Q_s_a).
    loss = cae_loss(
        weight=policy_net.fc_hidden.weight,
        x=target,
        recons_x=q_s_a,
        hidden=h1,
        lam=cae_lambda,
        recon="huber",
    )

    if meta:
        assert second_order_net is not None and optimizer2 is not None and scheduler2 is not None
        for _ in range(cascade_iterations_2):
            output_second, comparison_out = second_order_net(
                comparison_1, comparison_out, cascade_rate_2
            )
        loss_second = F.binary_cross_entropy_with_logits(output_second, targets_wagering)
        if train:
            loss_second.backward(retain_graph=True)
            optimizer2.step()
            loss.backward()
            optimizer.step()
            scheduler1.step()
            scheduler2.step()
        return loss, loss_second

    if train:
        loss.backward()
        optimizer.step()
        scheduler1.step()
    return loss
