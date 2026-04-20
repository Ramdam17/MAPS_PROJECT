"""SARL+CL training step — pure function port of the paper's ``train()``.

Ports the continual-learning DQN update from
``SARL_CL/examples_cl/maps.py:610-829`` (``train()``).

Relationship to :mod:`maps.experiments.sarl.trainer`
----------------------------------------------------
The standard SARL ``sarl_update_step`` covers **no-teacher** training
(paper settings 1-6 when running on a fresh task). When the CL curriculum
advances to task ≥ 2, we need THREE additional loss terms per network:

* ``task`` — current-task loss (CAE for FO, BCE-with-logits for SO).
* ``distillation`` — weight-regularization L2 anchor between student and
  frozen teacher (paper calls it "distillation" for parity; see
  :func:`maps.components.losses.weight_regularization` for why the name
  is misleading).
* ``feature`` — MSE between student and teacher intermediate activations
  (``h1`` for FO, ``comparison_out`` for SO).

These are normalized by a :class:`DynamicLossWeighter` (one per network)
via running-max division, then mixed with fixed scalar weights
(``task_weight``, ``distillation_weight``, ``feature_weight``). The paper
uses (1.0, 1.0, 1.0) — captured here via the ``LossMixingWeights`` dataclass.

Parity with the paper
---------------------
Statement order inside the meta + CL branch is preserved exactly::

    loss_second.backward(retain_graph=True)   # ① SO grads (→ policy_net via comparison_1)
    optimizer2.step()                          # ② SO weights move
    loss.backward()                            # ③ FO grads accumulate (+ ① residuals)
    optimizer.step()                           # ④ policy_net moves
    scheduler1.step(); scheduler2.step()       # ⑤ LR decay

This two-backward order is load-bearing: swapping it changes the
gradient accumulated into ``policy_net`` because ``comparison_1`` is
produced from ``policy_net`` layers.

Paper dead-code warning
-----------------------
The paper defines a ``DistillationLoss`` class (lines ~355-395 of the
source file) that is **never called** in ``train()`` — the actual anchor
is ``compute_weight_regularization`` (L2 param diff). We faithfully
carry this: the ``distillation`` KEY name is preserved in the loss dict
for :class:`DynamicLossWeighter` compatibility, but the VALUE is L2.

References
----------
- SARL_CL/examples_cl/maps.py:610-829 (source).
- Kirkpatrick et al. (2017). Overcoming catastrophic forgetting in NNs. PNAS.
- Vargas et al. (2025), MAPS TMLR submission §4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from maps.components import weight_regularization
from maps.experiments.sarl.data import Transition
from maps.experiments.sarl.losses import cae_loss
from maps.experiments.sarl_cl.loss_weighting import DynamicLossWeighter

# Paper constant: DQN discount factor.
GAMMA = 0.99

# Paper constant: Jacobian regularizer weight inside CAE_loss (maps.py:704).
CAE_LAMBDA = 1e-4


@dataclass
class LossMixingWeights:
    """Fixed scalar weights applied AFTER the DynamicLossWeighter.

    The paper keeps these at (1, 1, 1) but exposes them as WEIGHT1/2/3 —
    we surface them as a dataclass so ablations can vary them without
    touching the function signature. Paper refs:
    ``WEIGHT1`` = task mixing weight (``ce_loss_weight``), line 708.
    ``WEIGHT2`` = distillation mixing weight (``soft_target_loss_weight``),
    line 709.
    ``WEIGHT3`` = feature preservation mixing weight (``feature_weight``),
    line 710.
    """

    task: float = 1.0
    distillation: float = 1.0
    feature: float = 1.0


@dataclass
class SarlCLComponentLosses:
    """Decomposed CL losses for logging / debugging — all detached scalars."""

    task: float
    distillation: float
    feature: float


@dataclass
class SarlCLUpdateOutput:
    """Return shape of :func:`sarl_cl_update_step`.

    Uniform whether we're in CL mode (teacher present) or vanilla mode —
    CL-only fields are ``None`` when no teacher is supplied.
    """

    loss: torch.Tensor
    loss_second: torch.Tensor | None = None
    q_values: torch.Tensor | None = None
    wager_logits: torch.Tensor | None = None
    components_first: SarlCLComponentLosses | None = None
    components_second: SarlCLComponentLosses | None = None


def sarl_cl_update_step(
    sample: list[Transition],
    policy_net: torch.nn.Module,
    target_net: torch.nn.Module,
    second_order_net: torch.nn.Module | None,
    teacher_first_net: torch.nn.Module | None,
    teacher_second_net: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    optimizer2: torch.optim.Optimizer | None,
    scheduler1: Any,
    scheduler2: Any | None,
    loss_weighter: DynamicLossWeighter | None,
    loss_weighter_second: DynamicLossWeighter | None,
    mixing: LossMixingWeights,
    meta: bool,
    alpha: float,
    cascade_iterations_1: int,
    cascade_iterations_2: int,
    target_wager_fn: Any,
    train: bool = True,
    device: torch.device | str = "cpu",
) -> SarlCLUpdateOutput:
    """Run one CL-aware DQN update on ``sample``.

    When ``teacher_first_net`` is ``None``, the update degenerates to the
    standard SARL path (same numerical behavior as
    :func:`maps.experiments.sarl.trainer.sarl_update_step`, minus the
    fc_hidden tied-weights quirk — CL nets use an explicit ``fc_output``).

    When a teacher is present, the three-component loss composition is
    applied per network:

    * FO: ``task = CAE(W, target, Q_s_a, h1, λ)``;
      ``distillation = weight_regularization(policy, teacher_first)``;
      ``feature = MSE(h1, h1_teacher)``.
    * SO (if ``meta``): ``task = BCE-with-logits(wager, targets_wagering)``;
      ``distillation = weight_regularization(second, teacher_second)``;
      ``feature = MSE(comparison_out, comparison_out_teacher)``.

    Parameters
    ----------
    sample : list of Transition
        Batch drawn from the replay buffer.
    policy_net, target_net : nn.Module
        First-order Q-networks. The CL variant must expose the 4-tuple
        forward contract ``(q, h1, comparison_1, main_task_out)``.
        ``target_net`` is never updated here.
    second_order_net : nn.Module or None
        Required iff ``meta=True``. Returns ``(wager_logits, comparison_out)``.
    teacher_first_net, teacher_second_net : nn.Module or None
        Frozen teachers. When ``teacher_first_net`` is None we skip the CL
        branch entirely (no distillation, no feature MSE). If ``meta`` is
        True and ``teacher_first_net`` is set, ``teacher_second_net`` MUST
        also be set — raises ``AssertionError`` otherwise.
    optimizer, optimizer2 : Optimizer
        Paper uses ``optim.Adam``. ``optimizer2`` is required iff ``meta``.
    scheduler1, scheduler2 : LR scheduler
        Paper uses ``StepLR(step_size=1000, gamma=0.999)``. Stepped once per
        call IFF ``train=True``.
    loss_weighter, loss_weighter_second : DynamicLossWeighter or None
        Required when a teacher is present. The paper maintains one per
        network across the whole curriculum.
    mixing : LossMixingWeights
        Fixed (task/distillation/feature) mixing coefficients applied after
        dynamic normalization.
    meta : bool
        Turns the SO branch on/off.
    alpha : float
        Percent EMA coefficient for ``target_wager`` (paper passes percent
        and divides by 100 internally).
    cascade_iterations_1, cascade_iterations_2 : int
        Paper uses 1 when cascade is off, 50 when on.
    target_wager_fn : callable
        Wager-target generator (``rewards, alpha) -> Tensor``).
    train : bool, default True
        If True, runs backward + optimizer.step + scheduler.step. If False,
        only forward-computes the losses (for validation).
    device : torch.device or str

    Returns
    -------
    SarlCLUpdateOutput
        ``loss`` always populated. ``loss_second`` when ``meta``. Component
        breakdowns populated only when a teacher is present.
    """
    if meta:
        assert second_order_net is not None, "meta=True requires second_order_net"
        assert optimizer2 is not None, "meta=True requires optimizer2"
        assert scheduler2 is not None, "meta=True requires scheduler2"

    cl_enabled = teacher_first_net is not None
    if cl_enabled:
        assert loss_weighter is not None, "CL mode requires loss_weighter"
        if meta:
            assert teacher_second_net is not None, "meta=True CL mode requires teacher_second_net"
            assert loss_weighter_second is not None, (
                "meta=True CL mode requires loss_weighter_second"
            )

    cascade_rate_1 = 1.0 / cascade_iterations_1
    cascade_rate_2 = 1.0 / cascade_iterations_2

    # Cascade carry state (student).
    comparison_out: torch.Tensor | None = None
    main_task_out: torch.Tensor | None = None
    target_task_out: torch.Tensor | None = None
    # Cascade carry state (teacher).
    comparison_out_teacher: torch.Tensor | None = None
    main_task_out_teacher: torch.Tensor | None = None

    optimizer.zero_grad()
    if meta:
        optimizer2.zero_grad()

    batch_samples = Transition(*zip(*sample, strict=True))
    states = torch.cat(batch_samples.state)
    next_states = torch.cat(batch_samples.next_state)
    actions = torch.cat(batch_samples.action)
    rewards = torch.cat(batch_samples.reward)
    is_terminal = torch.cat(batch_samples.is_terminal)

    targets_wagering = target_wager_fn(rewards, alpha)

    # ── Student FO forward (cascade) ──────────────────────────────────────
    for _ in range(cascade_iterations_1):
        q_policy, h1, comparison_1, main_task_out = policy_net(
            states, main_task_out, cascade_rate_1
        )
    q_s_a = q_policy.gather(1, actions)

    # ── Teacher FO forward (no grad) ──────────────────────────────────────
    h1_teacher: torch.Tensor | None = None
    comparison_1_teacher: torch.Tensor | None = None
    if cl_enabled:
        with torch.no_grad():
            for _ in range(cascade_iterations_1):
                (
                    _,
                    h1_teacher,
                    comparison_1_teacher,
                    main_task_out_teacher,
                ) = teacher_first_net(states, main_task_out_teacher, cascade_rate_1)

    # ── TD target ─────────────────────────────────────────────────────────
    non_terminal_idx = torch.tensor(
        [i for i, done in enumerate(is_terminal) if done == 0],
        dtype=torch.int64,
        device=device,
    )
    non_terminal_next = next_states.index_select(0, non_terminal_idx)
    q_s_prime = torch.zeros(len(sample), 1, device=device)
    if len(non_terminal_next) != 0:
        for _ in range(cascade_iterations_1):
            q_target, _, _, target_task_out = target_net(
                non_terminal_next, target_task_out, cascade_rate_1
            )
        q_s_prime[non_terminal_idx] = q_target.detach().max(1)[0].unsqueeze(1)
    td_target = rewards + GAMMA * q_s_prime

    # Live weight view — gradients flow back through W via the Jacobian term.
    W = policy_net.state_dict()["fc_hidden.weight"]

    # ── FO loss composition ───────────────────────────────────────────────
    components_first: SarlCLComponentLosses | None = None
    if cl_enabled:
        loss_task = cae_loss(W, td_target, q_s_a, h1, CAE_LAMBDA)
        loss_distillation = weight_regularization(policy_net, teacher_first_net)
        loss_feature = F.mse_loss(h1, h1_teacher)

        raw = {"task": loss_task, "distillation": loss_distillation, "feature": loss_feature}
        loss_weighter.update(raw)
        weighted = loss_weighter.weight_losses(raw)

        loss = (
            mixing.task * weighted["task"]
            + mixing.distillation * weighted["distillation"]
            + mixing.feature * weighted["feature"]
        )
        components_first = SarlCLComponentLosses(
            task=float(weighted["task"].item()),
            distillation=float(weighted["distillation"].item()),
            feature=float(weighted["feature"].item()),
        )
    else:
        loss = cae_loss(W, td_target, q_s_a, h1, CAE_LAMBDA)

    # ── SO branch ─────────────────────────────────────────────────────────
    if not meta:
        if train:
            loss.backward()
            optimizer.step()
            scheduler1.step()
        return SarlCLUpdateOutput(
            loss=loss,
            q_values=q_s_a,
            components_first=components_first,
        )

    # SO forward (cascade on comparison residual).
    for _ in range(cascade_iterations_2):
        wager, comparison_out = second_order_net(comparison_1, comparison_out, cascade_rate_2)

    components_second: SarlCLComponentLosses | None = None
    if cl_enabled:
        with torch.no_grad():
            for _ in range(cascade_iterations_2):
                _, comparison_out_teacher = teacher_second_net(
                    comparison_1_teacher, comparison_out_teacher, cascade_rate_2
                )

        task_second = F.binary_cross_entropy_with_logits(wager, targets_wagering)
        distill_second = weight_regularization(second_order_net, teacher_second_net)
        feature_second = F.mse_loss(comparison_out, comparison_out_teacher)

        raw_second = {
            "task": task_second,
            "distillation": distill_second,
            "feature": feature_second,
        }
        loss_weighter_second.update(raw_second)
        weighted_second = loss_weighter_second.weight_losses(raw_second)

        loss_second = (
            mixing.task * weighted_second["task"]
            + mixing.distillation * weighted_second["distillation"]
            + mixing.feature * weighted_second["feature"]
        )
        components_second = SarlCLComponentLosses(
            task=float(weighted_second["task"].item()),
            distillation=float(weighted_second["distillation"].item()),
            feature=float(weighted_second["feature"].item()),
        )
    else:
        loss_second = F.binary_cross_entropy_with_logits(wager, targets_wagering)

    # ── Backward (order is load-bearing, see module docstring) ────────────
    if train:
        loss_second.backward(retain_graph=True)
        optimizer2.step()

        loss.backward()
        optimizer.step()

        scheduler1.step()
        scheduler2.step()

    return SarlCLUpdateOutput(
        loss=loss,
        loss_second=loss_second,
        q_values=q_s_a,
        wager_logits=wager,
        components_first=components_first,
        components_second=components_second,
    )
