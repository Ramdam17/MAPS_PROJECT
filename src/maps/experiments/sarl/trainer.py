"""SARL training step — pure function port of the paper's ``train()``.

Ports the **standard DQN + metacognitive** path from
``external/MinAtar/examples/maps.py:663-887`` (commit ``ec5bcb7``).

What's in, what's out
---------------------
* **In**: all sources of randomness and state are passed as arguments. No
  global ``policy_net`` / ``optimizer`` / ``scheduler1``. The function is a
  thin wrapper over a single DQN update.
* **Out**: the continuous-learning branch (``previous_loss != None``) lives
  in ``sarl_cl.trainer`` — training with a teacher/distillation is a separate
  experiment and deserves its own module. Keeping it here would tangle
  parity with EWC/feature-preservation logic that Sprint 04b doesn't touch.

Parity with the paper
---------------------
Statement order is preserved **exactly** inside the meta branch::

    loss_second.backward(retain_graph=True)   # ① populate grads via comparison_1
    optimizer2.step()                          # ② second-order weights move
    loss.backward()                            # ③ accumulate main-task grads
    optimizer.step()                           # ④ policy_net moves (sees ①+③)
    scheduler1.step(); scheduler2.step()       # ⑤ LR decay

Because ``comparison_1`` depends on ``policy_net`` parameters, step ① also
populates ``policy_net.grad``. Step ③ ADDS to those accumulated gradients;
step ④ then updates ``policy_net`` with contributions from BOTH losses.
Swapping steps ①/③ or using a single ``.backward()`` would change the
optimizer state and diverge from the paper numerically — this is what the
Tier 3 parity test guards against.

References
----------
- Mnih et al. (2015). Human-level control through deep reinforcement learning.
  Nature 518:529-533. — vanilla DQN loss/target structure.
- Pasquali, Timmermans & Cleeremans (2010). Know thyself.
  Consciousness and Cognition. — wagering head target.
- Vargas et al. (2025), MAPS TMLR submission §3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from maps.experiments.sarl.data import Transition
from maps.experiments.sarl.losses import cae_loss

# Paper constant: DQN discount factor.
GAMMA = 0.99

# Paper constant: Jacobian regularizer weight inside CAE_loss.
CAE_LAMBDA = 1e-4


@dataclass
class SarlUpdateOutput:
    """Return shape of :func:`sarl_update_step`.

    Uniform whether ``meta`` is on or off — the meta-only fields are ``None``
    when ``meta=False`` so callers don't have to branch on tuple length.
    """

    loss: torch.Tensor
    loss_second: torch.Tensor | None = None
    q_values: torch.Tensor | None = None
    wager_logits: torch.Tensor | None = None


def sarl_update_step(
    sample: list[Transition],
    policy_net: torch.nn.Module,
    target_net: torch.nn.Module,
    second_order_net: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    optimizer2: torch.optim.Optimizer | None,
    scheduler1: Any,
    scheduler2: Any | None,
    meta: bool,
    alpha: float,
    cascade_iterations_1: int,
    cascade_iterations_2: int,
    target_wager_fn: Any,
    device: torch.device | str = "cpu",
) -> SarlUpdateOutput:
    """Run one DQN update on ``sample``; apply the two-loss cross-gradient
    pattern when ``meta=True``.

    Parameters
    ----------
    sample : list of Transition
        Batch drawn from the replay buffer (``SarlReplayBuffer.sample(B)``).
    policy_net, target_net : nn.Module
        First-order Q-networks. ``target_net`` must be frozen from the caller;
        this function never updates it.
    second_order_net : nn.Module or None
        Required iff ``meta=True``. Produces the 2-unit wager logits.
    optimizer, optimizer2 : Optimizer
        Paper uses ``optim.Adam`` for both. ``optimizer2`` is required iff
        ``meta=True``.
    scheduler1, scheduler2 : LR scheduler
        Paper uses ``StepLR(step_size=1000, gamma=0.999)``. Stepped once per
        call. Pass ``None`` for ``scheduler2`` when ``meta=False``.
    meta : bool
        Turns the metacognitive branch on/off. Paper settings 1/2/3 have
        ``meta=False``; 4/5/6 have ``meta=True``.
    alpha : float
        EMA coefficient for ``target_wager`` (in percent, per the paper:
        the paper passes e.g. ``alpha=1`` and internally divides by 100).
    cascade_iterations_1, cascade_iterations_2 : int
        Paper uses 1 when cascade is off, 50 when on. Each iteration applies
        one ``cascade_update`` step on ``hidden`` (first-order) /
        ``comparison_out`` (second-order).
    target_wager_fn : callable
        The wager-target generator. Passed explicitly so tests can swap in the
        verbatim reference implementation; production code will pass
        ``maps.experiments.sarl.data.target_wager`` (Sprint 04b §4.5 follow-up).
    device : torch.device or str
        Device for the zero-initialized ``Q_s_prime_a_prime`` buffer. The
        networks are assumed to already live on this device.

    Returns
    -------
    SarlUpdateOutput
        Contains ``loss`` (first-order, always populated), and when
        ``meta=True`` also ``loss_second``, ``q_values``, ``wager_logits``.
    """
    if meta:
        assert second_order_net is not None, "meta=True requires second_order_net"
        assert optimizer2 is not None, "meta=True requires optimizer2"
        assert scheduler2 is not None, "meta=True requires scheduler2"

    cascade_rate_1 = 1.0 / cascade_iterations_1
    cascade_rate_2 = 1.0 / cascade_iterations_2

    # Cascade carry state — None on first iteration, then fed back in.
    comparison_out = None
    main_task_out = None
    target_task_out = None

    optimizer.zero_grad()
    if meta:
        optimizer2.zero_grad()

    # Unpack transitions in batch-major order.
    batch_samples = Transition(*zip(*sample, strict=True))
    states = torch.cat(batch_samples.state)
    next_states = torch.cat(batch_samples.next_state)
    actions = torch.cat(batch_samples.action)
    rewards = torch.cat(batch_samples.reward)
    is_terminal = torch.cat(batch_samples.is_terminal)

    targets_wagering = target_wager_fn(rewards, alpha)

    # ── First-order forward (with cascade) ────────────────────────────────
    for _ in range(cascade_iterations_1):
        q_policy, h1, comparison_1, main_task_out = policy_net(
            states, main_task_out, cascade_rate_1
        )
    q_s_a = q_policy.gather(1, actions)

    # ── TD target via target network (next-state max Q) ───────────────────
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

    # Standard DQN loss (paper: CAE_loss with signature W, x=target, recons=Q_s_a).
    loss = cae_loss(W, td_target, q_s_a, h1, CAE_LAMBDA)

    if meta:
        # ── Second-order forward (cascade on comparison residual) ─────────
        for _ in range(cascade_iterations_2):
            wager, comparison_out = second_order_net(comparison_1, comparison_out, cascade_rate_2)
        loss_second = F.binary_cross_entropy_with_logits(wager, targets_wagering)

        # ── Backward pass order is load-bearing (see module docstring) ────
        loss_second.backward(retain_graph=True)
        optimizer2.step()

        loss.backward()
        optimizer.step()

        scheduler1.step()
        scheduler2.step()

        return SarlUpdateOutput(
            loss=loss,
            loss_second=loss_second,
            q_values=q_s_a,
            wager_logits=wager,
        )

    # Non-metacognitive path (settings 1/2/3).
    loss.backward()
    optimizer.step()
    scheduler1.step()
    return SarlUpdateOutput(loss=loss, q_values=q_s_a)
