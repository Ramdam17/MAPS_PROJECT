"""Reference AGL pre-training loop — frozen snapshot for parity tests.

Trimmed copy of AGL/AGL_TMLR.py `pre_train` (L619-753).
Stripped of globals, precision tracking, and the `type_cascade` asymmetric
modes 2/3; kept only 1 (both orders cascade) and 4 (no cascade), which map
to the 2×2 factorial.

Read by tests only — do not import from production code.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from maps.experiments.agl.data import target_second
from tests.parity._reference_agl import (
    ReferenceAGLFirstOrderNetwork,
    ReferenceAGLSecondOrderNetwork,
)

_bce_sum = nn.BCELoss(reduction="sum")


def _cae_loss_reference(W, x, recons_x, h, lam):
    """Verbatim CAE_loss from AGL_TMLR.py L89-127 (identical to Blindsight's)."""
    mse = _bce_sum(recons_x, x)
    dh = h * (1 - h)
    w_sum = torch.sum(Variable(W) ** 2, dim=1)
    w_sum = w_sum.unsqueeze(1)
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)


def reference_agl_pre_train(
    first_order_network: ReferenceAGLFirstOrderNetwork,
    second_order_network: ReferenceAGLSecondOrderNetwork,
    *,
    batches: list,
    meta: bool,
    cascade_on: bool,
    cascade_rate: float,
    cascade_iterations: int,
    lr_1: float,
    lr_2: float,
    step_size: int,
    gamma: float,
    lam: float,
    num_units: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Verbatim port of AGL `pre_train` with `type_cascade ∈ {1, 4}` only.

    `cascade_on=True`  ⇔ type_cascade=1 (both networks cascade)
    `cascade_on=False` ⇔ type_cascade=4 (no cascade)

    Differences vs Blindsight's reference_pre_train:
    - Reconstruction target is `patterns_tensor` itself (not stim_present).
    - Wagering target is computed per-epoch via `target_second(x, out)`.
    - At end of loop, first_order weights are restored to initial (L751).
    - `optimizer_2.zero_grad()` fires *after* the scheduler step (L695).
    """
    optimizer_1 = torch.optim.Adamax(first_order_network.parameters(), lr=lr_1)
    optimizer_2 = torch.optim.Adamax(second_order_network.parameters(), lr=lr_2)
    scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=step_size, gamma=gamma)
    scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=step_size, gamma=gamma)

    cascade_rate_one = cascade_rate if cascade_on else 1.0
    cascade_iterations_one = cascade_iterations if cascade_on else 1
    cascade_rate_two = cascade_rate_one
    cascade_iterations_two = cascade_iterations_one

    initial_first_order_weights = copy.deepcopy(first_order_network.state_dict())

    n_epochs = len(batches)
    epoch_1_order = np.zeros(n_epochs)
    epoch_2_order = np.zeros(n_epochs)

    for epoch, batch in enumerate(batches):
        patterns_tensor = batch.patterns

        hidden_representation = None
        output_first_order = None
        comparison_out = None

        for _ in range(cascade_iterations_one):
            hidden_representation, output_first_order = first_order_network(
                patterns_tensor, hidden_representation, output_first_order, cascade_rate_one
            )

        patterns_tensor = patterns_tensor.requires_grad_(True)
        output_first_order = output_first_order.requires_grad_(True)

        optimizer_1.zero_grad()

        if meta:
            for _ in range(cascade_iterations_two):
                output_second_order, comparison_out = second_order_network(
                    patterns_tensor, output_first_order, comparison_out, cascade_rate_two
                )

            output_second_order = output_second_order.squeeze()
            output_second_order = output_second_order.requires_grad_(True)

            order_2_tensor_without = target_second(patterns_tensor, output_first_order)
            loss_2 = _bce_sum(output_second_order, order_2_tensor_without).requires_grad_()

            optimizer_2.zero_grad()
            loss_2.backward(retain_graph=True)
            optimizer_2.step()
            scheduler_2.step()
            optimizer_2.zero_grad()
            epoch_2_order[epoch] = loss_2.item()
        else:
            with torch.no_grad():
                for _ in range(cascade_iterations_two):
                    output_second_order, comparison_out = second_order_network(
                        patterns_tensor, output_first_order, comparison_out, cascade_rate_two
                    )

        W = first_order_network.state_dict()["fc1.weight"]
        loss_1 = _cae_loss_reference(
            W,
            patterns_tensor,
            output_first_order,
            hidden_representation,
            lam,
        )
        loss_1.backward()
        optimizer_1.step()
        scheduler_1.step()
        epoch_1_order[epoch] = loss_1.item()

    # Reference: restore first-order to initial weights (L751).
    first_order_network.load_state_dict(initial_first_order_weights)

    return epoch_1_order, epoch_2_order
