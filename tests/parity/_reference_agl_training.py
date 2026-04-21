"""Reference AGL training() loop — frozen snapshot for parity tests.

Trimmed copy of AGL/AGL_TMLR.py ``training`` (L904-1035).
Stripped of the multi-network pool (we test on a single (fo, so) pair to isolate
the loop logic) and per-epoch `tracker.log_point()` instrumentation.

Differences vs `pre_train` reference:
- Grammar-A patterns (not random — student L947 `Array_Words(2, ...)`).
- No first-order reset at end of loop (student L904 does not restore weights).
- Per-epoch precision tracked via inlined ``calculate_metrics`` L451 —
  WTA on 6-bit chunks, then TP/(TP+FP).
- ``meta = False`` override at student L969 : when meta is passed True, the
  body still forces it to False — 2nd-order frozen (forward runs for logging
  only, no backward/step).

Read by tests only — do not import from production code.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from maps.experiments.agl.data import BITS_PER_LETTER, target_second
from tests.parity._reference_agl import (
    ReferenceAGLFirstOrderNetwork,
    ReferenceAGLSecondOrderNetwork,
)

_bce_sum = nn.BCELoss(reduction="sum")


def _cae_loss_reference(W, x, recons_x, h, lam):
    """Verbatim CAE_loss from AGL_TMLR.py (identical to pretrain reference)."""
    mse = _bce_sum(recons_x, x)
    dh = h * (1 - h)
    w_sum = torch.sum(Variable(W) ** 2, dim=1)
    w_sum = w_sum.unsqueeze(1)
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)


def _calculate_metrics_precision(patterns_tensor, output_first_order, bits_per_letter):
    """Verbatim port of AGL_TMLR.py `calculate_metrics` L451-489 — returns
    precision only (student discards recall/f1/accuracy despite computing them).
    """
    predicted_patterns = []
    for pattern in output_first_order:
        predicted_pattern = torch.zeros_like(pattern)
        for i in range(0, len(pattern), bits_per_letter):
            pack = pattern[i : i + bits_per_letter]
            max_index = torch.argmax(pack)
            if pattern[i + max_index] > 0.1:
                predicted_pattern[i + max_index] = 1
        predicted_patterns.append(predicted_pattern)
    predicted_patterns_tensor = torch.stack(predicted_patterns)
    tp = 0.0
    fp = 0.0
    for i in range(len(patterns_tensor)):
        tp += (patterns_tensor[i] * predicted_patterns_tensor[i]).sum().item()
        fp += ((1 - patterns_tensor[i]) * predicted_patterns_tensor[i]).sum().item()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def reference_agl_training(
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
    bits_per_letter: int = BITS_PER_LETTER,
    meta_frozen_override: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Verbatim port of AGL `training` with `type_cascade ∈ {1, 4}` only.

    Returns (epoch_1_order, epoch_2_order, precision_per_epoch). All shape (n,).

    Parameters
    ----------
    meta : bool
        Whether the second-order path is active at all (branches the forward
        pass cascade scope). When the student's ``training()`` is called with
        meta=True AND meta_frozen_override is True, the loop behaves as if
        meta were False (forward 2nd-order for logging, no backward/step).
    meta_frozen_override : bool
        Student L969 override. True = student behaviour (freeze 2nd-order
        during training). False = alternative (2nd-order also trains).
    """
    optimizer_1 = torch.optim.Adamax(first_order_network.parameters(), lr=lr_1)
    optimizer_2 = torch.optim.Adamax(second_order_network.parameters(), lr=lr_2)
    scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=step_size, gamma=gamma)
    scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=step_size, gamma=gamma)

    cascade_rate_one = cascade_rate if cascade_on else 1.0
    cascade_iterations_one = cascade_iterations if cascade_on else 1
    cascade_rate_two = cascade_rate_one
    cascade_iterations_two = cascade_iterations_one

    # Effective meta for gradient flow (student L969 — override).
    effective_meta = bool(meta) and not meta_frozen_override

    n_epochs = len(batches)
    epoch_1_order = np.zeros(n_epochs)
    epoch_2_order = np.zeros(n_epochs)
    precision_arr = np.zeros(n_epochs)

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

        # Second-order forward + optional backward.
        if meta:
            for _ in range(cascade_iterations_two):
                output_second_order, comparison_out = second_order_network(
                    patterns_tensor, output_first_order, comparison_out, cascade_rate_two
                )
            output_second_order = output_second_order.squeeze()

            order_2_tensor = target_second(patterns_tensor, output_first_order)
            loss_2 = _bce_sum(output_second_order, order_2_tensor)
            epoch_2_order[epoch] = loss_2.item()

            if effective_meta:
                loss_2 = loss_2.requires_grad_()
                optimizer_2.zero_grad()
                loss_2.backward(retain_graph=True)
                optimizer_2.step()
                scheduler_2.step()
                optimizer_2.zero_grad()
            # else: student L969 path — forward only, no backward/step.

        # First-order CAE loss (always).
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

        # Per-epoch precision (inlined student ``calculate_metrics`` L1025).
        with torch.no_grad():
            precision_arr[epoch] = _calculate_metrics_precision(
                patterns_tensor, output_first_order, bits_per_letter
            )

    return epoch_1_order, epoch_2_order, precision_arr
