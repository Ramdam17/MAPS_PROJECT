"""Paper-faithful reference for SARL+CL parity tests (Sprint-08 D.22).

Origin
------
Extracted from ``external/paper_reference/sarl_cl_maps.py`` (Vargas et al.,
MAPS TMLR 2025). Like ``tests/parity/sarl/_reference_sarl.py`` after Sprint-08
D.15, this module tracks **paper Table 11** rather than the verbatim student
values where the two disagree. Constants aligned:

- ``GAMMA = 0.999``           (paper T.11 row 14; student = 0.99)
- ``step_size1 = 0.0003``      (paper T.11 row 9; student = 0.0003)
- ``step_size2 = 0.0002``      (paper T.11 row 10; student = 0.00005)
- ``adam_betas = (0.95, 0.95)`` (paper T.11 rows 11-12; student omitted)
- ``TARGET_UPDATE_FREQ = 500`` (CL paper sarl_cl_maps.py:1121 — unchanged)

The D-sarl-recon-bias paper eq.12 bias does NOT apply to the CL variant: the
CL QNetwork has an explicit ``fc_output`` decoder (not tied-weight), so the
default ``nn.Linear(bias=True)`` already carries a learnable bias.

Stripped vs original source
---------------------------
- No module-level ``NvidiaEnergyTracker`` instantiation (import side effect).
- No module-level ``print("using ", device)``.
- ``device`` resolved to CPU (parity tests run on CPU).
- The paper's ``train()`` function is rewritten as a pure-function
  ``reference_dqn_update_step_cl`` with globals lifted to kwargs, matching
  the sarl/_reference_sarl.reference_dqn_update_step pattern.

Nothing else is altered — formatting, capitalisation, and the paper's
misspelling of "Comparisson" are preserved. The ``ruff: noqa`` block at
the top tolerates the paper naming style.
"""
# ruff: noqa: N801, N802, N803, N806, E741, B007, SIM108, UP008, UP032, RUF001, E712, E711, F841

# flake8: noqa

from __future__ import annotations

import copy
import random
from collections import namedtuple
from typing import Any

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init
from torch.autograd import Variable  # deprecated but present in paper source

# Parity tests pin CPU for reproducibility.
device = torch.device("cpu")

# Paper Table 11 constants (D.22 alignment; see module docstring).
GAMMA = 0.999
MIN_SQUARED_GRAD = 0.01
step_size1 = 0.0003
step_size2 = 0.0002
scheduler_step = 0.999
TARGET_UPDATE_FREQ = 500  # CL-specific: sarl_cl_maps.py:1121
# Paper Table 11 CL rows 21-23:
WEIGHT1 = 0.3  # task mixing weight
WEIGHT2 = 0.6  # distillation (weight-reg) mixing weight
WEIGHT3 = 0.1  # feature preservation mixing weight


# ─── Helpers ────────────────────────────────────────────────────────────────


def size_linear_unit(size, kernel_size=3, stride=1):
    return (size - (kernel_size - 1) - 1) // stride + 1


num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16


# ─── QNetwork (standard CL variant, explicit fc_output) ─────────────────────


class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(QNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
        self.fc_output = nn.Linear(in_features=128, out_features=num_linear_units)
        self.actions = nn.Linear(in_features=num_linear_units, out_features=num_actions)

    def forward(self, x, prev_h2, cascade_rate):
        x = f.relu(self.conv(x))
        Input = x.view(x.size(0), -1)
        Hidden = f.relu(self.fc_hidden(Input))
        Output = f.relu(self.fc_output(Hidden))

        if prev_h2 is not None:
            Output = cascade_rate * Output + (1 - cascade_rate) * prev_h2

        x = self.actions(Output)
        Comparisson = Input - Output

        return x, Hidden, Comparisson, Output


# ─── AdaptiveQNetwork (variable-channel variant) ────────────────────────────


class AdaptiveQNetwork(nn.Module):
    def __init__(self, max_input_channels, num_actions):
        super(AdaptiveQNetwork, self).__init__()

        self.max_input_channels = max_input_channels

        self.input_adapter = nn.Sequential(
            nn.Conv2d(max_input_channels, max_input_channels, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.conv = nn.Conv2d(max_input_channels, 16, kernel_size=3, stride=1)

        conv_output_size = self._get_conv_output_size((max_input_channels, 10, 10))

        self.fc_hidden = nn.Linear(in_features=conv_output_size, out_features=128)
        self.fc_output = nn.Linear(in_features=128, out_features=conv_output_size)
        self.actions = nn.Linear(in_features=conv_output_size, out_features=num_actions)

    def _get_conv_output_size(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output = self.conv(input)
        return int(numpy.prod(output.size()[1:]))

    def adapt_input(self, x):
        if x.size(1) < self.max_input_channels:
            padding = torch.zeros(
                x.size(0),
                self.max_input_channels - x.size(1),
                x.size(2),
                x.size(3),
                device=x.device,
            )
            x = torch.cat([x, padding], dim=1)
        return x

    def forward(self, x, prev_h2, cascade_rate):
        x = self.adapt_input(x)
        x = self.input_adapter(x)
        x = f.relu(self.conv(x))
        Input = x.view(x.size(0), -1)
        Hidden = f.relu(self.fc_hidden(Input))
        Output = f.relu(self.fc_output(Hidden))

        if prev_h2 is not None:
            Output = cascade_rate * Output + (1 - cascade_rate) * prev_h2

        x = self.actions(Output)
        Comparison = Input - Output

        return x, Hidden, Comparison, Output


# ─── SecondOrderNetwork (CL variant with explicit comparison_layer) ─────────


class SecondOrderNetwork(nn.Module):
    def __init__(self, in_channels):
        super(SecondOrderNetwork, self).__init__()

        self.comparison_layer = nn.Linear(
            in_features=num_linear_units, out_features=num_linear_units
        )
        self.wager = nn.Linear(num_linear_units, 2)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        init.uniform_(self.comparison_layer.weight, -1.0, 1.0)
        init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(self, comparison_matrix, prev_comparison, cascade_rate):
        comparison_out = self.dropout(f.relu(self.comparison_layer(comparison_matrix)))

        if prev_comparison is not None:
            comparison_out = cascade_rate * comparison_out + (1 - cascade_rate) * prev_comparison

        wager = self.wager(comparison_out)

        return wager, comparison_out


# ─── Replay buffer + helpers ────────────────────────────────────────────────


transition = namedtuple("transition", "state, next_state, action, reward, is_terminal")


class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


# ─── target_wager (EMA wagering target) ─────────────────────────────────────


def target_wager(rewards, alpha):
    flattened_rewards = rewards.view(-1)
    alpha = float(alpha / 100)
    EMA = 0.0

    batch_size = rewards.size(0)
    new_tensor = torch.zeros(batch_size, 2, device=rewards.device)

    for i in range(batch_size):
        G = flattened_rewards[i]
        EMA = alpha * G + (1 - alpha) * EMA

        if G > EMA:
            new_tensor[i] = torch.tensor([1, 0], device=rewards.device)
        else:
            new_tensor[i] = torch.tensor([0, 1], device=rewards.device)

    return new_tensor


# ─── CAE_loss (Huber reconstruction + Jacobian regularizer) ─────────────────


def CAE_loss(W, x, recons_x, h, lam):
    mse = f.huber_loss(recons_x, x)

    dh = h * (1 - h)  # Hadamard product
    w_sum = torch.sum(Variable(W) ** 2, dim=1)
    w_sum = w_sum.unsqueeze(1)
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)


# ─── compute_weight_regularization (L2 anchor vs teacher) ───────────────────


def compute_weight_regularization(model, teacher_model):
    reg_loss = 0
    for param, param_old in zip(model.parameters(), teacher_model.parameters()):
        reg_loss = reg_loss + torch.sum((param - param_old) ** 2)
    return reg_loss


# ─── reference_dqn_update_step_cl ───────────────────────────────────────────
#
# De-globalized adaptation of ``train()`` from SARL_CL/examples_cl/maps.py
# (lines 610-829) for Tier-3 parity testing:
#
# - All globals (policy_net, target_net, second_order_net, optimizer,
#   optimizer2, scheduler1, scheduler2, teacher_*, loss_weighter_*) are
#   passed as explicit arguments.
# - ``train_or_test`` hardcoded to True (always-update for parity).
# - ``counter_list_losses`` increment + anomaly detection removed.
# - Two code paths preserved:
#     * non-CL (``previous_loss is None``): standard 1-term loss
#       (CAE for FO, BCE-with-logits for SO).
#     * CL (``previous_loss is not None``): 3-term loss per network
#       (task + distillation + feature), normalised by the shared
#       DynamicLossWeighter instances and mixed with (WEIGHT1, WEIGHT2,
#       WEIGHT3).
# - Backward order preserved: loss_second.backward(retain_graph=True) →
#   optimizer2.step() → loss.backward() → optimizer.step() → schedulers.


def reference_dqn_update_step_cl(
    sample,
    policy_net,
    target_net,
    second_order_net,
    optimizer,
    optimizer2,
    scheduler1,
    scheduler2,
    meta,
    alpha,
    cascade_iterations_1,
    cascade_iterations_2,
    *,
    teacher_first_net=None,
    teacher_second_net=None,
    loss_weighter=None,
    loss_weighter_second=None,
    lam: float = 1e-4,
):
    """Verbatim-structure CL update step, globals lifted to kwargs."""
    cl_enabled = teacher_first_net is not None

    cascade_rate_1 = float(1.0 / cascade_iterations_1)
    cascade_rate_2 = float(1.0 / cascade_iterations_2)

    comparison_out = None
    main_task_out = None
    target_task_out = None

    comparison_out_teacher = None
    main_task_out_teacher = None

    optimizer.zero_grad()
    if meta:
        optimizer2.zero_grad()

    batch_samples = transition(*zip(*sample))
    states = torch.cat(batch_samples.state)
    next_states = torch.cat(batch_samples.next_state)
    actions = torch.cat(batch_samples.action)
    rewards = torch.cat(batch_samples.reward)

    targets_wagering = target_wager(rewards, alpha)

    is_terminal = torch.cat(batch_samples.is_terminal)

    # FO forward (student cascade).
    for j in range(cascade_iterations_1):
        output_DQN_policy, h1, comparison_1, main_task_out = policy_net(
            states, main_task_out, cascade_rate_1
        )

    # Teacher FO forward (no grad).
    h1_teacher = None
    comparison_1_teacher = None
    if cl_enabled:
        with torch.no_grad():
            for j in range(cascade_iterations_1):
                _, h1_teacher, comparison_1_teacher, main_task_out_teacher = teacher_first_net(
                    states, main_task_out_teacher, cascade_rate_1
                )

    Q_s_a = output_DQN_policy.gather(1, actions)

    none_terminal_next_state_index = torch.tensor(
        [i for i, is_term in enumerate(is_terminal) if is_term == 0],
        dtype=torch.int64,
        device=device,
    )
    none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)

    Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=device)
    if len(none_terminal_next_states) != 0:
        for k in range(cascade_iterations_1):
            output_DQN_target, _, _, target_task_out = target_net(
                none_terminal_next_states, target_task_out, cascade_rate_1
            )
        Q_s_prime_a_prime[none_terminal_next_state_index] = (
            output_DQN_target.detach().max(1)[0].unsqueeze(1)
        )

    target = rewards + GAMMA * Q_s_prime_a_prime

    W = policy_net.state_dict()["fc_hidden.weight"]

    # FO loss composition (CL 3-term vs non-CL single-term).
    if cl_enabled:
        loss_task = CAE_loss(W, target, Q_s_a, h1, lam)
        loss_previous_task = compute_weight_regularization(policy_net, teacher_first_net)
        feature_loss = f.mse_loss(h1, h1_teacher)

        current_losses = {
            "task": loss_task,
            "distillation": loss_previous_task,
            "feature": feature_loss,
        }

        loss_weighter.update(current_losses)
        weighted_losses = loss_weighter.weight_losses(current_losses)

        loss = (
            WEIGHT1 * weighted_losses["task"]
            + WEIGHT2 * weighted_losses["distillation"]
            + WEIGHT3 * weighted_losses["feature"]
        )
    else:
        loss = CAE_loss(W, target, Q_s_a, h1, lam)

    # SO branch.
    if meta:
        with torch.set_grad_enabled(True):
            for i in range(cascade_iterations_2):
                output_second, comparison_out = second_order_net(
                    comparison_1, comparison_out, cascade_rate_2
                )

        if cl_enabled:
            with torch.no_grad():
                for i in range(cascade_iterations_2):
                    _, comparison_out_teacher = teacher_second_net(
                        comparison_1_teacher, comparison_out_teacher, cascade_rate_2
                    )

            task_loss_second = f.binary_cross_entropy_with_logits(
                output_second, targets_wagering
            )
            previous_task_loss_second = compute_weight_regularization(
                second_order_net, teacher_second_net
            )
            feature_loss_second = f.mse_loss(comparison_out, comparison_out_teacher)

            current_losses_second = {
                "task": task_loss_second,
                "distillation": previous_task_loss_second,
                "feature": feature_loss_second,
            }

            loss_weighter_second.update(current_losses_second)
            weighted_losses_second = loss_weighter_second.weight_losses(current_losses_second)

            loss_second = (
                WEIGHT1 * weighted_losses_second["task"]
                + WEIGHT2 * weighted_losses_second["distillation"]
                + WEIGHT3 * weighted_losses_second["feature"]
            )
        else:
            loss_second = f.binary_cross_entropy_with_logits(output_second, targets_wagering)

        # Backward order is load-bearing.
        loss_second.backward(retain_graph=True)
        optimizer2.step()

        loss.backward()
        optimizer.step()

        scheduler1.step()
        scheduler2.step()

        return loss, loss_second, Q_s_a, output_second

    loss.backward()
    optimizer.step()
    scheduler1.step()
    return loss, None, Q_s_a, None
