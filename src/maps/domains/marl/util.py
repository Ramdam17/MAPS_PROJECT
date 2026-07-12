"""MARL small helpers — MAPS §5.

Ported verbatim from ``external/paper_reference/marl_tmlr/onpolicy/algorithms/utils/util.py``:
``init`` (layer init), ``check`` (numpy→tensor), ``calculate_conv_params`` (kernel/
stride/padding heuristic — the padding is computed but the CNN uses valid padding,
faithful to the source), ``get_grad_norm``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def init(module: nn.Module, weight_init, bias_init, gain: float = 1.0) -> nn.Module:
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def check(x):
    return torch.from_numpy(x) if isinstance(x, np.ndarray) else x


def calculate_conv_params(input_size: tuple[int, int, int]) -> tuple[int, int, int]:
    """(kernel_size, stride, padding) heuristic. Verbatim source.

    Note: the CNN layer uses this kernel/stride but NOT the padding (valid conv) —
    faithful to the source (padding is computed then discarded).
    """
    height, width, _channels = input_size
    kernel_size = 5 if (height > 100 or width > 100) else 3
    stride = 1
    padding = (kernel_size - 1) // 2
    return kernel_size, stride, padding


def get_grad_norm(parameters) -> float:
    """L2 norm of gradients over parameters (MAPPO get_grad_norm)."""
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float((p.grad.detach() ** 2).sum().item())
    return total**0.5


def update_linear_schedule(optimizer, epoch: int, total_num_epochs: int, initial_lr: float) -> None:
    """Verbatim ``update_linear_schedule`` (onpolicy/utils/util.py:17-21) — linear LR decay."""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def huber_loss(e, d):
    """Verbatim ``huber_loss`` (onpolicy/utils/util.py:23-26)."""
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    """Verbatim ``mse_loss`` (onpolicy/utils/util.py:28-29)."""
    return e**2 / 2


def get_shape_from_obs_space(obs_space) -> tuple[int, ...]:
    """Verbatim ``get_shape_from_obs_space`` (onpolicy/utils/util.py:31-39)."""
    if obs_space.__class__.__name__ == "Box":
        return obs_space.shape
    if obs_space.__class__.__name__ == "list":
        return obs_space
    raise NotImplementedError


def get_shape_from_act_space(act_space) -> int:
    """Verbatim ``get_shape_from_act_space`` (onpolicy/utils/util.py:41-52), Discrete path."""
    if act_space.__class__.__name__ == "Discrete":
        return 1
    if act_space.__class__.__name__ == "MultiDiscrete":
        return act_space.shape
    if act_space.__class__.__name__ in ("Box", "MultiBinary"):
        return act_space.shape[0]
    return act_space[0].shape[0] + 1
