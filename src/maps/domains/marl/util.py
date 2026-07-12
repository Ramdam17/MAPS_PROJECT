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
