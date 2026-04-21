"""Low-level helpers ported from student ``onpolicy/algorithms/utils/util.py``.

Scope (E.5 lock) : only the helpers actually used by the MAPS-faithful MARL
port are kept. Dropped : ``weight_init`` (covers LSTM/Embedding/MultiHead that
we never instantiate), ``get_clones`` (used by attention extensions only).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["init", "check", "calculate_conv_params"]


def init(module: nn.Module, weight_init, bias_init, gain: float = 1.0) -> nn.Module:
    """Apply ``weight_init`` to ``module.weight`` and ``bias_init`` to ``module.bias``.

    Verbatim port of student L76-79. Used by ``Categorical`` / ``v_out`` /
    ``CNNLayer`` to combine weight initializer + bias initializer into a
    single-line helper.
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def check(x) -> torch.Tensor:
    """Convert np.ndarray to tensor if needed ; pass through otherwise.

    Verbatim port of student L86-88.
    """
    return torch.from_numpy(x) if isinstance(x, np.ndarray) else x


def calculate_conv_params(input_size: tuple[int, int, int]) -> tuple[int, int, int]:
    """Heuristic padding / stride / kernel selection preserving spatial dims.

    Verbatim port of student L91-121. For MeltingPot 11×11×3 (post-downsample)
    this yields ``(kernel=3, stride=1, padding=1)``.
    """
    height, width, _ = input_size
    kernel_size = 5 if (height > 100 or width > 100) else 3
    stride = 1
    padding = (kernel_size - 1) // 2
    return kernel_size, stride, padding
