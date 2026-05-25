"""Low-level helpers ported from student ``onpolicy/algorithms/utils/util.py``
and ``onpolicy/utils/util.py``.

Scope (E.5 lock) : only the helpers actually used by the MAPS-faithful MARL
port are kept. Dropped : ``weight_init`` (covers LSTM/Embedding/MultiHead that
we never instantiate), ``get_clones`` (used by attention extensions only),
``get_shape_from_act_space`` extras (MultiDiscrete/MultiBinary/Box paths —
port is Discrete-only per E.5), ``tile_images`` (rendering helper, not needed).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "init",
    "check",
    "calculate_conv_params",
    "huber_loss",
    "mse_loss",
    "get_grad_norm",
    "update_linear_schedule",
    "get_shape_from_obs_space",
]


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


def huber_loss(e: torch.Tensor, d: float) -> torch.Tensor:
    """Verbatim port of student ``onpolicy/utils/util.py:23-26``.

    Quadratic below threshold ``d``, linear above. Common in MAPPO value loss.
    """
    a = (torch.abs(e) <= d).float()
    b = (torch.abs(e) > d).float()
    return a * e**2 / 2 + b * d * (torch.abs(e) - d / 2)


def mse_loss(e: torch.Tensor) -> torch.Tensor:
    """Half-squared error (student ``utils/util.py:28-29``)."""
    return e**2 / 2


def get_grad_norm(params) -> float:
    """Port of student ``get_gard_norm`` (L9-15) — typo-fixed name."""
    sum_sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        sum_sq += (p.grad.norm() ** 2).item()
    return math.sqrt(sum_sq)


def update_linear_schedule(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    initial_lr: float,
) -> None:
    """Linear LR decay (student ``utils/util.py:17-21``)."""
    lr = initial_lr - (initial_lr * (epoch / float(total_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_shape_from_obs_space(obs_space) -> tuple[int, ...]:
    """Extract shape tuple from gym(nasium) Box observation space."""
    if obs_space.__class__.__name__ == "Box":
        return tuple(obs_space.shape)
    if isinstance(obs_space, (tuple, list)):
        return tuple(obs_space)
    raise NotImplementedError(
        f"Unsupported obs_space type {obs_space.__class__.__name__}. Port supports Box."
    )
