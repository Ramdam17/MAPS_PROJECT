"""Observation encoder for MARL — paper Fig.4 ConvEncoder.

Ports student ``onpolicy/algorithms/utils/cnn.py`` ``CNNBase`` + ``CNNLayer``
(L26-85). Handles MeltingPot RGB observations (11×11×3 after 8× downsample).

Pipeline (paper Fig.4 + student L48-66) :
1. Normalize RGB : ``x / 255.0`` (converts uint8 → float32 in [0, 1]).
2. Permute ``(H, W, C) → (C, H, W)`` for PyTorch Conv2d.
3. ``Conv2d(C, hidden//2, kernel=3, stride=1, padding=0)`` + BatchNorm2d + ReLU.
4. ``Flatten`` + ``Linear(flat, hidden) + LayerNorm + ReLU``.
5. ``Linear(hidden, hidden) + LayerNorm + ReLU``.

Output : ``(batch, hidden_size)``. The extra 2 Linear+LayerNorm layers
(post-conv) come from student L58-65 and are not individually in paper Fig.4,
but the overall ConvEncoder block is.

Dropped from student (E.5 scope lock) :
- ``MLPBase`` (1D obs path) : MeltingPot obs is always 3D RGB.
- ``Encoder`` / ``ResidualBlock`` / perceiver stuff (L88+) : unused by MAPPO.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from maps.experiments.marl.util import calculate_conv_params, init

__all__ = ["CNNBase", "CNNLayer"]


class _Flatten(nn.Module):
    """Batch-preserving flatten, verbatim student L9-11."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.size(0), -1)


class CNNLayer(nn.Module):
    """Student ``CNNLayer`` L26-66 — the conv + 2×linear stack.

    Parameters
    ----------
    obs_shape : tuple[int, int, int]
        RGB observation shape. Supports ``(C, H, W)`` or ``(H, W, C)`` — we
        auto-detect via the student heuristic (``obs_shape[0] == 3`` → CHW).
    hidden_size : int
        Output feature dimension. Paper T.12 = 100.
    use_orthogonal : bool, default True
        Orthogonal weight init (vs xavier_uniform).
    use_ReLU : bool, default True
        ReLU activation (vs Tanh).
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        hidden_size: int,
        use_orthogonal: bool = True,
        use_ReLU: bool = True,
    ):
        super().__init__()
        active_func = nn.ReLU() if use_ReLU else nn.Tanh()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][int(use_orthogonal)]
        gain = nn.init.calculate_gain("relu" if use_ReLU else "tanh")

        def init_(m: nn.Module) -> nn.Module:
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        # Auto-detect channel position (student L40-46).
        if obs_shape[0] == 3:
            input_channel, input_width, input_height = obs_shape
        elif obs_shape[2] == 3:
            input_width, input_height, input_channel = obs_shape
        else:
            raise ValueError(f"Cannot infer RGB channel position in obs_shape={obs_shape}")

        kernel_size, stride, _padding = calculate_conv_params((input_width, input_height, input_channel))

        conv_out_w = input_width - kernel_size + stride
        conv_out_h = input_height - kernel_size + stride

        self.cnn = nn.Sequential(
            init_(
                nn.Conv2d(
                    in_channels=input_channel,
                    out_channels=hidden_size // 2,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            ),
            nn.BatchNorm2d(hidden_size // 2),
            active_func,
            _Flatten(),
            init_(
                nn.Linear(
                    hidden_size // 2 * conv_out_w * conv_out_h,
                    hidden_size,
                )
            ),
            nn.LayerNorm(hidden_size),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.LayerNorm(hidden_size),
            active_func,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input : ``(B, H, W, C)`` uint8 or float. Output : ``(B, hidden_size)``."""
        x = x / 255.0
        # If the input is HWC, permute to CHW for Conv2d. Student L71 ``permute(0, 3, 1, 2)``.
        if x.dim() == 4 and x.size(-1) == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        return self.cnn(x)


class CNNBase(nn.Module):
    """Paper Fig.4 ConvEncoder. Thin wrapper around :class:`CNNLayer`."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        hidden_size: int = 100,
        use_orthogonal: bool = True,
        use_ReLU: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cnn = CNNLayer(obs_shape, hidden_size, use_orthogonal, use_ReLU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)
