"""Verbatim copy of student ``cnn.py`` — ``Flatten`` + ``CNNLayer`` + ``CNNBase``
(non-attention, no Encoder / ResidualBlock)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .util import calculate_conv_params, init


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class CNNLayer(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal, use_ReLU):
        super().__init__()
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        if obs_shape[0] == 3:
            input_channel = obs_shape[0]
            input_width = obs_shape[1]
            input_height = obs_shape[2]
        elif obs_shape[2] == 3:
            input_channel = obs_shape[2]
            input_width = obs_shape[0]
            input_height = obs_shape[1]
        else:
            raise ValueError(f"Unexpected obs_shape {obs_shape}")

        kernel_size, stride, _padding = calculate_conv_params(
            (input_width, input_height, input_channel)
        )

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
            Flatten(),
            init_(
                nn.Linear(
                    hidden_size
                    // 2
                    * (input_width - kernel_size + stride)
                    * (input_height - kernel_size + stride),
                    hidden_size,
                )
            ),
            nn.LayerNorm(hidden_size),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.LayerNorm(hidden_size),
            active_func,
        )

    def forward(self, x):
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
        return self.cnn(x)


class CNNBase(nn.Module):
    """Student ``CNNBase`` : thin wrapper around :class:`CNNLayer`.

    Student reads ``args.use_orthogonal`` / ``args.use_ReLU`` / ``args.hidden_size``.
    We accept the plain fields instead to keep the ref dep-light.
    """

    def __init__(self, obs_shape, hidden_size, use_orthogonal, use_ReLU):
        super().__init__()
        self._use_orthogonal = use_orthogonal
        self._use_ReLU = use_ReLU
        self.hidden_size = hidden_size
        self.cnn = CNNLayer(obs_shape, hidden_size, use_orthogonal, use_ReLU)

    def forward(self, x):
        return self.cnn(x)
