"""MARL CNN encoder — MAPS §5 (MeltingPot MAPPO).

Ported verbatim from ``external/paper_reference/marl_tmlr/onpolicy/algorithms/utils/cnn.py``
(``CNNLayer`` / ``CNNBase``), the RGB-observation encoder the MeltingPot actor/critic
use. conv → BatchNorm → act → flatten → Linear → LayerNorm → act → Linear → LayerNorm
→ act. Input is scaled by 1/255 and permuted (B,H,W,C)→(B,C,H,W). The conv uses
**valid padding** (calculate_conv_params' padding is discarded — faithful to source).

References
----------
Vargas et al. (2025), MAPS, TMLR submission §5, Table 7.
"""

from __future__ import annotations

from torch import Tensor, nn

from maps.domains.marl.util import calculate_conv_params, init


class Flatten(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.size(0), -1)


class CNNLayer(nn.Module):
    """Verbatim ``CNNLayer`` (cnn.py:26-71)."""

    def __init__(self, obs_shape, hidden_size: int, use_orthogonal: bool, use_relu: bool) -> None:
        super().__init__()
        active_func = [nn.Tanh(), nn.ReLU()][use_relu]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_relu])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        if obs_shape[0] == 3:
            input_channel, input_width, input_height = obs_shape[0], obs_shape[1], obs_shape[2]
        else:  # obs_shape[2] == 3 (H, W, C)
            input_channel, input_width, input_height = obs_shape[2], obs_shape[0], obs_shape[1]

        kernel_size, stride, _padding = calculate_conv_params(
            (input_width, input_height, input_channel)
        )

        self.cnn = nn.Sequential(
            init_(
                nn.Conv2d(input_channel, hidden_size // 2, kernel_size=kernel_size, stride=stride)
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

    def forward(self, x: Tensor) -> Tensor:
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)  # (B,H,W,C) → (B,C,H,W)
        return self.cnn(x)


class CNNBase(nn.Module):
    """Verbatim ``CNNBase`` (cnn.py:73-85)."""

    def __init__(
        self, obs_shape, hidden_size: int, use_orthogonal: bool = True, use_relu: bool = True
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cnn = CNNLayer(obs_shape, hidden_size, use_orthogonal, use_relu)

    def forward(self, x: Tensor) -> Tensor:
        return self.cnn(x)
