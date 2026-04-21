"""Observation encoder for MARL — paper Fig.4 ConvEncoder (E.7 scaffold).

Ports student `onpolicy/algorithms/utils/cnn.py` ``CNNBase`` + ``CNNLayer``
(paper Fig.4). Operates on MeltingPot RGB observations (11×11×3 after 8x
downsample, or 88×88×3 at original resolution).

Output : flat feature vector of size ``hidden_size`` (paper T.12 = 100).

Empty module : implementation is E.8's scope. See
``docs/reviews/marl-maps-additions.md`` §(i) for the target architecture.
"""

from __future__ import annotations

import torch.nn as nn

__all__ = ["CNNBase"]


class CNNBase(nn.Module):
    """Paper Fig.4 ConvEncoder — to be implemented in E.8.

    Pipeline :
    1. Normalize RGB to [0, 1] via division by 255.
    2. Conv2d → BatchNorm2d → ReLU.
    3. Flatten.
    4. Linear → LayerNorm → ReLU.
    5. Linear → LayerNorm → ReLU.

    Output shape : ``(batch, hidden_size)``. Matches paper Fig.4 where the
    encoder output feeds the GRU input.
    """

    def __init__(self, obs_shape: tuple[int, int, int], hidden_size: int = 100):
        super().__init__()
        raise NotImplementedError("E.8 will implement this.")

    def forward(self, obs):  # pragma: no cover — stub
        raise NotImplementedError("E.8 will implement this.")
