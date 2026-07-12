"""SARL+CL networks — MAPS §4 (continual learning), paper-canonical v1.

Ported verbatim from ``external/paper_reference/sarl_cl_maps.py`` (D15.1):
- ``QNetwork``           L117-158  → :class:`SarlCLQNetwork`
- ``AdaptiveQNetwork``   L160-219  → :class:`AdaptiveQNetwork`  (the CL innovation)
- ``SecondOrderNetwork`` L221-253  → :class:`SarlCLSecondOrderNetwork`

Networks are **intentionally duplicated** from ``domains/sarl`` (spec model.md,
D15.3) — the branch is self-contained. ``SarlCLQNetwork`` /
``SarlCLSecondOrderNetwork`` are the SARL v1 architecture; ``AdaptiveQNetwork``
adds a 1×1 conv adapter + channel zero-padding so a teacher trained on one game
(e.g. Breakout, 4 channels) can be continued on another (e.g. Seaquest, 10
channels) without rebuilding — the cross-game transfer backbone.

Cascade blend reuses :func:`maps.core.cascade.cascade_update`.

References
----------
Vargas et al. (2025), MAPS, TMLR submission §4, Figure 7.
Kirkpatrick et al. (2017). Overcoming catastrophic forgetting (EWC).
"""

from __future__ import annotations

import numpy
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init

from maps.core.cascade import cascade_update


def size_linear_unit(size: int, kernel_size: int = 3, stride: int = 1) -> int:
    return (size - (kernel_size - 1) - 1) // stride + 1


NUM_LINEAR_UNITS = size_linear_unit(10) * size_linear_unit(10) * 16  # = 1024


class SarlCLQNetwork(nn.Module):
    """First-order Q-network (v1) for CL. Verbatim ``QNetwork`` L117-158."""

    def __init__(self, in_channels: int, num_actions: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        self.sigmoid = nn.Sigmoid()  # paper-code artifact, never called (kept, no params)
        self.fc_hidden = nn.Linear(in_features=NUM_LINEAR_UNITS, out_features=128)
        self.fc_output = nn.Linear(in_features=128, out_features=NUM_LINEAR_UNITS)
        self.actions = nn.Linear(in_features=NUM_LINEAR_UNITS, out_features=num_actions)

    def forward(
        self, x: Tensor, prev_h2: Tensor | None, cascade_rate: float
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = F.relu(self.conv(x))
        inp = x.view(x.size(0), -1)
        hidden = F.relu(self.fc_hidden(inp))
        output = F.relu(self.fc_output(hidden))
        output = cascade_update(output, prev_h2, cascade_rate)
        q = self.actions(output)
        comparison = inp - output
        return q, hidden, comparison, output


class AdaptiveQNetwork(nn.Module):
    """Channel-adaptive first-order Q-network. Verbatim ``AdaptiveQNetwork`` L160-219.

    A 1×1 conv adapter (``max_input_channels → max_input_channels``) precedes the
    main conv; inputs with fewer channels are zero-padded up to
    ``max_input_channels`` in :meth:`adapt_input` — enabling cross-game transfer.
    """

    def __init__(self, max_input_channels: int, num_actions: int) -> None:
        super().__init__()
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

    def _get_conv_output_size(self, shape: tuple[int, int, int]) -> int:
        # torch.rand probe (verbatim source) — consumes global RNG at build time.
        bs = 1
        probe = torch.rand(bs, *shape)
        out = self.conv(probe)
        return int(numpy.prod(out.size()[1:]))

    def adapt_input(self, x: Tensor) -> Tensor:
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

    def forward(
        self, x: Tensor, prev_h2: Tensor | None, cascade_rate: float
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.adapt_input(x)
        x = self.input_adapter(x)
        x = F.relu(self.conv(x))
        inp = x.view(x.size(0), -1)
        hidden = F.relu(self.fc_hidden(inp))
        output = F.relu(self.fc_output(hidden))
        output = cascade_update(output, prev_h2, cascade_rate)
        q = self.actions(output)
        comparison = inp - output
        return q, hidden, comparison, output


class SarlCLSecondOrderNetwork(nn.Module):
    """Second-order comparator + wager (v1). Verbatim ``SecondOrderNetwork`` L221-253."""

    def __init__(self, in_channels: int | None = None) -> None:
        super().__init__()
        self.comparison_layer = nn.Linear(NUM_LINEAR_UNITS, NUM_LINEAR_UNITS)
        self.wager = nn.Linear(NUM_LINEAR_UNITS, 2)
        self.dropout = nn.Dropout(p=0.1)
        self._init_weights()

    def _init_weights(self) -> None:
        init.uniform_(self.comparison_layer.weight, -1.0, 1.0)
        init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(
        self, comparison_matrix: Tensor, prev_comparison: Tensor | None, cascade_rate: float
    ) -> tuple[Tensor, Tensor]:
        comparison_out = self.dropout(F.relu(self.comparison_layer(comparison_matrix)))
        comparison_out = cascade_update(comparison_out, prev_comparison, cascade_rate)
        wager = self.wager(comparison_out)
        return wager, comparison_out
