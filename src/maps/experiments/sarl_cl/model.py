"""SARL+CL (MinAtar Continual-Learning DQN) model architectures.

Ports the three networks from ``SARL_CL/examples_cl/maps.py`` (commit
``ec5bcb7``) faithfully — they differ from the standard SARL nets in
:mod:`maps.experiments.sarl.model` in ways that matter numerically:

1. ``SarlCLQNetwork`` — has an **explicit** ``fc_output`` decoder (no tied
   weights), and the cascade integration happens on the **output**
   activation (1024-dim) rather than the hidden activation (128-dim).
   The reconstruction branch is the untied ``fc_output(Hidden)`` path.

2. ``SarlCLSecondOrderNetwork`` — adds an explicit ``comparison_layer``
   (not present in standard SARL) before the wager head, and uses
   uniform-in-[-1, 1] init for that layer. Like the SARL variant it
   returns **raw logits** (the author defines ``softmax`` and ``sigmoid``
   attributes but never calls them in ``forward``).

3. ``AdaptiveQNetwork`` — same topology as ``SarlCLQNetwork`` but with a
   1×1 input-adapter conv prepended and a zero-padding pre-step for
   variable-channel inputs (enables cross-game transfer, which MinAtar
   games need because their channel counts differ: Space Invaders 6,
   Breakout 4, Freeway 7, etc.).

Do **not** refactor to share layers with :mod:`maps.experiments.sarl.model`.
The architectures genuinely differ; silent unification would change the
numbers the paper's CL experiments reported.

References
----------
- SARL_CL/examples_cl/maps.py:117-219 (source of the three classes).
- Vargas et al. (2025), MAPS TMLR submission §4 (continual learning).
- McClelland, J. L. (1989). Parallel distributed processing.
- Pasquali, Timmermans & Cleeremans (2010). Know thyself.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from maps.components.cascade import cascade_update


def _size_linear_unit(size: int, kernel_size: int = 3, stride: int = 1) -> int:
    """Single-axis output size of a 2D conv (no padding, no dilation)."""
    return (size - (kernel_size - 1) - 1) // stride + 1


# MinAtar grids are 10×10; with kernel=3 stride=1 that's 8×8, ×16 filters = 1024.
NUM_LINEAR_UNITS = _size_linear_unit(10) * _size_linear_unit(10) * 16


# ── SarlCLQNetwork ──────────────────────────────────────────────────────────


class SarlCLQNetwork(nn.Module):
    """First-order Q-network with explicit decoder (NOT tied weights).

    Differences from :class:`maps.experiments.sarl.model.SarlQNetwork`
    (standard SARL variant):

    +--------------------+------------------------------+----------------------------+
    |                    | Standard SARL                | SARL+CL (this class)       |
    +====================+==============================+============================+
    | Decoder            | tied: ``fc_hidden.weight.T`` | separate ``fc_output``     |
    | Cascade applied to | ``Hidden`` (128-dim)         | ``Output`` (1024-dim)      |
    | ``actions``        | 128 → num_actions            | 1024 → num_actions         |
    | Comparison         | Input − ReLU(tied_reco)      | Input − Output             |
    +--------------------+------------------------------+----------------------------+

    Forward returns a 4-tuple ``(q_values, hidden, comparison, output)``:

    * ``q_values`` — (B, num_actions), fed from ``actions(Output)``.
    * ``hidden`` — (B, 128), the 128-dim bottleneck activation.
    * ``comparison`` — (B, 1024), ``Input − Output`` reconstruction residual.
    * ``output`` — (B, 1024), cascade-integrated decoder output (this is
      what's fed back as ``prev_h2`` on the next cascade iteration).

    Parameters
    ----------
    in_channels : int
        Number of MinAtar state channels for the game.
    num_actions : int
        Action-space cardinality.
    """

    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        # Construction order matches the paper for init-RNG reproducibility.
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        # ``sigmoid`` is a paper-code artifact — defined but never called.
        # Kept as an attribute to preserve state_dict key parity if we
        # ever load a legacy checkpoint.
        self.sigmoid = nn.Sigmoid()
        self.fc_hidden = nn.Linear(NUM_LINEAR_UNITS, 128)
        self.fc_output = nn.Linear(128, NUM_LINEAR_UNITS)
        self.actions = nn.Linear(NUM_LINEAR_UNITS, num_actions)

    def forward(
        self,
        x: torch.Tensor,
        prev_h2: torch.Tensor | None,
        cascade_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        conv_out = F.relu(self.conv(x))  # (B, 16, 8, 8)
        flat_input = conv_out.view(conv_out.size(0), -1)  # (B, 1024)
        hidden = F.relu(self.fc_hidden(flat_input))  # (B, 128)
        output = F.relu(self.fc_output(hidden))  # (B, 1024)
        output = cascade_update(output, prev_h2, cascade_rate)

        q_values = self.actions(output)  # (B, num_actions)
        comparison = flat_input - output

        return q_values, hidden, comparison, output


# ── SarlCLSecondOrderNetwork ────────────────────────────────────────────────


class SarlCLSecondOrderNetwork(nn.Module):
    """Second-order network with explicit comparison layer (CL variant).

    Differences from :class:`maps.experiments.sarl.model.SarlSecondOrderNetwork`:

    * Adds a ``comparison_layer`` (Linear 1024→1024) with ``ReLU + Dropout``
      applied BEFORE the cascade integration. Standard SARL feeds the
      comparison matrix directly into cascade → wager.
    * Weight init: ``comparison_layer`` uniform in ``[-1, 1]`` (paper line
      239); ``wager`` uniform in ``[0, 0.1]`` (same as standard SARL).

    Both variants return raw logits ``(B, 2)`` for the wager; the paper's
    ``softmax`` / ``sigmoid`` attributes are never invoked in forward.

    Parameters
    ----------
    in_channels : int
        Kept for API symmetry with the paper constructor
        (``__init__(in_channels)``) even though the fixed layer sizes
        (1024 → 1024 → 2) don't depend on it.
    dropout : float, default 0.1
        Paper value — do not change without logging in
        ``docs/reproduction/deviations.md``.
    """

    def __init__(self, in_channels: int, dropout: float = 0.1):
        super().__init__()
        self._in_channels = in_channels  # API parity; unused in forward
        # Construction order matches the paper (comparison_layer → wager).
        self.comparison_layer = nn.Linear(NUM_LINEAR_UNITS, NUM_LINEAR_UNITS)
        self.wager = nn.Linear(NUM_LINEAR_UNITS, 2)
        self.dropout = nn.Dropout(p=dropout)
        # Author defines these but never calls them — kept for state_dict
        # parity. DO NOT invoke in forward.
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self) -> None:
        """Paper-order init: comparison_layer in [-1, 1], wager in [0, 0.1]."""
        init.uniform_(self.comparison_layer.weight, -1.0, 1.0)
        init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(
        self,
        comparison_matrix: torch.Tensor,
        prev_comparison: torch.Tensor | None,
        cascade_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Paper line 245: ``dropout(f.relu(self.comparison_layer(comparison_matrix)))``
        comparison_out = self.dropout(F.relu(self.comparison_layer(comparison_matrix)))
        comparison_out = cascade_update(comparison_out, prev_comparison, cascade_rate)
        wager = self.wager(comparison_out)  # raw logits (B, 2)
        return wager, comparison_out


# ── AdaptiveQNetwork ────────────────────────────────────────────────────────


class AdaptiveQNetwork(nn.Module):
    """Variable-channel first-order Q-network for cross-game transfer.

    Prepends a 1×1 convolution + ReLU "input adapter" to the standard
    ``SarlCLQNetwork`` topology, and zero-pads the channel axis when the
    input has fewer channels than ``max_input_channels``. Useful when the
    curriculum is Space Invaders (6) → Breakout (4) → Freeway (7) etc.

    Forward contract matches :class:`SarlCLQNetwork` — returns
    ``(q_values, hidden, comparison, output)``.

    Parameters
    ----------
    max_input_channels : int
        Upper bound on the number of MinAtar state channels across the
        curriculum games. Inputs with fewer channels are zero-padded.
    num_actions : int
        Action-space cardinality (paper uses the max across curriculum games).
    """

    def __init__(self, max_input_channels: int, num_actions: int):
        super().__init__()
        self.max_input_channels = max_input_channels

        self.input_adapter = nn.Sequential(
            nn.Conv2d(max_input_channels, max_input_channels, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.conv = nn.Conv2d(max_input_channels, 16, kernel_size=3, stride=1)

        # Paper computes conv output size dynamically via a dummy forward —
        # we reproduce that exactly (``numpy.prod(output.size()[1:])``) so
        # any future change to the kernel/stride flows through automatically.
        conv_output_size = self._get_conv_output_size((max_input_channels, 10, 10))

        self.fc_hidden = nn.Linear(conv_output_size, 128)
        self.fc_output = nn.Linear(128, conv_output_size)
        self.actions = nn.Linear(conv_output_size, num_actions)

    def _get_conv_output_size(self, shape: tuple[int, int, int]) -> int:
        """Probe the conv layer with a dummy input to get its flat size."""
        dummy = torch.rand(1, *shape)
        with torch.no_grad():
            out = self.conv(dummy)
        return int(np.prod(out.size()[1:]))

    def adapt_input(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-pad channels on the right when input has fewer than max."""
        if x.size(1) < self.max_input_channels:
            padding = torch.zeros(
                x.size(0),
                self.max_input_channels - x.size(1),
                x.size(2),
                x.size(3),
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, padding], dim=1)
        return x

    def forward(
        self,
        x: torch.Tensor,
        prev_h2: torch.Tensor | None,
        cascade_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.adapt_input(x)
        x = self.input_adapter(x)
        conv_out = F.relu(self.conv(x))
        flat_input = conv_out.view(conv_out.size(0), -1)
        hidden = F.relu(self.fc_hidden(flat_input))
        output = F.relu(self.fc_output(hidden))
        output = cascade_update(output, prev_h2, cascade_rate)
        q_values = self.actions(output)
        comparison = flat_input - output
        return q_values, hidden, comparison, output
