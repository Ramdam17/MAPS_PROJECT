"""SARL (MinAtar DQN) model architectures.

Ports the two networks from ``external/MinAtar/examples/maps.py`` (commit
``ec5bcb7``) into the package conventions of ``src/maps/``:

* ``SarlQNetwork`` — conv stack + MLP hidden + Q-values + tied-weight
  reconstruction for the comparator (paper §2.1 eq.1). The tied-weight trick
  (``F.linear(Hidden, self.fc_hidden.weight.t())``) reuses ``fc_hidden`` as its
  own decoder — this is a *deliberate* choice by the paper authors to avoid
  extra parameters in the autoencoder branch. Do not replace with a standalone
  ``nn.Linear`` decoder; doing so changes parameter count and training dynamics.
* ``SarlSecondOrderNetwork`` — dropout + cascade + 2-unit linear wager head.
  Returns **raw logits** (no softmax/sigmoid), matching the paper; the
  downstream loss applies its own activation.

Both classes preserve the paper's forward-pass math exactly, so the Sprint 04b
Tier 1 parity tests (``tests/parity/sarl/test_tier1_forward.py``) can assert
``torch.allclose(ref_out, ours, atol=1e-6)`` after ``load_state_dict``.

References
----------
- Vargas et al. (2025), MAPS TMLR submission §2.1, §3.
- McClelland, J. L. (1989). Parallel distributed processing.
- Pasquali, Timmermans & Cleeremans (2010). Know thyself.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from maps.components.cascade import cascade_update


def _size_linear_unit(size: int, kernel_size: int = 3, stride: int = 1) -> int:
    """Single-axis output size of a 2D conv (no padding, no dilation)."""
    return (size - (kernel_size - 1) - 1) // stride + 1


# MinAtar grids are 10×10; the 3×3 stride-1 conv produces 8×8, then × 16 filters
# → 1024 flat features. Module-level constant matches ``maps.py:132``.
NUM_LINEAR_UNITS = _size_linear_unit(10) * _size_linear_unit(10) * 16


class SarlQNetwork(nn.Module):
    """First-order Q-network with tied-weight reconstruction branch.

    Forward returns a 4-tuple ``(q_values, hidden, comparison, hidden_copy)``:

    * ``q_values``: shape ``(B, num_actions)`` — DQN head.
    * ``hidden``: shape ``(B, 128)`` — cascade-integrated hidden activations
      (fed back as ``prev_h2`` on the next cascade iteration).
    * ``comparison``: shape ``(B, 1024)`` — reconstruction residual
      ``Input − ReLU(Input → Hidden → Input)`` used as the SecondOrder input.
    * ``hidden_copy``: same tensor as ``hidden`` (kept for API symmetry with
      the paper code, which returns Hidden twice — downstream callers can use
      either slot).

    Parameters
    ----------
    in_channels : int
        Number of MinAtar state channels for the game.
    num_actions : int
        Action space size for the game.
    """

    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        # Layer construction order matches the paper for init RNG reproducibility
        # (PyTorch draws weights sequentially from the default generator).
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        self.fc_hidden = nn.Linear(in_features=NUM_LINEAR_UNITS, out_features=128)
        self.actions = nn.Linear(in_features=128, out_features=num_actions)

    def forward(
        self,
        x: torch.Tensor,
        prev_h2: torch.Tensor | None,
        cascade_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        conv_out = F.relu(self.conv(x))  # (B, 16, 8, 8)
        flat_input = conv_out.view(conv_out.size(0), -1)  # (B, 1024)
        hidden = F.relu(self.fc_hidden(flat_input))  # (B, 128)
        hidden = cascade_update(hidden, prev_h2, cascade_rate)

        q_values = self.actions(hidden)  # (B, num_actions)

        # Tied-weight reconstruction: fc_hidden.weight.t() maps (B, 128) → (B, 1024).
        reconstruction = F.relu(F.linear(hidden, self.fc_hidden.weight.t()))
        comparison = flat_input - reconstruction

        return q_values, hidden, comparison, hidden


class SarlSecondOrderNetwork(nn.Module):
    """Second-order network: dropout + cascade + 2-unit raw-logit wager head.

    Unlike ``maps.components.second_order.SecondOrderNetwork``, this variant:
    * Accepts the comparison matrix already computed by the first-order
      network (the paper's Q-network computes ``Input − ReLU(decoder(Hidden))``
      inline), rather than computing it internally.
    * Uses 10 % dropout (``p=0.1``) — paper value, not the 0.5 shared-component
      default.
    * Returns **raw logits** (shape ``(B, 2)``) — no sigmoid/softmax applied.
      The downstream cross-entropy / MSE-on-softmax loss applies its own
      activation.
    * Uses the ``maps.components.cascade.cascade_update`` primitive — identical
      math to the paper's inline ``cascade_rate * x + (1 - cascade_rate) * prev``.

    Parameters
    ----------
    in_channels : int
        Kept for API symmetry with the paper constructor (``__init__(in_channels)``)
        even though the layer sizes are fixed at ``NUM_LINEAR_UNITS → 2``.
    dropout : float, default 0.1
        Paper value. Do not change without logging in
        ``docs/reproduction/deviations.md``.
    weight_init_range : tuple[float, float], default (0.0, 0.1)
        Uniform init range for the wager weights, matching
        ``maps.py:264`` (``init.uniform_(self.wager.weight, 0.0, 0.1)``).
    """

    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.1,
        weight_init_range: tuple[float, float] = (0.0, 0.1),
    ):
        super().__init__()
        self._in_channels = in_channels  # stored for API parity; unused in forward
        # Layer order matches paper (wager → dropout) for init RNG reproducibility.
        self.wager = nn.Linear(NUM_LINEAR_UNITS, 2)
        self.dropout = nn.Dropout(p=dropout)
        init.uniform_(self.wager.weight, *weight_init_range)

    def forward(
        self,
        comparison_matrix: torch.Tensor,
        prev_comparison: torch.Tensor | None,
        cascade_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        comparison_out = self.dropout(comparison_matrix)
        comparison_out = cascade_update(comparison_out, prev_comparison, cascade_rate)
        wager = self.wager(comparison_out)  # raw logits, shape (B, 2)
        return wager, comparison_out
