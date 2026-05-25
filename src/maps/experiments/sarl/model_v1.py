"""SARL (MinAtar DQN) model architectures — **v1 (paper-canonical)**.

Ports the two networks from ``external/paper_reference/sarl/maps_v1.py``
(Juan's fork, pre-commit ``8aa1138`` ; restored 2026-05-24 by commit
``6c59744``) into the package conventions of ``src/maps/``.

**Why v1 and not v2?**
``SARL_Training_Standard.sh`` (the only shell that produced the paper's
Table 6 numbers) invokes ``maps_v1.py`` with ``base=2000000`` frames and
``-ema 25`` (α=0.25). The Sprint-04b port (``model.py``, classes
``SarlQNetwork`` and ``SarlSecondOrderNetwork``) targets ``maps_v2.py``,
justified by TD-008 — a token-presence comparison that wrongly concluded
"v2 adds only ``count_parameters`` and ``curriculum``". The variants are
structurally distinct on load-bearing pieces : Q-head input dim (1024 vs
128), reconstruction decoder (dedicated ``Linear(128, 1024)`` vs
tied-weight ``F.linear(Hidden, fc_hidden.weight.t())``), and
``comparison_layer`` (active 1024×1024 vs commented-out). See
``docs/reproduction/sarl-v1-vs-v2.md`` and deviations.md
``D-sarl-wrong-variant``.

**What's in this module:**

* :class:`SarlQNetworkV1` — conv stack + MLP hidden + **dedicated decoder**
  + Q-head from the **1024-dim Output** (post-decoder, post-ReLU). Cascade
  is applied to Output, not Hidden. Forward returns
  ``(q_values, hidden, comparison, output)`` where the 4th slot is the
  cascade-integrated Output (1024-dim) — distinct from v2's 4th slot.
* :class:`SarlSecondOrderNetworkV1` — **active comparison_layer** (1024×1024,
  uniform init in [-1.0, 1.0]) followed by ReLU, dropout, cascade, and
  the 2-unit raw-logit wager head.

Forward-pass parity with the v1 reference (``external/paper_reference/
sarl/maps_v1.py``) is asserted bit-exact at ``atol=1e-6`` after
``load_state_dict`` in :mod:`tests.parity.sarl.test_sarl_v1_parity`.

References
----------
- Vargas et al. (2025), MAPS TMLR submission §2.1, §3 (paper canonical).
- Pasquali, Timmermans & Cleeremans (2010). Know thyself (eq.3 wager head).
- McClelland, J. L. (1989). Parallel distributed processing (cascade eq.).
- ``docs/reproduction/sarl-v1-vs-v2.md`` (full v1 vs v2 architectural diff).
- ``docs/sprints/sprint-09-sarl-v1-port-and-ruff-sweep.md`` (Phase 9.1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from maps.components.cascade import cascade_update
from maps.experiments.sarl.model import NUM_LINEAR_UNITS


class SarlQNetworkV1(nn.Module):
    """First-order Q-network — **v1 paper-canonical** variant.

    The forward pass mirrors :class:`SarlQNetwork` (v2) except:

    1. A **dedicated decoder** ``fc_output = nn.Linear(128, 1024)`` replaces
       v2's tied-weight ``F.linear(Hidden, fc_hidden.weight.t())``. The
       decoder's bias serves as paper eq.12's ``b_recon``, so no extra
       parameter is needed (v2 had to add ``b_recon`` separately).
    2. The **cascade is applied to Output (1024-dim, post-decoder)**, not
       Hidden (128-dim). The ``prev_h2`` argument is therefore the
       previous Output, not the previous Hidden.
    3. The **Q-head reads from Output (1024-dim)** via
       ``actions = nn.Linear(num_linear_units, num_actions)`` — v2 reads
       from Hidden (128-dim).
    4. The forward returns ``(q_values, hidden, comparison, output)``
       where the 4th slot is **Output (cascade-integrated, 1024-dim)** —
       v2's 4th slot is Hidden (cascade-integrated, 128-dim).

    Forward returns a 4-tuple ``(q_values, hidden, comparison, output)``:

    * ``q_values``: shape ``(B, num_actions)`` — DQN head from 1024-dim Output.
    * ``hidden``: shape ``(B, 128)`` — encoder activations (NOT cascade-integrated
      in v1; only Output is cascade-integrated).
    * ``comparison``: shape ``(B, 1024)`` — reconstruction residual
      ``Input − Output`` used as the SecondOrder input.
    * ``output``: shape ``(B, 1024)`` — cascade-integrated decoder output,
      fed back as ``prev_h2`` on the next cascade iteration.

    Parameters
    ----------
    in_channels : int
        Number of MinAtar state channels for the game.
    num_actions : int
        Action space size for the game.

    Notes
    -----
    **Cascade no-op (D-sarl-cascade-noop)** — same as v2. The forward has
    no dropout, so calling it ``cascade_iterations_1`` times with the
    same input produces identical Output every iteration; 50 iters ≡
    1 iter mathematically. The paper value (50) is kept for parity with
    Juan's reference. Post-reproduction, dropout in the decoder branch
    could be added to make the cascade do real averaging.
    """

    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        # Layer construction order matches the paper reference for init RNG
        # reproducibility (PyTorch draws weights sequentially from the default
        # generator). Order: conv → fc_hidden → fc_output → actions.
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        self.fc_hidden = nn.Linear(in_features=NUM_LINEAR_UNITS, out_features=128)
        # v1's dedicated decoder — replaces v2's tied-weight + b_recon trick.
        # The decoder's bias (default bias=True on nn.Linear) serves as paper
        # eq.12's b_recon term.
        self.fc_output = nn.Linear(in_features=128, out_features=NUM_LINEAR_UNITS)
        # Q-head from the 1024-dim Output (v1) vs from 128-dim Hidden (v2).
        self.actions = nn.Linear(in_features=NUM_LINEAR_UNITS, out_features=num_actions)

    def forward(
        self,
        x: torch.Tensor,
        prev_h2: torch.Tensor | None,
        cascade_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        conv_out = F.relu(self.conv(x))  # (B, 16, 8, 8)
        flat_input = conv_out.view(conv_out.size(0), -1)  # (B, 1024)
        hidden = F.relu(self.fc_hidden(flat_input))  # (B, 128)
        output = F.relu(self.fc_output(hidden))  # (B, 1024) — dedicated decoder

        # Cascade is applied to Output (1024-dim), not Hidden. ``prev_h2`` is
        # therefore the previous Output, not the previous Hidden — the
        # parameter name is preserved across v1 and v2 for API symmetry but
        # its semantic differs.
        output = cascade_update(output, prev_h2, cascade_rate)

        q_values = self.actions(output)  # (B, num_actions) — Q from 1024-dim Output
        comparison = flat_input - output  # (B, 1024) — paper §2.1 eq.1

        return q_values, hidden, comparison, output


class SarlSecondOrderNetworkV1(nn.Module):
    """Second-order network — **v1 paper-canonical** variant.

    Differences from :class:`SarlSecondOrderNetwork` (v2):

    1. The **comparison_layer is active** : ``nn.Linear(1024, 1024)``
       (~1.05M parameters) initialized uniform in [-1.0, 1.0]. v2 has it
       commented out (no linear, only dropout).
    2. Forward pipeline: ``dropout(ReLU(comparison_layer(comparison_matrix)))``
       — full linear+ReLU+dropout chain. v2 forwards just
       ``dropout(comparison_matrix)``.

    The wager head and cascade behaviour are identical to v2 (2-unit
    raw-logit head, ``cascade_update`` on comparison_out).

    Parameters
    ----------
    in_channels : int
        Kept for API symmetry with the paper constructor
        (``__init__(in_channels)``) even though the layer sizes are fixed
        at ``NUM_LINEAR_UNITS → NUM_LINEAR_UNITS → 2``.
    dropout : float, default 0.1
        Paper value (``maps_v1.py:230``). Do not change without logging
        in ``docs/reproduction/deviations.md``.
    comparison_init_range : tuple[float, float], default (-1.0, 1.0)
        Uniform init range for the comparison_layer weights, matching
        ``maps_v1.py:239`` (``init.uniform_(self.comparison_layer.weight, -1.0, 1.0)``).
    wager_init_range : tuple[float, float], default (0.0, 0.1)
        Uniform init range for the wager weights, matching
        ``maps_v1.py:240`` (``init.uniform_(self.wager.weight, 0.0, 0.1)``).
    """

    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.1,
        comparison_init_range: tuple[float, float] = (-1.0, 1.0),
        wager_init_range: tuple[float, float] = (0.0, 0.1),
    ):
        super().__init__()
        self._in_channels = in_channels  # stored for API parity; unused in forward
        # Layer construction order matches paper for init RNG reproducibility:
        # comparison_layer → wager → dropout.
        self.comparison_layer = nn.Linear(
            in_features=NUM_LINEAR_UNITS, out_features=NUM_LINEAR_UNITS
        )
        self.wager = nn.Linear(NUM_LINEAR_UNITS, 2)
        self.dropout = nn.Dropout(p=dropout)
        init.uniform_(self.comparison_layer.weight, *comparison_init_range)
        init.uniform_(self.wager.weight, *wager_init_range)

    def forward(
        self,
        comparison_matrix: torch.Tensor,
        prev_comparison: torch.Tensor | None,
        cascade_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # v1: full linear+ReLU+dropout pipeline (vs v2's bare dropout).
        comparison_out = self.dropout(F.relu(self.comparison_layer(comparison_matrix)))
        comparison_out = cascade_update(comparison_out, prev_comparison, cascade_rate)
        wager = self.wager(comparison_out)  # raw logits, shape (B, 2)
        return wager, comparison_out
