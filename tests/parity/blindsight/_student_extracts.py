"""Verbatim extracts from ``external/paper_reference/blindsight_tmlr.py``.

These are the reference implementations that produced the paper Table 5a
numbers. The parity tests compare our port (Sprint 12) against these
extracts bit-by-bit (Tier 1) or near-bit-exact within float32 noise
(Tier 2-4).

Each extract carries its source line range for traceability. Cosmetic
deltas (deprecated kwargs renamed, autograd ``Variable`` dropped, etc.)
are noted ; maths is byte-identical.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Verbatim generate_patterns from blindsight_tmlr.py:259-328
# ---------------------------------------------------------------------------


def student_generate_patterns(
    patterns_number: int,
    num_units: int,
    factor: int,
    condition: int,
    noise_level: float,
    *,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verbatim port of blindsight_tmlr.py:259-328.

    Returns (patterns, stim_present, stim_absent_EMPTY, order_2_pr).
    The 4th tensor is intentionally empty — student bug, preserved.
    """
    patterns_number = patterns_number * factor
    patterns_number = int(patterns_number)

    patterns: list[np.ndarray] = []
    stim_present: list[np.ndarray] = []
    stim_absent: list[np.ndarray] = []  # student bug : never appended to
    order_2_pr: list[list[float]] = []

    if condition == 0:
        random_limit, baseline, multiplier = 0.0, 0.0, 1.0
    elif condition == 1:
        random_limit, baseline, multiplier = 0.02, noise_level, 1.0
    elif condition == 2:
        random_limit, baseline, multiplier = 0.02, noise_level, 0.3
    else:
        raise ValueError(f"unknown condition {condition}")

    for i in range(patterns_number):
        if i < patterns_number // 2:
            pattern = multiplier * np.random.uniform(0.0, random_limit, num_units) + baseline
            patterns.append(pattern)
            stim_present.append(np.zeros(num_units))
            order_2_pr.append([0.0, 1.0])
        else:
            stimulus_number = random.randint(0, num_units - 1)
            pattern = np.random.uniform(0.0, random_limit, num_units) + baseline
            pattern[stimulus_number] = np.random.uniform(0.0, 1.0) * multiplier
            patterns.append(pattern)
            present = np.zeros(num_units)
            if pattern[stimulus_number] >= multiplier / 2:
                order_2_pr.append([1.0, 0.0])
                present[stimulus_number] = 1.0
            else:
                order_2_pr.append([0.0, 1.0])
                present[stimulus_number] = 0.0
            stim_present.append(present)

    patterns_t = torch.Tensor(np.asarray(patterns)).to(device).requires_grad_(True)
    stim_present_t = torch.Tensor(np.asarray(stim_present)).to(device).requires_grad_(True)
    stim_absent_t = torch.Tensor(
        np.asarray(stim_absent) if stim_absent else np.zeros((0, num_units))
    ).to(device)
    order_2_t = torch.Tensor(order_2_pr).to(device).requires_grad_(True)
    return patterns_t, stim_present_t, stim_absent_t, order_2_t


# ---------------------------------------------------------------------------
# Verbatim FirstOrderNetwork from blindsight_tmlr.py:160-207
# ---------------------------------------------------------------------------


class StudentFirstOrderNetwork(nn.Module):
    """Verbatim port of blindsight_tmlr.py:160-207 (Blindsight variant).

    Matches student exactly :
    - Encoder: Linear(100, hidden, bias=False) → ReLU → Dropout(0.1)
    - Decoder: Linear(hidden, 100, bias=False) → sigmoid
    - Weight init uniform(-1, 1) on both
    - Forward returns (h1, h2). Cascade on h2 only.

    Cosmetic deltas vs student :
    - ``use_gelu`` flag dropped (student always uses ReLU)
    - hardcoded num_units=100 (Blindsight only) — matches student behaviour
    """

    def __init__(self, hidden: int = 40, dropout_p: float = 0.1) -> None:
        super().__init__()
        num_units = 100
        self.fc1 = nn.Linear(num_units, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, num_units, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)
        nn.init.uniform_(self.fc1.weight, -1.0, 1.0)
        nn.init.uniform_(self.fc2.weight, -1.0, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        prev_h1: torch.Tensor | None = None,
        prev_h2: torch.Tensor | None = None,
        cascade_rate: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Encoder (no cascade)
        h1 = torch.relu(self.fc1(x))
        h1 = self.dropout(h1)
        del prev_h1

        # Decoder + cascade on h2
        h2 = torch.sigmoid(self.fc2(h1))
        if prev_h2 is not None:
            h2 = cascade_rate * h2 + (1 - cascade_rate) * prev_h2
        return h1, h2


# ---------------------------------------------------------------------------
# Verbatim CAE_loss from blindsight_tmlr.py:91-129
# ---------------------------------------------------------------------------


class StudentSecondOrderNetwork(nn.Module):
    """Verbatim port of blindsight_tmlr.py:213-252 (Blindsight variant).

    Matches student exactly — NO Pasquali hidden layer (student
    accepts ``hidden_2nd`` param but never instantiates the layer ;
    D.25 bug preserved here for parity).
    """

    def __init__(self, input_dim: int = 100, dropout_p: float = 0.5) -> None:
        super().__init__()
        self.wager = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout_p)
        nn.init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(
        self,
        first_order_input: torch.Tensor,
        first_order_output: torch.Tensor,
        prev_comparison: torch.Tensor | None,
        cascade_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        comparison_matrix = first_order_input - first_order_output
        comparison_out = self.dropout(comparison_matrix)
        if prev_comparison is not None:
            comparison_out = cascade_rate * comparison_out + (1 - cascade_rate) * prev_comparison
        wager = torch.sigmoid(self.wager(comparison_out))
        return wager, comparison_out


def student_cae_loss(
    W: torch.Tensor,
    x: torch.Tensor,
    recons_x: torch.Tensor,
    h: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Verbatim port of blindsight_tmlr.py:91-129.

    Note : the student trainer calls this with
    ``first_order_network.state_dict()['fc1.weight']`` which returns a
    **detached** copy (``keep_vars=False`` default). We replicate that
    detach inside the function so callers passing a live parameter
    still match the original semantics.
    """
    W = W.detach()  # match student state_dict() default behaviour
    mse_loss = nn.BCELoss(reduction="sum")
    mse = mse_loss(recons_x, x)
    dh = h * (1 - h)
    w_sum = torch.sum(W**2, dim=1)
    w_sum = w_sum.unsqueeze(1)
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)
