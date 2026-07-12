"""MARL confidence judge — MAPS §2.2 second-order network (Pasquali & Cleeremans 2010).

``SecondOrderNetwork`` (comparator + 2-unit wager head) is the metacognitive judge.

**Fix (a) — the judge acts (2026-07, run/marl-20seeds-1M).** This network is now attached
INSIDE the acting ``R_Actor`` (see :mod:`maps.domains.marl.policy`): it reads the actor's own
representation and its wager BCE (vs ``advantage > 0``) is added to the actor loss, so a single
backward co-trains the acting network (Blindsight/AGL pattern). The paper's separate
``R_Actor_Meta`` / ``R_Critic_Meta`` / ``RNNLayer_Meta`` ghost networks — which never co-trained
the actor (M-C2/M-C3) — have been REMOVED. See ``docs/reviews/fix-a-meta-cotraining-marl.md``.

References
----------
Vargas et al. (2025), MAPS, TMLR submission §5, Table 7.
Pasquali, Timmermans & Cleeremans (2010). Know thyself: metacognitive networks. Cognition.
"""

from __future__ import annotations

import torch
from torch import nn


class SecondOrderNetwork(nn.Module):
    """Comparator + wager head (hidden-sized). Verbatim r_actor_critic_meta.py:17-42.

    ``forward(comparison_matrix, prev_comparison, cascade_rate) -> (wager_logits, comparison_out)``.
    ``wager`` is 2 raw logits (fed to BCE-with-logits by the trainer). ``comparison_out`` threads
    the graded cascade (McClelland 1989) across ``cascade_iterations_2`` calls.
    """

    def __init__(self, num_linear_units: int) -> None:
        super().__init__()
        self.comparison_layer = nn.Linear(num_linear_units, num_linear_units)
        self.wager = nn.Linear(num_linear_units, 2)
        self.dropout = nn.Dropout(p=0.1)
        nn.init.uniform_(self.comparison_layer.weight, -1.0, 1.0)
        nn.init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(self, comparison_matrix, prev_comparison, cascade_rate):
        comparison_out = self.dropout(
            torch.nn.functional.relu(self.comparison_layer(comparison_matrix))
        )
        if prev_comparison is not None:
            comparison_out = cascade_rate * comparison_out + (1 - cascade_rate) * prev_comparison
        return self.wager(comparison_out), comparison_out
