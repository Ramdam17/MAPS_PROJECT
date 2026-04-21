"""Student ``ACTLayer`` — Discrete branch only (student L16-77 + L120 forward).

Other action-space branches (Box / MultiBinary / MultiDiscrete / Mixed) are
not needed for MeltingPot parity and are stripped.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .distributions import Categorical


class ACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        super().__init__()
        self.action_type = action_space.__class__.__name__
        assert self.action_type == "Discrete", (
            f"Only Discrete supported in parity ref (got {self.action_type})"
        )
        self.action_out = Categorical(inputs_dim, action_space.n, use_orthogonal, gain)

    def forward(self, x, available_actions=None, deterministic=False):
        action_logits = self.action_out(x, available_actions)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        action_log_probs = action_logits.log_prob(actions.squeeze(-1))
        return actions, action_log_probs
