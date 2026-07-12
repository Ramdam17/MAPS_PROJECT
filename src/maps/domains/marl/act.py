"""MARL action layer (Discrete) — MAPS §5 (MeltingPot MAPPO).

Ported from ``onpolicy/algorithms/utils/{distributions,act}.py``, reduced to the
**Discrete** action path (MeltingPot substrates are Discrete). Other action types
(Box/MultiDiscrete/mixed) are out of scope for this domain.

Faithful detail: ``forward`` uses ``dist.log_prob(actions.squeeze(-1))`` (shape
``(B,)``) while ``evaluate_actions`` uses ``dist.log_probs(action)`` (shape
``(B,1)``) — reproduced as in the source (an intentional asymmetry).

References
----------
Vargas et al. (2025), MAPS, TMLR submission §5, Table 7.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from maps.domains.marl.util import init


class FixedCategorical(torch.distributions.Categorical):
    """Categorical with MAPPO-shaped sample/log_probs/mode. Verbatim distributions.py:14-28."""

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    """Linear → FixedCategorical. Verbatim distributions.py:55-69."""

    def __init__(
        self, num_inputs: int, num_outputs: int, use_orthogonal: bool = True, gain: float = 0.01
    ) -> None:
        super().__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x: Tensor, available_actions: Tensor | None = None) -> FixedCategorical:
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class ACTLayer(nn.Module):
    """Discrete action head. Port of ``ACTLayer`` (act.py) — Discrete path only."""

    def __init__(self, action_space, inputs_dim: int, use_orthogonal: bool, gain: float) -> None:
        super().__init__()
        if action_space.__class__.__name__ != "Discrete":
            raise NotImplementedError(
                f"MARL domain supports Discrete action spaces only; got {action_space.__class__.__name__}"
            )
        self.action_out = Categorical(inputs_dim, action_space.n, use_orthogonal, gain)

    def forward(
        self, x: Tensor, available_actions: Tensor | None = None, deterministic: bool = False
    ):
        action_logits = self.action_out(x, available_actions)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        # Source uses raw .log_prob(squeeze) here (shape (B,)) — faithful.
        action_log_probs = action_logits.log_prob(actions.squeeze(-1))
        return actions, action_log_probs

    def get_probs(self, x: Tensor, available_actions: Tensor | None = None) -> Tensor:
        return self.action_out(x, available_actions).probs

    def evaluate_actions(
        self,
        x: Tensor,
        action: Tensor,
        available_actions: Tensor | None = None,
        active_masks: Tensor | None = None,
    ):
        action_logits = self.action_out(x, available_actions)
        action_log_probs = action_logits.log_probs(action)  # (B,1), source .log_probs
        if active_masks is not None:
            dist_entropy = (
                action_logits.entropy() * active_masks.squeeze(-1)
            ).sum() / active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()
        return action_log_probs, dist_entropy
