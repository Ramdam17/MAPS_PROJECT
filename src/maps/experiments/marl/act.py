"""Action head — Discrete only (E.5 scope lock).

MeltingPot substrates expose discrete action spaces per agent (see
``MeltingPotEnv._convert_spaces_tuple_to_dict`` → `spec_to_space` → `Discrete`).

The student ``act.py`` supports Discrete / Box / MultiBinary / MultiDiscrete /
Mixed. We port only the Discrete path ; others are omitted per E.5.

Ports :
- ``distributions.FixedCategorical`` → :class:`FixedCategorical`
- ``distributions.Categorical`` → :class:`Categorical`
- ``act.ACTLayer`` Discrete branch → :class:`ACTLayer`
"""

from __future__ import annotations

import torch
import torch.nn as nn

from maps.experiments.marl.util import init

__all__ = ["FixedCategorical", "Categorical", "ACTLayer"]


class FixedCategorical(torch.distributions.Categorical):
    """Categorical distribution with student's shape conventions.

    - ``sample()`` returns actions with an extra trailing dim (L16).
    - ``log_probs()`` sums log-probs across the last dim (L18-25).
    - ``mode()`` returns argmax of probs with trailing dim (L27-28).
    """

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    """Linear → categorical logits, used as the discrete action head."""

    def __init__(self, num_inputs: int, num_outputs: int, use_orthogonal: bool = True, gain: float = 0.01):
        super().__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][int(use_orthogonal)]

        def init_(m: nn.Module) -> nn.Module:
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x: torch.Tensor, available_actions: torch.Tensor | None = None) -> FixedCategorical:
        logits = self.linear(x)
        if available_actions is not None:
            logits[available_actions == 0] = -1e10
        return FixedCategorical(logits=logits)


class ACTLayer(nn.Module):
    """Discrete action head — student's ``ACTLayer`` with only the Discrete branch.

    The MeltingPot substrates we target (commons_harvest_*, chemistry,
    territory_inside_out) all expose Discrete action spaces, so this is the
    only path we need. Other action space types (Box, MultiBinary,
    MultiDiscrete, Mixed) are OMITTED per E.5 scope lock.
    """

    def __init__(self, action_space, inputs_dim: int, use_orthogonal: bool = True, gain: float = 0.01):
        super().__init__()
        if action_space.__class__.__name__ != "Discrete":
            raise NotImplementedError(
                f"Port supports only Discrete action space (E.5 scope). Got {action_space.__class__.__name__}."
            )
        self.action_type = "Discrete"
        self.action_out = Categorical(inputs_dim, action_space.n, use_orthogonal, gain)

    def forward(
        self,
        x: torch.Tensor,
        available_actions: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(actions, action_log_probs)``."""
        action_logits = self.action_out(x, available_actions)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        # Student uses ``log_prob(actions.squeeze(-1))`` here (not ``log_probs``),
        # see r_actor_critic.py:120 path. Preserve that :
        action_log_probs = action_logits.log_prob(actions.squeeze(-1))
        return actions, action_log_probs

    def evaluate_actions(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        available_actions: torch.Tensor | None = None,
        active_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(action_log_probs, dist_entropy)``."""
        action_logits = self.action_out(x, available_actions)
        action_log_probs = action_logits.log_probs(action)
        if active_masks is not None:
            dist_entropy = (action_logits.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()
        return action_log_probs, dist_entropy
