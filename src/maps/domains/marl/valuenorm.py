"""MARL value normalisation — MAPPO running value normaliser.

Ported verbatim from ``external/paper_reference/marl_tmlr/onpolicy/utils/valuenorm.py``.
A debiased exponential-moving-average normaliser over the value targets, used by
the critic loss and the GAE ``compute_returns`` (which calls :meth:`denormalize`).

References
----------
Yu et al. (2022). The surprising effectiveness of PPO in cooperative MARL (MAPPO).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ValueNorm(nn.Module):
    """Normalise a vector across the first ``norm_axes`` dimensions (debiased EMA)."""

    def __init__(
        self,
        input_shape: int,
        norm_axes: int = 1,
        beta: float = 0.99999,
        per_element_update: bool = False,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update

        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector) -> None:
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(self.running_mean.device)

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector**2).mean(dim=tuple(range(self.norm_axes)))

        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[: self.norm_axes])
            weight = self.beta**batch_size
        else:
            weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(self.running_mean.device)
        mean, var = self.running_mean_var()
        return (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[
            (None,) * self.norm_axes
        ]

    def denormalize(self, input_vector):
        """Transform normalised data back to the original distribution (returns numpy)."""
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(self.running_mean.device)
        mean, var = self.running_mean_var()
        out = (
            input_vector * torch.sqrt(var)[(None,) * self.norm_axes]
            + mean[(None,) * self.norm_axes]
        )
        return out.cpu().numpy()
