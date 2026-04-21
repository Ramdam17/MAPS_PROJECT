"""Value normalization (port of student ``onpolicy/utils/valuenorm.py``).

Implements the running-mean/variance normalizer used to stabilize MAPPO's
value loss — tracks debiased EMA of returns and normalizes/denormalizes them
transparently. Enabled via ``cfg.ppo.use_valuenorm = True`` (port default).

Reference
---------
Yu et al. 2022 "The Surprising Effectiveness of PPO in Cooperative Multi-Agent
Games" — PPO + value normalization stabilizer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["ValueNorm"]


class ValueNorm(nn.Module):
    """Normalize a vector of observations across the first ``norm_axes`` dims.

    Verbatim port of student L8-78. All parameters are NN parameters with
    ``requires_grad=False`` (serialized cleanly via state_dict).
    """

    def __init__(
        self,
        input_shape: int | tuple[int, ...],
        norm_axes: int = 1,
        beta: float = 0.99999,
        per_element_update: bool = False,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update

        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self) -> tuple[torch.Tensor, torch.Tensor]:
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
            batch_size = int(np.prod(input_vector.size()[: self.norm_axes]))
            weight = self.beta**batch_size
        else:
            weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector) -> torch.Tensor:
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(self.running_mean.device)

        mean, var = self.running_mean_var()
        return (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]

    def denormalize(self, input_vector):
        """Returns numpy for API compatibility with student (L76)."""
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(self.running_mean.device)

        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        return out.detach().cpu().numpy()
