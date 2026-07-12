"""MARL recurrent layer — MAPS §5 (MeltingPot MAPPO).

Ported verbatim from ``external/paper_reference/marl_tmlr/onpolicy/algorithms/utils/rnn.py``
(``RNNLayer``): a GRU + LayerNorm handling both rollout (1-step) and minibatch
(chunked) modes with mask-based sequence resets.

The cascade blend (``output_cascade1 = rate*x + (1-rate)*prev_h1``) is returned as a
**separate** output; the layer itself returns ``norm(x)`` (raw GRU output normalised),
NOT the cascade — only the meta path consumes ``output_cascade1`` (M-H3, faithful).

References
----------
Vargas et al. (2025), MAPS, TMLR submission §5, Table 7.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RNNLayer(nn.Module):
    """Verbatim ``RNNLayer`` (rnn.py:6-90)."""

    def __init__(
        self, inputs_dim: int, outputs_dim: int, recurrent_n: int, use_orthogonal: bool
    ) -> None:
        super().__init__()
        self._recurrent_N = recurrent_n
        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=recurrent_n)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                (nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_)(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(
        self, x: Tensor, hxs: Tensor, masks: Tensor, prev_h1: Tensor | None, cascade_rate1: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x.size(0) == hxs.size(0):
            # Rollout (1-step).
            x, hxs = self.rnn(
                x.unsqueeze(0),
                (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1))
                .transpose(0, 1)
                .contiguous(),
            )
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # Minibatch (chunked): x is (episode_len*batch_num, -1).
            batch_num = hxs.size(0)
            episode_len = int(x.size(0) / batch_num)
            x = x.view(episode_len, batch_num, x.size(1))
            masks = masks.view(episode_len, batch_num)
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()
            has_zeros = [0, *has_zeros, episode_len]
            hxs = hxs.transpose(0, 1)
            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx, end_idx = has_zeros[i], has_zeros[i + 1]
                temp = (
                    hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)
                ).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)
            x = torch.cat(outputs, dim=0)
            x = x.reshape(episode_len * batch_num, -1)
            hxs = hxs.transpose(0, 1)

        output_cascade1 = x
        if prev_h1 is not None:
            output_cascade1 = cascade_rate1 * x + (1 - cascade_rate1) * prev_h1
        x = self.norm(x)
        return x, hxs, output_cascade1
