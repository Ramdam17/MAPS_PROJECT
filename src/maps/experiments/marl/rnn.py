"""GRU + cascade layers for MARL.

Ports :
- :class:`RNNLayer` — student ``utils/rnn.py:RNNLayer`` L7-90 (GRU + LayerNorm
  + optional cascade update on activations, paper eq.6).
- :class:`RNNLayerMeta` — student ``utils/rnn_meta.py:RNNLayer_Meta`` L7-117
  with the dead ``wager=True`` branch DROPPED (E.4 audit confirmed 0 call
  sites). Only the ``wager=False`` path is retained ; meta-layer has same
  shape as baseline RNNLayer plus a cascade update.

Both classes handle the student's **two call shapes** :
1. Rollout mode : ``x.size(0) == hxs.size(0)`` → single-step forward.
2. Minibatch mode : batch contains multiple episode steps, mask transitions
   split the pass into contiguous sub-batches.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RNNLayer", "RNNLayerMeta"]


def _init_gru(rnn: nn.GRU, use_orthogonal: bool) -> None:
    """Student's GRU init L17-24. Constant-zero biases, orthogonal/xavier weights."""
    for name, param in rnn.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            if use_orthogonal:
                nn.init.orthogonal_(param)
            else:
                nn.init.xavier_uniform_(param)


def _rnn_forward(
    rnn: nn.GRU,
    recurrent_n: int,
    x: torch.Tensor,
    hxs: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared GRU forward — handles rollout (1-step) and minibatch (N-step) modes.

    Ported verbatim from student ``rnn.py:28-83``. The logic is load-bearing for
    student-parity ; kept as close to source as possible.
    """
    if x.size(0) == hxs.size(0):
        # Rollout mode (single step). ``x`` is (batch, input_size), hxs is (batch, N, H).
        x, hxs = rnn(
            x.unsqueeze(0),
            (hxs * masks.repeat(1, recurrent_n).unsqueeze(-1)).transpose(0, 1).contiguous(),
        )
        x = x.squeeze(0)
        hxs = hxs.transpose(0, 1)
    else:
        # Minibatch mode : ``x`` is (episode_len * batch_num, input_size).
        batch_num = hxs.size(0)
        episode_len = int(x.size(0) / batch_num)
        x = x.view(episode_len, batch_num, x.size(1))
        masks = masks.view(episode_len, batch_num)

        has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())
        if has_zeros.dim() == 0:
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()
        has_zeros = [0] + has_zeros + [episode_len]

        hxs = hxs.transpose(0, 1)
        outputs = []
        for i in range(len(has_zeros) - 1):
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]
            temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(recurrent_n, 1, 1)).contiguous()
            rnn_scores, hxs = rnn(x[start_idx:end_idx], temp)
            outputs.append(rnn_scores)
        x = torch.cat(outputs, dim=0)
        x = x.reshape(episode_len * batch_num, -1)
        hxs = hxs.transpose(0, 1)

    return x, hxs


class RNNLayer(nn.Module):
    """Baseline GRU + LayerNorm + cascade update.

    Used by :class:`MAPPOActor` / :class:`MAPPOCritic` (settings meta=False).
    Student source : ``utils/rnn.py:7-90``.

    Note on cascade : the student class also threads a ``cascade_rate1`` update
    — even in the non-meta path. We preserve that for parity (paper eq.6
    applies to any GRU-output if caller requests).
    """

    def __init__(self, inputs_dim: int, outputs_dim: int, recurrent_n: int = 1, use_orthogonal: bool = True):
        super().__init__()
        self._recurrent_n = recurrent_n
        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=recurrent_n)
        _init_gru(self.rnn, use_orthogonal)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(
        self,
        x: torch.Tensor,
        hxs: torch.Tensor,
        masks: torch.Tensor,
        prev_cascade: torch.Tensor | None,
        cascade_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(normed_x, hxs, output_cascade)``."""
        x, hxs = _rnn_forward(self.rnn, self._recurrent_n, x, hxs, masks)

        output_cascade = x
        if prev_cascade is not None:
            output_cascade = cascade_rate * x + (1.0 - cascade_rate) * prev_cascade

        x = self.norm(x)
        return x, hxs, output_cascade


class RNNLayerMeta(nn.Module):
    """GRU + LayerNorm + cascade update (meta variant).

    Used by :class:`MAPSActor` / :class:`MAPSCritic` (settings meta=True).
    Student source : ``utils/rnn_meta.py:7-96`` (only the ``wager=False`` branch).
    The internal ``wager=True`` branch (L97-107 in student) was unreachable
    (grep confirmed 0 call sites in E.4) and is OMITTED per E.5 scope lock.
    """

    def __init__(self, inputs_dim: int, outputs_dim: int, recurrent_n: int = 1, use_orthogonal: bool = True):
        super().__init__()
        self._recurrent_n = recurrent_n
        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=recurrent_n)
        _init_gru(self.rnn, use_orthogonal)
        self.norm = nn.LayerNorm(outputs_dim)
        # Note: student also defined ``self.wager`` + ``self.sigmoid`` inside
        # ``RNNLayer_Meta`` but they're dead code. NOT ported.

    def forward(
        self,
        x: torch.Tensor,
        hxs: torch.Tensor,
        masks: torch.Tensor,
        prev_cascade: torch.Tensor | None,
        cascade_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(normed_x, hxs, output_cascade)``. Identical in shape to :class:`RNNLayer`."""
        x, hxs = _rnn_forward(self.rnn, self._recurrent_n, x, hxs, masks)

        output_cascade = x
        if prev_cascade is not None:
            output_cascade = cascade_rate * x + (1.0 - cascade_rate) * prev_cascade

        x = self.norm(output_cascade)
        return x, hxs, output_cascade
