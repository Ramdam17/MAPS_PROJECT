"""Recurrent layers for MARL — paper Fig.4 GRU + cascade (E.7 scaffold).

Two classes :
- :class:`RNNLayer` — baseline GRU + LayerNorm (used by ``MAPPOActor`` /
  ``MAPPOCritic``). Ports ``onpolicy/algorithms/utils/rnn.py``.
- :class:`RNNLayerMeta` — GRU + LayerNorm + cascade update on activations
  (paper eq.6). Used by ``MAPSActor`` / ``MAPSCritic``. Student's
  ``rnn_meta.py`` had an extra ``wager=True`` branch (dead code ; see
  E.4 audit), dropped in port.

Empty module : E.8 scope.
"""

from __future__ import annotations

import torch.nn as nn

__all__ = ["RNNLayer", "RNNLayerMeta"]


class RNNLayer(nn.Module):
    """Baseline GRU + LayerNorm — to be implemented in E.8."""

    def __init__(self, input_dim: int, output_dim: int, recurrent_n: int, use_orthogonal: bool):
        super().__init__()
        raise NotImplementedError("E.8 will implement this.")

    def forward(self, x, hxs, masks):  # pragma: no cover — stub
        raise NotImplementedError("E.8 will implement this.")


class RNNLayerMeta(nn.Module):
    """GRU + LayerNorm + cascade update on activations (paper eq.6).

    To be implemented in E.8. The internal ``wager`` Linear + sigmoid from the
    student's ``rnn_meta.py`` (unreachable ``wager=True`` branch) is NOT ported
    (E.4 decision).
    """

    def __init__(self, input_dim: int, output_dim: int, recurrent_n: int, use_orthogonal: bool):
        super().__init__()
        raise NotImplementedError("E.8 will implement this.")

    def forward(self, x, hxs, masks, prev_cascade, cascade_rate):  # pragma: no cover — stub
        raise NotImplementedError("E.8 will implement this.")
