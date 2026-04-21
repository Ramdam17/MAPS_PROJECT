"""MARL policy networks — paper Fig.4 actor + critic + MAPS additions.

Five classes (E.8 scope — empty stubs here) :

- :class:`MAPPOActor` — baseline actor (CNN → RNN → ACTLayer).
- :class:`MAPPOCritic` — baseline critic (CNN → RNN → v_out).
- :class:`MAPSActor` — MAPS actor (same as MAPPOActor + MarlSecondOrderNetwork
  producing wager from pre-RNN vs post-RNN feature comparator).
- :class:`MAPSCritic` — MAPS critic (no 2nd-order — paper Fig.4 only wagers
  on actor side ; a separate ``MAPSCritic`` exists in student code but without
  a SecondOrderNetwork, just using ``RNNLayerMeta``).
- :class:`MarlSecondOrderNetwork` — comparator (``Linear + ReLU + Dropout``) →
  cascade → wager (``Linear(H, 2)`` raw logits). Paper §2.2 eq.1-3 + MARL's
  own comparison_layer (paper-silent but student-present, see E.4 audit).
- :class:`MAPPOPolicy` — wrapper managing 4 networks + their optimizers ;
  dispatches to actor / actor_meta per setting.

References
----------
- Paper §2.2 + Fig.4
- Student ``r_mappo/algorithm/r_actor_critic.py`` + ``r_actor_critic_meta.py``.
- Architecture summary : ``docs/reviews/marl-maps-additions.md`` §(i).
"""

from __future__ import annotations

import torch.nn as nn

__all__ = [
    "MAPPOActor",
    "MAPPOCritic",
    "MAPSActor",
    "MAPSCritic",
    "MarlSecondOrderNetwork",
    "MAPPOPolicy",
]


class MarlSecondOrderNetwork(nn.Module):
    """Paper §2.2 eq.1-3 + MARL's ``comparison_layer`` (E.4 audit).

    Flow : ``comparator (pre-RNN − post-RNN features) → Dropout(ReLU(Linear(C))) →
    cascade (eq.6) → Linear(H, 2) raw logits``. Downstream loss uses
    ``binary_cross_entropy_with_logits`` (eq.5).

    To be implemented in E.8.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        raise NotImplementedError("E.8 will implement this.")

    def forward(self, comparator, prev_comparator, cascade_rate):  # pragma: no cover
        raise NotImplementedError("E.8 will implement this.")


class MAPPOActor(nn.Module):
    """Baseline actor (setting meta=False). To be implemented in E.8."""

    def __init__(self, cfg, obs_space, action_space, device):
        super().__init__()
        raise NotImplementedError("E.8 will implement this.")


class MAPPOCritic(nn.Module):
    """Baseline critic (setting meta=False). To be implemented in E.8."""

    def __init__(self, cfg, cent_obs_space, device):
        super().__init__()
        raise NotImplementedError("E.8 will implement this.")


class MAPSActor(nn.Module):
    """MAPS actor (setting meta=True). To be implemented in E.8.

    Includes ``MarlSecondOrderNetwork`` on the wager path + ``layer_input``
    pre-RNN projection (student-present, paper-silent — E.4 decision to keep).
    """

    def __init__(self, cfg, obs_space, action_space, device):
        super().__init__()
        raise NotImplementedError("E.8 will implement this.")


class MAPSCritic(nn.Module):
    """MAPS critic (setting meta=True). To be implemented in E.8.

    No 2nd-order on critic side (paper Fig.4). Uses ``RNNLayerMeta`` but
    without a wager head.
    """

    def __init__(self, cfg, cent_obs_space, device):
        super().__init__()
        raise NotImplementedError("E.8 will implement this.")


class MAPPOPolicy:
    """Wraps 4 networks (actor, critic, actor_meta, critic_meta) + optimizers.

    Dispatches to the right branch per setting. Optimizer choice : Adam (paper
    T.12 default) + RangerVA (optional, from torch-optimizer). Other variants
    (AMS, AdamW, SWATS, SGD, RangerQH, etc.) explicitly OMITTED per E.5 scope lock.

    To be implemented in E.8.
    """

    def __init__(self, cfg, obs_space, cent_obs_space, action_space, device):
        raise NotImplementedError("E.8 will implement this.")
