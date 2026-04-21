"""MARL trainer — MAPPO + MAPS wager integration (E.7 scaffold).

Ports student ``r_mappo/r_mappo.py``. Standard PPO (clip + value + entropy)
plus a metacognitive loss (``binary_cross_entropy_with_logits`` on ``values_meta``
vs ``wager_objective`` when ``setting.meta=True``).

Paper §2.2 eq.5 + §2.2 eq.13-14 + standard PPO (Yu et al. 2022).

E.9 scope — not implemented here.
"""

from __future__ import annotations

__all__ = ["MAPPOTrainer"]


class MAPPOTrainer:
    """PPO trainer with optional MAPS wager loss. To be implemented in E.9."""

    def __init__(self, cfg, policy, device):
        raise NotImplementedError("E.9 will implement this.")

    def train(self, buffer, wager_objective=None, update_actor: bool = True, meta: bool = False):
        raise NotImplementedError("E.9 will implement this.")

    def prep_training(self):
        raise NotImplementedError("E.9 will implement this.")

    def prep_rollout(self):
        raise NotImplementedError("E.9 will implement this.")
