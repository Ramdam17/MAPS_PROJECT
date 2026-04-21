"""MARL rollout runner (E.7 scaffold).

Ports student ``onpolicy/runner/separated/meltingpot_runner.py`` (the only
paper-faithful runner — shared variant has syntax bug, see E.3 audit).

Key responsibilities :
- Rollout collection : ``n_rollout_threads`` × ``episode_length`` env steps
  per episode, one step at a time per agent.
- EMA wager signal (paper eq.13-14) : ``grad_rewards = α · r_t + (1-α) · EMA_{t-1}``
  with ``α=0.45`` (paper), ``wager_objective = (1, 0) if r_t > EMA_t``.
  **Note :** student uses ``α=0.25`` + ``EMA > 0`` (not paper-faithful, see
  E.3 audit ``D-marl-ema-alpha`` + ``D-marl-wager-condition``). Port aligns
  to paper.
- Advantage computation + PPO update call.
- Per-substrate logging.

E.9 scope — not implemented here.
"""

from __future__ import annotations

__all__ = ["MeltingpotRunner"]


class MeltingpotRunner:
    """MARL rollout + train loop. To be implemented in E.9."""

    def __init__(self, config):
        raise NotImplementedError("E.9 will implement this.")

    def run(self):
        raise NotImplementedError("E.9 will implement this.")
