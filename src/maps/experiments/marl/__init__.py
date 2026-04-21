"""MARL (Multi-Agent Reinforcement Learning) experiment package.

Ports student ``MARL/MAPPO-ATTENTION/`` → ``external/paper_reference/marl_tmlr/``
into a paper-faithful MAPPO + MAPS implementation running on MeltingPot 2.x
substrates.

See :
- ``docs/plans/plan-20260421-phase-e-marl.md`` for the 18-sub-phase plan.
- ``docs/reviews/marl-architecture.md`` (E.2 audit).
- ``docs/reviews/marl-env.md`` (E.3 audit).
- ``docs/reviews/marl-maps-additions.md`` (E.4 audit).
- ``docs/reviews/marl-scope-decisions.md`` (E.5 scope lock).
- ``docs/install_marl_drac.md`` (E.6 install recipe).

Package layout
--------------
- ``setting.py`` : :class:`MarlSetting` dataclass (6 factorial cells).
- ``encoder.py`` : ConvEncoder (paper Fig.4). **E.8 scope.**
- ``rnn.py`` : RNNLayer + RNNLayerMeta. **E.8 scope.**
- ``policy.py`` : MAPPOActor + MAPPOCritic + MAPSActor + MAPSCritic +
  MarlSecondOrderNetwork + MAPPOPolicy wrapper. **E.8 scope.**
- ``trainer.py`` : MAPPOTrainer (PPO clip + value + entropy + MAPS wager BCE).
  **E.9 scope.**
- ``env.py`` : MeltingPotEnv wrapper. **E.10 scope.**
- ``runner.py`` : MeltingpotRunner (rollout + EMA wager + train loop).
  **E.9 scope.**
- ``data.py`` : rollout buffer dataclasses. **E.9 scope.**

Runtime environment
-------------------
MARL runs from the **dedicated ``.venv-marl`` (Python 3.11.4)** — the main
``.venv`` (Python 3.12) cannot install ``dmlab2d`` / ``meltingpot`` due to
missing cp312 wheels. See ``docs/install_marl_drac.md``.

Exports
-------
- :class:`MarlSetting` (E.7)
- :class:`MAPPOActor`, :class:`MAPPOCritic`, :class:`MAPSActor`,
  :class:`MAPSCritic`, :class:`MarlSecondOrderNetwork`, :class:`MAPPOPolicy` (E.8)
- :class:`MAPPOTrainer`, :class:`TrainInfo`, :class:`ValueNorm` (E.9a)
- :class:`RolloutBuffer`, :class:`MeltingpotRunner`, :class:`RunnerConfig`,
  :func:`compute_wager_objective` (E.9b)

Env wrappers raise ``NotImplementedError`` until E.10.
"""

from __future__ import annotations

from maps.experiments.marl.data import RolloutBuffer
from maps.experiments.marl.policy import (
    MAPPOActor,
    MAPPOCritic,
    MAPPOPolicy,
    MAPSActor,
    MAPSCritic,
    MarlSecondOrderNetwork,
)
from maps.experiments.marl.runner import (
    MeltingpotRunner,
    RunnerConfig,
    compute_wager_objective,
)
from maps.experiments.marl.setting import MarlSetting
from maps.experiments.marl.trainer import MAPPOTrainer, TrainInfo
from maps.experiments.marl.valuenorm import ValueNorm

__all__ = [
    "MAPPOActor",
    "MAPPOCritic",
    "MAPPOPolicy",
    "MAPPOTrainer",
    "MAPSActor",
    "MAPSCritic",
    "MarlSecondOrderNetwork",
    "MarlSetting",
    "MeltingpotRunner",
    "RolloutBuffer",
    "RunnerConfig",
    "TrainInfo",
    "ValueNorm",
    "compute_wager_objective",
]
