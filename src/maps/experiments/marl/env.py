"""MeltingPot env wrapper (E.7 scaffold).

Ports student ``onpolicy/envs/meltingpot/MeltingPot_Env.py``. Provides :

- :class:`MeltingPotEnv` : wraps a dmlab2d substrate as a rllib MultiAgentEnv
  with per-agent dict observations + rewards.
- :class:`DownSamplingSubstrateWrapper` : reduces 88×88 RGB to 11×11 via
  ``cv2.resize(..., INTER_AREA)``.
- :func:`env_creator` : factory combining substrate + downsample + MultiAgentEnv.

All imports gated on ``dmlab2d`` / ``meltingpot`` availability — these live
in the MARL-only .venv-marl (Python 3.11), see ``docs/install_marl_drac.md``.

E.10 scope — not implemented here.
"""

from __future__ import annotations

__all__ = ["MeltingPotEnv", "DownSamplingSubstrateWrapper", "env_creator"]


class MeltingPotEnv:
    """To be implemented in E.10. Will import ``dmlab2d`` + ``meltingpot``."""

    def __init__(self, env, max_cycles: int = 400):
        raise NotImplementedError("E.10 will implement this.")


class DownSamplingSubstrateWrapper:
    """To be implemented in E.10."""

    def __init__(self, substrate, scaled: int = 8):
        raise NotImplementedError("E.10 will implement this.")


def env_creator(env_config):  # pragma: no cover — stub
    """To be implemented in E.10."""
    raise NotImplementedError("E.10 will implement this.")
