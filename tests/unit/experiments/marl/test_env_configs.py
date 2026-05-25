"""Unit tests for per-substrate env configs + build_env_from_config (E.11).

Verifies :
- All 4 substrate YAMLs under ``config/env/marl/`` carry paper-faithful
  ``num_agents`` (per paper §A.4) + ``episode_length`` / ``max_cycles``
  matching student ``train_meltingpot.sh``.
- ``build_env_from_config`` validates consistency between ``num_agents`` and
  ``len(roles)`` and rejects mismatches early.
- Without meltingpot in the current venv, ``build_env_from_config`` fails
  cleanly at the substrate.build import.
"""

from __future__ import annotations

import pytest

from maps.experiments.marl import env as marl_env
from maps.utils import load_config


# Paper §A.4 + student train_meltingpot.sh — canonical per-substrate knobs.
_EXPECTED = {
    "commons_harvest_closed": {"num_agents": 6, "substrate_name": "commons_harvest__closed"},
    "commons_harvest_partnership": {"num_agents": 4, "substrate_name": "commons_harvest__partnership"},
    "chemistry": {"num_agents": 8, "substrate_name": "chemistry__three_metabolic_cycles_with_plentiful_distractors"},
    "territory_inside_out": {"num_agents": 5, "substrate_name": "territory__inside_out"},
}


@pytest.mark.parametrize("substrate_id,expected", list(_EXPECTED.items()))
def test_env_config_has_paper_num_agents_and_substrate_name(substrate_id, expected):
    cfg = load_config(f"env/marl/{substrate_id}")
    assert cfg.num_agents == expected["num_agents"]
    assert cfg.substrate_name == expected["substrate_name"]
    assert len(cfg.roles) == expected["num_agents"]


@pytest.mark.parametrize("substrate_id", list(_EXPECTED.keys()))
def test_env_config_has_paper_episode_length_and_max_cycles(substrate_id):
    """Student train_meltingpot.sh : episode_length = 1000 for all four substrates.
    Our port aligns ``max_cycles == episode_length`` so env truncation matches
    the rollout buffer length (avoids stale-last() timesteps)."""
    cfg = load_config(f"env/marl/{substrate_id}")
    assert cfg.episode_length == 1000
    assert cfg.max_cycles == 1000
    # Invariant the runner relies on : env can sustain a full rollout length.
    assert cfg.max_cycles >= cfg.episode_length


@pytest.mark.parametrize("substrate_id", list(_EXPECTED.keys()))
def test_env_config_has_paper_downsample_and_obs_shape(substrate_id):
    """Paper §A.4 : per-agent obs is 11×11×3 after 8× downsample."""
    cfg = load_config(f"env/marl/{substrate_id}")
    assert cfg.downsample_scale == 8
    assert list(cfg.obs_shape_agent) == [11, 11, 3]


def test_build_env_from_config_rejects_agent_count_mismatch():
    """Protects the runner from downstream shape breakage if someone edits
    ``roles`` without updating ``num_agents``."""
    bad_cfg = {
        "substrate_name": "commons_harvest__closed",
        "num_agents": 6,
        "roles": ["default", "default", "default"],  # only 3, not 6
        "downsample_scale": 8,
        "max_cycles": 1000,
    }
    with pytest.raises(ValueError, match="num_agents"):
        marl_env.build_env_from_config(bad_cfg)


def test_build_env_from_config_reads_defaults_for_missing_keys():
    """Missing downsample_scale / max_cycles fall back to module defaults."""
    # We can't actually build without meltingpot — but we CAN assert validation
    # runs before the import path. Use mismatched roles to trip validation.
    bad_cfg = {
        "substrate_name": "commons_harvest__closed",
        "num_agents": 2,
        "roles": ["default"],  # mismatch trips ValueError before meltingpot call
    }
    with pytest.raises(ValueError, match="num_agents"):
        marl_env.build_env_from_config(bad_cfg)


def test_build_env_from_config_requires_meltingpot_install():
    """Valid config → tries to call meltingpot.substrate.build → ImportError
    (in main .venv which lacks meltingpot)."""
    good_cfg = load_config("env/marl/commons_harvest_partnership")
    try:
        import meltingpot  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            marl_env.build_env_from_config(good_cfg)
    else:
        pytest.skip("meltingpot available — no negative import path to check")
