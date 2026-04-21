"""E.7 scaffold verification : configs load, MarlSetting builds, stubs raise."""

from __future__ import annotations

import pytest

from maps.experiments.marl import MarlSetting
from maps.utils import load_config


def test_marl_training_config_loads_with_paper_T12_defaults():
    cfg = load_config("training/marl")
    # Paper T.12 knobs present + aligned.
    assert cfg.model.hidden_size == 100
    assert cfg.optimizer.name == "Adam"
    assert cfg.optimizer.actor_lr == 7e-5
    assert cfg.optimizer.critic_lr == 7e-5
    assert cfg.optimizer.weight_decay == 1e-5
    assert cfg.ppo.clip_param == 0.2
    assert cfg.ppo.ppo_epoch == 15
    assert cfg.ppo.entropy_coef == 0.01
    assert cfg.training.num_env_steps == 15_000_000
    # Paper eq.13-14 MAPS wager knobs.
    assert cfg.maps.ema_alpha == 0.45
    assert cfg.maps.wager_condition == "r_t_gt_ema"
    # Default = setting 1 (baseline).
    assert cfg.meta is False
    assert cfg.cascade_iterations1 == 1
    assert cfg.cascade_iterations2 == 1


def test_marl_factorial_has_6_settings():
    factorial = load_config("experiments/factorial_marl")
    assert len(factorial.settings) == 6
    ids = [s.id for s in factorial.settings]
    assert ids == [
        "baseline",
        "cascade_1st_no_meta",
        "meta_no_cascade",
        "maps",
        "meta_cascade_2nd",
        "meta_cascade_both",
    ]


def test_marl_setting_dataclass_from_dict():
    factorial = load_config("experiments/factorial_marl")
    maps_setting = next(s for s in factorial.settings if s.id == "maps")
    setting = MarlSetting.from_dict(maps_setting)
    assert setting.id == "maps"
    assert setting.meta is True
    assert setting.cascade_iterations1 == 50
    assert setting.cascade_iterations2 == 1
    assert setting.has_cascade_1st is True
    assert setting.has_cascade_2nd is False


def test_marl_factorial_baseline_has_no_meta_no_cascade():
    factorial = load_config("experiments/factorial_marl")
    baseline = next(s for s in factorial.settings if s.id == "baseline")
    setting = MarlSetting.from_dict(baseline)
    assert setting.meta is False
    assert setting.has_cascade_1st is False
    assert setting.has_cascade_2nd is False


def test_marl_factorial_meta_cascade_both_has_both_cascades():
    factorial = load_config("experiments/factorial_marl")
    both = next(s for s in factorial.settings if s.id == "meta_cascade_both")
    setting = MarlSetting.from_dict(both)
    assert setting.meta is True
    assert setting.has_cascade_1st is True
    assert setting.has_cascade_2nd is True


@pytest.mark.parametrize(
    "substrate_id,expected_num_agents",
    [
        ("commons_harvest_closed", 6),
        ("commons_harvest_partnership", 4),
        ("chemistry", 8),
        ("territory_inside_out", 5),
    ],
)
def test_marl_env_configs_have_paper_num_agents(substrate_id, expected_num_agents):
    """Paper §A.4 num_agents : 6/4/8/5 respectively."""
    cfg = load_config(f"env/marl/{substrate_id}")
    assert cfg.num_agents == expected_num_agents
    assert len(cfg.roles) == expected_num_agents
    assert cfg.obs_shape_agent == [11, 11, 3]


def test_marl_factorial_all_3_seeds():
    factorial = load_config("experiments/factorial_marl")
    assert factorial.n_seeds == 3
    assert len(factorial.seeds) == 3
    assert len(factorial.substrates) == 4


# E.9+ stubs : ensure remaining stubs raise NotImplementedError.
# (E.8 implemented networks — their shape/forward tests live in test_networks.py.)


def test_trainer_stub_raises():
    from maps.experiments.marl.trainer import MAPPOTrainer

    with pytest.raises(NotImplementedError):
        MAPPOTrainer(cfg=None, policy=None, device=None)


def test_runner_stub_raises():
    from maps.experiments.marl.runner import MeltingpotRunner

    with pytest.raises(NotImplementedError):
        MeltingpotRunner(config=None)
