"""Unit tests for maps.utils.config — YAML loading, composition, overrides."""

from __future__ import annotations

import pytest

from maps.utils.config import load_config


def test_load_maps_has_locked_constants():
    cfg = load_config("maps")
    assert cfg.cascade.alpha == 0.02
    assert cfg.cascade.n_iterations == 50
    assert cfg.first_order.hidden_dim == 40
    assert cfg.second_order.input_dim == 100
    assert cfg.seed == 42


def test_composition_inherits_and_overrides():
    """training/blindsight.yaml composes maps.yaml and overrides first_order.input_dim."""
    cfg = load_config("training/blindsight")
    # Inherited verbatim:
    assert cfg.cascade.alpha == 0.02
    assert cfg.second_order.input_dim == 100
    # Overridden by blindsight.yaml:
    assert cfg.first_order.input_dim == 100
    assert cfg.first_order.hidden_dim == 100
    # Domain-specific:
    assert cfg.optimizer.name == "ADAMAX"
    assert cfg.scheduler.step_size == 25


def test_agl_composition():
    cfg = load_config("training/agl")
    assert cfg.cascade.alpha == 0.02  # inherited
    assert cfg.first_order.input_dim == 48  # AGL
    assert cfg.bits_per_letter == 6
    assert cfg.optimizer.lr_first_order == 0.4


def test_cli_overrides_apply_last():
    cfg = load_config(
        "training/blindsight",
        overrides=["train.n_epochs=10", "cascade.alpha=0.05"],
    )
    assert cfg.train.n_epochs == 10
    assert cfg.cascade.alpha == 0.05


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")


def test_experiments_factorial_structure():
    cfg = load_config("experiments/factorial_2x2")
    assert len(cfg.settings) == 4
    ids = {s.id for s in cfg.settings}
    assert ids == {"neither", "cascade_only", "second_order_only", "both"}
    assert cfg.n_seeds == 5
    assert list(cfg.seeds) == [42, 43, 44, 45, 46]
