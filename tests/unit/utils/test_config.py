"""Unit tests for :mod:`maps.utils.config`."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from maps.utils.config import (
    CONFIG_ROOT,
    _resolve_config_path,
    load_config,
    parse_overrides,
)

# ---------------------------------------------------------------------------
# load_config — happy path
# ---------------------------------------------------------------------------


def test_load_maps_yaml_has_canonical_constants() -> None:
    cfg = load_config("maps")
    assert isinstance(cfg, DictConfig)
    assert cfg.cascade.alpha == 0.02
    assert cfg.cascade.n_iterations == 50
    assert cfg.first_order.hidden_dim == 40
    assert cfg.second_order.dropout == 0.5
    assert cfg.losses.cae_lambda == 1.0e-4
    assert cfg.seed == 42
    assert cfg.first_order_loss.kind == "cae"


def test_blindsight_training_composes_from_maps() -> None:
    """Loading domains/blindsight/training inherits maps.yaml constants
    AND applies Blindsight-specific overrides."""
    cfg = load_config("domains/blindsight/training")

    # Inherited from maps.yaml
    assert cfg.cascade.alpha == 0.02
    assert cfg.cascade.n_iterations == 50
    assert cfg.losses.cae_lambda == 1.0e-4

    # Blindsight overrides
    assert cfg.first_order.input_dim == 100  # paper: 100, overrides default 48
    assert cfg.first_order.hidden_dim == 40  # D12.3, RG-002 H5
    assert cfg.second_order.hidden_dim == 100  # D.25 Pasquali restored

    # Blindsight-specific knobs
    assert cfg.optimizer.name == "ADAMAX"
    assert cfg.optimizer.lr_first_order == 0.5
    assert cfg.train.n_epochs == 200


# ---------------------------------------------------------------------------
# load_config — overrides
# ---------------------------------------------------------------------------


def test_overrides_change_values() -> None:
    cfg = load_config("maps", overrides=["cascade.alpha=0.1"])
    assert cfg.cascade.alpha == 0.1


def test_overrides_can_set_deep_paths() -> None:
    cfg = load_config(
        "domains/blindsight/training",
        overrides=["train.n_epochs=42", "first_order.hidden_dim=99"],
    )
    assert cfg.train.n_epochs == 42
    assert cfg.first_order.hidden_dim == 99


def test_overrides_empty_or_none_is_noop() -> None:
    cfg_a = load_config("maps")
    cfg_b = load_config("maps", overrides=[])
    cfg_c = load_config("maps", overrides=None)
    assert cfg_a.cascade.alpha == cfg_b.cascade.alpha == cfg_c.cascade.alpha


# ---------------------------------------------------------------------------
# load_config — errors
# ---------------------------------------------------------------------------


def test_missing_file_raises_filenotfound() -> None:
    with pytest.raises(FileNotFoundError, match="config file not found"):
        load_config("this_file_does_not_exist")


def test_resolve_false_keeps_interpolation_literal() -> None:
    """domains/blindsight/env.yaml uses ${train.noise_level} which is
    NOT defined in env.yaml itself — must opt out of resolve to load
    it standalone."""
    cfg = load_config("domains/blindsight/env", resolve=False)
    # subthreshold.baseline literally references ${train.noise_level}
    # at this point — checking the field exists, not its resolved value
    assert "subthreshold" in cfg.conditions


# ---------------------------------------------------------------------------
# _resolve_config_path
# ---------------------------------------------------------------------------


def test_resolve_bare_name_uses_config_root() -> None:
    path = _resolve_config_path("maps")
    assert path == CONFIG_ROOT / "config" / "maps.yaml"


def test_resolve_nested_bare_name() -> None:
    path = _resolve_config_path("domains/blindsight/training")
    assert path == CONFIG_ROOT / "config" / "domains/blindsight/training.yaml"


def test_resolve_absolute_path_passthrough() -> None:
    abs_path = CONFIG_ROOT / "config" / "maps.yaml"
    path = _resolve_config_path(abs_path)
    assert path == abs_path


# ---------------------------------------------------------------------------
# _apply_defaults — cycle detection
# ---------------------------------------------------------------------------


def test_cycle_detection_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two files that reference each other → ValueError, not stack overflow."""
    # Create a fake config tree
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "a.yaml").write_text("defaults:\n  - b\nfoo: 1\n")
    (cfg_dir / "b.yaml").write_text("defaults:\n  - a\nbar: 2\n")
    # We need maps.yaml present so _find_config_root works
    (cfg_dir / "maps.yaml").write_text("seed: 1\n")

    monkeypatch.setattr("maps.utils.config.CONFIG_ROOT", tmp_path)
    with pytest.raises(ValueError, match="cycle detected"):
        load_config("a")


# ---------------------------------------------------------------------------
# parse_overrides
# ---------------------------------------------------------------------------


def test_parse_overrides_valid() -> None:
    out = parse_overrides("a=1", "b.c=2.5", "nested.deep.key=hello")
    assert out == ["a=1", "b.c=2.5", "nested.deep.key=hello"]


def test_parse_overrides_missing_equals_raises() -> None:
    with pytest.raises(ValueError, match="key=value"):
        parse_overrides("invalid_token")
