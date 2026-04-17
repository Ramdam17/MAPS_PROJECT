"""Unit tests for maps.utils.paths — filesystem layout resolution."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from maps.utils.paths import Paths, get_paths


def test_get_paths_returns_absolute(tmp_path: Path):
    paths = get_paths(root=tmp_path)
    assert paths.root == tmp_path.resolve()
    for name in ("data", "outputs", "models", "logs", "figures", "reports"):
        sub = getattr(paths, name)
        assert sub.is_absolute()
        assert sub.is_relative_to(paths.root)


def test_ensure_dirs_creates_everything(tmp_path: Path):
    paths = get_paths(root=tmp_path)
    for name in ("data", "outputs", "models", "logs"):
        assert not getattr(paths, name).exists()
    paths.ensure_dirs()
    for name in ("data", "outputs", "models", "logs", "figures", "reports"):
        assert getattr(paths, name).is_dir()


def test_paths_is_frozen(tmp_path: Path):
    """Paths is immutable — prevents accidental mutation mid-run."""
    paths: Paths = get_paths(root=tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        paths.root = tmp_path / "elsewhere"  # type: ignore[misc]


def test_figures_is_under_outputs(tmp_path: Path):
    """paths.yaml declares `figures: outputs/figures` — keep that relation."""
    paths: Paths = get_paths(root=tmp_path)
    assert paths.figures.is_relative_to(paths.outputs)
