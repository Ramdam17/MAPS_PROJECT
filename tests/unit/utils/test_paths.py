"""Unit tests for maps.utils.paths — filesystem layout resolution."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from maps.utils.paths import Paths, _discover_root, get_paths


def test_get_paths_returns_absolute(tmp_path: Path):
    paths = get_paths(root=tmp_path)
    assert paths.root == tmp_path.resolve()
    for name in ("data", "outputs", "models", "logs", "figures", "reports"):
        sub = getattr(paths, name)
        assert sub.is_absolute()
        assert sub.is_relative_to(paths.root)


def test_scratch_root_falls_back_to_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Without $SCRATCH, scratch_root follows the yaml default (`outputs`)."""
    monkeypatch.delenv("SCRATCH", raising=False)
    paths = get_paths(root=tmp_path)
    assert paths.scratch_root == paths.outputs


def test_scratch_root_honors_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """On HPC, $SCRATCH wins over the yaml — scratch_root is the env path."""
    # Sibling of project root — mimics DRAC layout where $SCRATCH is outside /project.
    project = tmp_path / "project"
    project.mkdir()
    scratch = tmp_path / "scratch" / "rram17"
    scratch.mkdir(parents=True)
    monkeypatch.setenv("SCRATCH", str(scratch))
    paths = get_paths(root=project)
    assert paths.scratch_root == scratch.resolve()
    # Env override is absolute and outside project root.
    assert not paths.scratch_root.is_relative_to(paths.root)


def test_ensure_dirs_skips_scratch_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """ensure_dirs never touches scratch_root — on HPC it pre-exists and is not ours."""
    scratch = tmp_path / "phantom_scratch"
    monkeypatch.setenv("SCRATCH", str(scratch))
    assert not scratch.exists()
    paths = get_paths(root=tmp_path)
    paths.ensure_dirs()
    assert not scratch.exists(), "ensure_dirs must not create scratch_root"


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


def test_discover_root_honors_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`MAPS_ROOT` env var wins over cwd-based discovery (useful in CI)."""
    monkeypatch.setenv("MAPS_ROOT", str(tmp_path))
    assert _discover_root() == tmp_path.resolve()


def test_discover_root_walks_up_to_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When run from a subdir, discovery finds the nearest ancestor with config/paths.yaml."""
    monkeypatch.delenv("MAPS_ROOT", raising=False)

    # Build a fake project: tmp_path/config/paths.yaml + tmp_path/sub/sub2/.
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "paths.yaml").write_text("root: .\n")
    deep = tmp_path / "sub" / "sub2"
    deep.mkdir(parents=True)

    monkeypatch.chdir(deep)
    assert _discover_root() == tmp_path.resolve()


def test_discover_root_falls_back_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """If no ancestor has config/paths.yaml, discovery returns cwd unchanged."""
    monkeypatch.delenv("MAPS_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    # No config/paths.yaml anywhere above tmp_path — expect cwd.
    assert _discover_root() == tmp_path.resolve()
