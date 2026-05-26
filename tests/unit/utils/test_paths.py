"""Unit tests for :mod:`maps.utils.paths`."""

from __future__ import annotations

from pathlib import Path

import pytest

from maps.utils.config import CONFIG_ROOT
from maps.utils.paths import Paths, get_paths


@pytest.fixture(autouse=True)
def clear_paths_cache() -> None:
    """Clear lru_cache before each test so env-var overrides take effect."""
    get_paths.cache_clear()


def test_get_paths_returns_paths_dataclass() -> None:
    paths = get_paths()
    assert isinstance(paths, Paths)


def test_root_falls_back_to_config_root_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAPS_ROOT", raising=False)
    get_paths.cache_clear()
    paths = get_paths()
    assert paths.root == CONFIG_ROOT


def test_root_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MAPS_ROOT", str(tmp_path))
    get_paths.cache_clear()
    paths = get_paths()
    assert paths.root == tmp_path.resolve()


def test_relative_paths_resolved_against_root() -> None:
    paths = get_paths()
    # paths.yaml says data: data (relative)
    assert paths.data.is_absolute()
    assert paths.data == paths.root / "data"


def test_scratch_env_wins_over_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scratch = tmp_path / "scratch_hpc"
    scratch.mkdir()
    monkeypatch.setenv("SCRATCH", str(scratch))
    get_paths.cache_clear()
    paths = get_paths()
    assert paths.scratch_root == scratch.resolve()


def test_scratch_falls_back_to_yaml_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCRATCH", raising=False)
    get_paths.cache_clear()
    paths = get_paths()
    # paths.yaml's scratch_root is "outputs" → resolves to root/outputs
    assert paths.scratch_root == paths.root / "outputs"


def test_paths_dataclass_is_frozen() -> None:
    """`Paths` is `@dataclass(frozen=True)` so mutation raises
    :class:`dataclasses.FrozenInstanceError`."""
    from dataclasses import FrozenInstanceError

    paths = get_paths()
    with pytest.raises(FrozenInstanceError):
        paths.root = Path("/tmp")  # type: ignore[misc]
