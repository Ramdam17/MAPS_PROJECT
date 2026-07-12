"""Sprint 16.H — MARL env/runner/cli (vendored, CODE-ONLY).

``runner.py`` and ``base_runner.py`` have no import-time meltingpot dependency, so we
import them and check structure directly. ``env.py`` needs dmlab2d/meltingpot (absent
here) → its test skips. ``cli.py`` also needs the un-ported onpolicy plumbing → we
only assert it byte-compiles (syntax check without executing imports).
"""

from __future__ import annotations

import py_compile
from pathlib import Path

import numpy as np
import pytest

MARL = Path(__file__).resolve().parents[4] / "src" / "maps" / "domains" / "marl"


def test_runner_module_imports_and_structure():
    import maps.domains.marl.runner as runner
    from maps.domains.marl.base_runner import Runner

    assert issubclass(runner.MeltingpotRunner, Runner)
    # Runner-loop methods overridden by the meltingpot runner.
    for method in ("run", "warmup", "collect", "insert", "meta", "eval", "render"):
        assert callable(getattr(runner.MeltingpotRunner, method))
    # Module-level helpers.
    for helper in ("flatten_lists", "get_episode_parameters", "count_parameters", "_t2n"):
        assert callable(getattr(runner, helper))


def test_base_runner_has_maps_training_hooks():
    from maps.domains.marl.base_runner import Runner

    # The MAPS-specific + MAPPO scaffold methods must be present.
    for method in ("compute", "get_wager_objective", "train", "save", "restore"):
        assert callable(getattr(Runner, method))


def test_get_episode_parameters_lookup_and_default():
    from maps.domains.marl.runner import get_episode_parameters

    assert get_episode_parameters("territory__open") == (0.2, 1000)
    assert get_episode_parameters("allelopathic_harvest__open") == (1.0, 2000)
    # Unknown substrate → documented default.
    assert get_episode_parameters("does_not_exist") == (0.2, 1000)


def test_flatten_lists_concatenates_inner_arrays():
    from maps.domains.marl.runner import flatten_lists

    out = flatten_lists([[np.array([1, 2]), np.array([3])], [np.array([4, 5])]])
    assert len(out) == 2
    assert np.array_equal(out[0], np.array([1, 2, 3]))
    assert np.array_equal(out[1], np.array([4, 5]))


def test_energy_tracker_stubs_are_inert():
    """energy_tracker is absent from the reference → the stubs must be no-ops."""
    import maps.domains.marl.runner as runner

    tracker = runner.NvidiaEnergyTracker(project_name="x", output_dir="y", tracking_interval=10)
    assert tracker.log_point() is None
    eff = runner.MLModelEnergyEfficiency(tracker, model_name="m")
    assert eff.start_tracking() is None
    assert eff.stop_tracking() is None  # None → stop_get_results_tracker() skips its prints


def test_env_importable_only_with_meltingpot():
    pytest.importorskip("dmlab2d")
    pytest.importorskip("meltingpot")
    import maps.domains.marl.env as env

    assert hasattr(env, "env_creator")


def test_cli_byte_compiles():
    """cli.py needs the un-ported onpolicy plumbing to import; assert it compiles."""
    py_compile.compile(str(MARL / "cli.py"), doraise=True)


def test_env_byte_compiles():
    """env.py needs meltingpot to import; assert it at least compiles (syntax)."""
    py_compile.compile(str(MARL / "env.py"), doraise=True)
