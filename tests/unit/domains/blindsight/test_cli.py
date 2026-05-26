"""Smoke tests for :mod:`maps.domains.blindsight.cli`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from maps.domains.blindsight.cli import _parse_seeds, _resolve_output_dir, app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_parse_seeds_simple_list() -> None:
    assert _parse_seeds("42,43,44") == [42, 43, 44]


def test_parse_seeds_whitespace_tolerant() -> None:
    assert _parse_seeds("  42 , 43 , 44 ") == [42, 43, 44]


def test_parse_seeds_empty_returns_none() -> None:
    assert _parse_seeds(None) is None
    assert _parse_seeds("") is None


def test_resolve_output_dir_uses_override_when_given(tmp_path: Path) -> None:
    out = _resolve_output_dir("setting-1-baseline", 42, tmp_path)
    assert out == tmp_path / "setting-1-baseline" / "seed-42"


def test_resolve_output_dir_uses_scratch_root_when_no_override() -> None:
    out = _resolve_output_dir("setting-6-full-maps", 7, None)
    assert out.parts[-1] == "seed-7"
    assert out.parts[-2] == "setting-6-full-maps"
    assert "blindsight" in out.parts


# ---------------------------------------------------------------------------
# CLI integration — full single-run smoke (uses Typer's CliRunner)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_cli_single_run_smoke(tmp_path: Path) -> None:
    """End-to-end : invoke the CLI on Setting 1 (smallest, baseline)
    with 2 epochs / batch 10 / eval 10. Verify exit code 0 + artifacts
    saved to the override output dir."""
    result = runner.invoke(
        app,
        [
            "--setting",
            "setting-1-baseline",
            "--seed",
            "42",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
            "-o",
            "train.n_epochs=2",
            "-o",
            "train.batch_size=10",
            "-o",
            "eval.patterns_number=10",
        ],
    )
    assert result.exit_code == 0, result.stdout

    run_dir = tmp_path / "setting-1-baseline" / "seed-42"
    assert run_dir.is_dir()
    assert (run_dir / "losses_1.npy").is_file()
    assert (run_dir / "losses_2.npy").is_file()
    assert (run_dir / "first_order.pt").is_file()
    assert (run_dir / "second_order.pt").is_file()

    summary_path = run_dir / "summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["setting"] == "setting-1-baseline"
    assert summary["seed"] == 42
    assert summary["n_epochs"] == 2
    assert "discrimination_accuracy" in summary
    assert "wager_accuracy" in summary
    assert set(summary["discrimination_accuracy"]) == {
        "superthreshold",
        "subthreshold",
        "low_vision",
    }


def test_cli_unknown_setting_raises(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "--setting",
            "setting-99-does-not-exist",
            "--output-dir",
            str(tmp_path),
            "-o",
            "train.n_epochs=1",
            "-o",
            "train.batch_size=10",
        ],
    )
    assert result.exit_code != 0


def test_cli_help_runs() -> None:
    """--help must work for discoverability."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Blindsight" in result.stdout
