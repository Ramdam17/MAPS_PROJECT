"""Integration test for the SARL CLI (Sprint 14.H).

Runs the whole train → evaluate → save pipeline end-to-end on a tiny config via
Typer's CliRunner (real MinAtar env; first import ~60s).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from maps.domains.sarl.cli import app

runner = CliRunner()


def test_cli_single_run_produces_artifacts(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "--game",
            "breakout",
            "--setting",
            "1",
            "--seed",
            "42",
            "--num-frames",
            "150",
            "--eval-episodes",
            "2",
            "--output-dir",
            str(tmp_path),
            "-o",
            "training.replay_start_size=20",
            "-o",
            "training.batch_size=8",
            "-o",
            "training.replay_buffer_size=500",
        ],
    )
    assert result.exit_code == 0, result.output

    out = tmp_path / "breakout" / "setting-1-baseline" / "seed-42"
    assert (out / "policy_net.pt").is_file()
    assert (out / "episode_returns.npy").is_file()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["game"] == "breakout"
    assert summary["setting"] == "setting-1-baseline"
    assert summary["seed"] == 42
    assert summary["meta"] is False
    assert summary["num_frames"] >= 150
    assert "mean_return" in summary["eval"]


def test_cli_invalid_setting_errors(tmp_path: Path):
    result = runner.invoke(app, ["--setting", "9", "--output-dir", str(tmp_path)])
    assert result.exit_code != 0
