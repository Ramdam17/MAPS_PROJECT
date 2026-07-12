"""Integration test for the SARL+CL CLI curriculum (Sprint 15.H).

Runs a tiny 2-stage curriculum end-to-end via CliRunner (real MinAtar env).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from maps.domains.sarl_cl.cli import app

runner = CliRunner()


def test_cli_curriculum_produces_artifacts(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "--games",
            "breakout,space_invaders",
            "--setting",
            "1",
            "--seed",
            "42",
            "--frames-per-stage",
            "120",
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
    out = tmp_path / "breakout-space_invaders" / "setting-1-baseline" / "seed-42"
    assert (out / "policy_net_final.pt").is_file()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["games"] == ["breakout", "space_invaders"]
    assert len(summary["stages"]) == 2
    assert summary["stages"][0]["game"] == "breakout"


def test_cli_invalid_setting_errors(tmp_path: Path):
    result = runner.invoke(app, ["--setting", "9", "--output-dir", str(tmp_path)])
    assert result.exit_code != 0
