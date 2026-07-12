"""Integration test for the AGL CLI 4-phase pipeline (Sprint 13.F).

Runs the whole pretrain → replicate → train High/Low → evaluate pipeline
end-to-end on a tiny config via Typer's CliRunner, and asserts exit 0 plus all
expected artifacts on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from maps.domains.agl.cli import app

runner = CliRunner()


def test_cli_single_run_produces_artifacts(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "--setting",
            "setting-1-baseline",
            "--seed",
            "42",
            "--output-dir",
            str(tmp_path),
            "-o",
            "train.n_epochs_pretrain=1",
            "-o",
            "train.batch_size_pretrain=6",
            "-o",
            "train.batch_size_training=6",
            "-o",
            "train.num_networks=4",
            "-o",
            "train.n_epochs_training_high=1",
            "-o",
            "train.n_epochs_training_low=1",
        ],
    )
    assert result.exit_code == 0, result.output

    out = tmp_path / "setting-1-baseline" / "seed-42"
    for name in (
        "pretrain_losses_1.npy",
        "pretrain_losses_2.npy",
        "first_order_reset.pt",
        "second_order_postpre.pt",
        "training_high_losses_1.npy",
        "training_high_precision.npy",
        "training_low_losses_1.npy",
        "summary.json",
    ):
        assert (out / name).is_file(), f"missing artifact {name}"

    summary = json.loads((out / "summary.json").read_text())
    assert summary["setting"] == "setting-1-baseline"
    assert summary["seed"] == 42
    assert summary["num_networks"] == 4
    assert set(summary["evaluation"]) >= {"high", "low", "overall"}
    assert summary["training_high"]["n_cells"] == 2
    assert summary["training_low"]["n_cells"] == 2


def test_cli_unknown_setting_errors(tmp_path: Path):
    result = runner.invoke(
        app,
        ["--setting", "setting-999-nope", "--seed", "42", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code != 0
