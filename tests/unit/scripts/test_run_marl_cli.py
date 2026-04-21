"""CLI-surface tests for ``scripts/run_marl.py`` (E.12).

Exercises argument validation (unknown substrate / setting, invalid
resume-from) without actually launching the MeltingPot env — we stop
before :func:`build_env_from_config` is reached.

The full end-to-end path requires the .venv-marl with meltingpot installed
and will be exercised on compute nodes via scripts/slurm/marl_array.sh.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner


_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_marl.py"


def _import_cli_module():
    """Load ``scripts/run_marl.py`` as a module so we can reach its ``app``."""
    spec = importlib.util.spec_from_file_location("maps_scripts_run_marl", _SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover — safety
        raise RuntimeError(f"cannot load {_SCRIPT_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["maps_scripts_run_marl"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cli_app():
    return _import_cli_module().app


def test_run_marl_rejects_unknown_substrate(cli_app):
    runner = CliRunner()
    result = runner.invoke(cli_app, ["--substrate", "bogus_substrate", "--setting", "baseline"])
    assert result.exit_code != 0
    assert "Unknown substrate" in (result.output or "") or "Unknown substrate" in (
        str(result.exception) if result.exception else ""
    )


def test_run_marl_rejects_unknown_setting(cli_app):
    runner = CliRunner()
    result = runner.invoke(
        cli_app, ["--substrate", "chemistry", "--setting", "bogus_setting"]
    )
    assert result.exit_code != 0


def test_run_marl_resume_from_raises_not_implemented(cli_app, tmp_path):
    runner = CliRunner()
    fake_ckpt = tmp_path / "ckpt.pt"
    fake_ckpt.write_bytes(b"stub")
    result = runner.invoke(
        cli_app,
        [
            "--substrate",
            "commons_harvest_closed",
            "--setting",
            "baseline",
            "--resume-from",
            str(fake_ckpt),
        ],
    )
    assert result.exit_code != 0
    # Either raised directly or captured by typer → ensure NotImplementedError surfaced.
    err = str(result.exception) + (result.output or "")
    assert "resume-from is not supported" in err or "NotImplementedError" in err


def test_run_marl_help_includes_paper_settings(cli_app):
    runner = CliRunner()
    result = runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0
    assert "--substrate" in result.output
    assert "--setting" in result.output
    assert "--num-env-steps" in result.output


def test_run_marl_resolves_setting_from_factorial():
    """Smoke-check the internal helper against the 6-setting factorial."""
    mod = _import_cli_module()
    from maps.utils import load_config

    factorial = load_config("experiments/factorial_marl")
    for sid in (
        "baseline",
        "cascade_1st_no_meta",
        "meta_no_cascade",
        "maps",
        "meta_cascade_2nd",
        "meta_cascade_both",
    ):
        setting = mod._resolve_setting(factorial, sid)
        assert setting.id == sid


def test_run_marl_resolve_setting_raises_on_unknown():
    mod = _import_cli_module()
    from maps.utils import load_config

    factorial = load_config("experiments/factorial_marl")
    with pytest.raises(Exception):  # typer.BadParameter, but imported lazily
        mod._resolve_setting(factorial, "not_a_real_setting")
