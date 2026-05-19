"""Blindsight experiment driver.

Loads the composed config (config/training/blindsight.yaml ← config/maps.yaml),
seeds every RNG, builds the networks described by ``cfg``, runs pre-training
for one paper Table 5 factorial setting, and saves loss curves + final
model state under ``$SCRATCH/maps/outputs/blindsight/<setting>/seed-<seed>/`` (falls
back to ``./outputs/blindsight/...`` when ``$SCRATCH`` is unset — dev boxes).

Two factorial configs are shipped:

- ``experiments/factorial_6cell`` (default) — paper Table 5 settings 1-6,
  including the headline Setting 4 (MAPS) and the asymmetric Setting 5.
- ``experiments/factorial_2x2`` — legacy 4-cell schema (paper settings 1/2/3/6
  via symmetric cascade). Retained for reproducibility of pre-2026-05 archives.

Usage
-----
    uv run python scripts/run_blindsight.py --setting setting-4-maps
    uv run python scripts/run_blindsight.py --setting setting-5-cascade-2nd --seed 43
    uv run python scripts/run_blindsight.py --setting setting-6-full-maps -o train.n_epochs=20

    # Legacy 4-cell config (pre-2026-05 archive parity):
    uv run python scripts/run_blindsight.py \\
        --factorial-config experiments/factorial_2x2 --setting both

The ``--all-settings`` flag loops over the loaded factorial's settings and
runs every (setting × seed) cell sequentially. Use SLURM array jobs in
production (see ``scripts/slurm/blindsight_array.sh``) rather than the
single-process loop.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import typer
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maps.experiments.blindsight import BlindsightSetting, BlindsightTrainer
from maps.utils import configure_logging, get_paths, load_config, set_all_seeds

app = typer.Typer(add_completion=False, help=__doc__)
log = logging.getLogger("maps.run_blindsight")


def _run_one(cfg, setting: BlindsightSetting, seed: int, out_dir: Path) -> dict:
    """Run one (setting, seed) cell and save artifacts. Returns a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(seed)
    trainer = BlindsightTrainer(cfg, setting)
    trainer.build()

    t0 = time.perf_counter()
    losses_1, losses_2 = trainer.pre_train()
    train_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    eval_metrics = trainer.evaluate()
    eval_elapsed = time.perf_counter() - t1

    np.save(out_dir / "losses_1.npy", losses_1)
    np.save(out_dir / "losses_2.npy", losses_2)
    torch.save(trainer.first_order.state_dict(), out_dir / "first_order.pt")
    torch.save(trainer.second_order.state_dict(), out_dir / "second_order.pt")

    summary = {
        "setting": setting.id,
        "seed": seed,
        "n_epochs": int(cfg.train.n_epochs),
        "loss_1_final": float(losses_1[-1]),
        "loss_2_final": float(losses_2[-1]),
        "loss_1_min": float(losses_1.min()),
        "loss_2_min": float(losses_2.min()) if setting.second_order else 0.0,
        "elapsed_seconds": train_elapsed,
        "eval": eval_metrics,
        "eval_elapsed_seconds": eval_elapsed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(
        "[%s | seed=%d] %d epochs in %.1fs (+%.1fs eval), "
        "loss_1[-1]=%.3f, loss_2[-1]=%.3f, super.wager=%.3f",
        setting.id,
        seed,
        cfg.train.n_epochs,
        train_elapsed,
        eval_elapsed,
        losses_1[-1],
        losses_2[-1],
        eval_metrics["superthreshold"].get("wager_accuracy", float("nan")),
    )
    return summary


def _setting_from_cfg(factorial_cfg, setting_id: str) -> BlindsightSetting:
    for s in factorial_cfg.settings:
        if s.id == setting_id:
            return BlindsightSetting.from_dict(s)
    valid = [s.id for s in factorial_cfg.settings]
    raise typer.BadParameter(f"Unknown setting {setting_id!r}. Valid: {valid}")


def _resolve_seed_pool(factorial, *, seeds_cli: str | None) -> list[int]:
    """Resolve the seed pool from CLI > YAML explicit list > YAML n_seeds count.

    Falls back to ``range(n_seeds)`` when the YAML omits an explicit ``seeds``
    list, which keeps the 6-cell config compact (500-seed pools as range
    rather than 500 literal entries).
    """
    if seeds_cli is not None:
        return [int(x) for x in seeds_cli.split(",")]
    explicit = factorial.get("seeds", None)
    if explicit is not None:
        return [int(x) for x in explicit]
    return list(range(int(factorial.n_seeds)))


@app.command()
def main(
    setting: str = typer.Option(
        "setting-6-full-maps",
        help=(
            "Factorial setting id. New 6-cell schema (default factorial): "
            "setting-1-baseline | setting-2-cascade-1st | setting-3-second-order-only | "
            "setting-4-maps | setting-5-cascade-2nd | setting-6-full-maps. "
            "Legacy 2×2 schema (--factorial-config experiments/factorial_2x2): "
            "neither | cascade_only | second_order_only | both."
        ),
    ),
    factorial_config: str = typer.Option(
        "experiments/factorial_6cell",
        "--factorial-config",
        help=(
            "Experiment YAML defining the factorial settings and seed pool. "
            "Default: experiments/factorial_6cell (paper Table 5 6 cells). "
            "Use experiments/factorial_2x2 for pre-2026-05 archive parity."
        ),
    ),
    all_settings: bool = typer.Option(
        False, "--all-settings", help="Loop over every factorial setting × seed."
    ),
    seed: int | None = typer.Option(None, help="Override seed (single-setting mode only)."),
    seeds: str | None = typer.Option(
        None,
        "--seeds",
        help="Comma-separated seed list overriding the factorial seed pool in --all-settings mode (e.g. '42,43,...,51').",
    ),
    override: list[str] = typer.Option(  # noqa: B008
        [],
        "--override",
        "-o",
        help="Hydra-style override, e.g. `-o train.n_epochs=10`. Repeatable.",
    ),
    output_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        help="Override base output dir. Default: $SCRATCH/maps/outputs/blindsight/ (or ./outputs/blindsight/... when $SCRATCH unset). The <setting>/seed-<seed>/ tail is appended automatically.",
    ),
    log_level: str = typer.Option("INFO", help="Python logging level"),
) -> None:
    configure_logging(level=log_level)

    cfg = load_config("training/blindsight", overrides=list(override))
    factorial = load_config(factorial_config)
    paths = get_paths()
    paths.ensure_dirs()

    base_out = (
        output_dir if output_dir is not None else paths.scratch_root / "maps" / "outputs" / "blindsight"
    )

    if all_settings:
        seed_pool = _resolve_seed_pool(factorial, seeds_cli=seeds)
        runs = [
            (BlindsightSetting.from_dict(s), s_idx)
            for s_idx in seed_pool
            for s in factorial.settings
        ]
        log.info(
            "Running %d cells (%d settings × %d seeds): seeds=%s",
            len(runs),
            len(factorial.settings),
            len(seed_pool),
            seed_pool,
        )
        for s, sd in runs:
            _run_one(cfg, s, sd, base_out / s.id / f"seed-{sd}")
        return

    # Single setting.
    s = _setting_from_cfg(factorial, setting)
    sd = seed if seed is not None else int(cfg.seed)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    _run_one(cfg, s, sd, base_out / s.id / f"seed-{sd}")


if __name__ == "__main__":
    app()
