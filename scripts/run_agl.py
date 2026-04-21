"""AGL experiment driver.

Loads the composed config (config/training/agl.yaml ← config/maps.yaml),
seeds every RNG, builds the networks described by ``cfg``, runs pre-training
for one of the four 2×2 factorial settings, and saves loss curves + final
model state under ``$SCRATCH/maps/outputs/agl/<setting>/seed-<seed>/`` (falls back to
``./outputs/agl/...`` when ``$SCRATCH`` is unset — dev boxes).

Usage
-----
    uv run python scripts/run_agl.py --setting both
    uv run python scripts/run_agl.py --setting neither --seed 43
    uv run python scripts/run_agl.py --setting both -o train.n_epochs_pretrain=20
    uv run python scripts/run_agl.py --all-settings

Notes
-----
AGL's reference ``pre_train`` restores the first-order network to its *initial*
weights at the end of the loop (AGL_TMLR.py L751). We preserve that behavior,
so ``first_order.pt`` saved here contains the **reset initial weights**, not
the weights at the end of pre-training. The useful training signal lives in
``second_order.pt`` (the wagering circuit trained against the first-order's
per-epoch output). See ``src/maps/experiments/agl/trainer.py`` docstring.

The ``--all-settings`` flag loops over the factorial_2x2 experiment config
and runs every (setting × seed) cell sequentially.
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

from maps.experiments.agl import AGLSetting, AGLTrainer
from maps.utils import configure_logging, get_paths, load_config, set_all_seeds

app = typer.Typer(add_completion=False, help=__doc__)
log = logging.getLogger("maps.run_agl")


def _run_one(cfg, setting: AGLSetting, seed: int, out_dir: Path) -> dict:
    """Run one (setting, seed) cell and save artifacts. Returns a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(seed)
    trainer = AGLTrainer(cfg, setting)
    trainer.build()

    t0 = time.perf_counter()
    losses_1, losses_2 = trainer.pre_train()
    train_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    eval_metrics = trainer.evaluate()
    eval_elapsed = time.perf_counter() - t1

    np.save(out_dir / "losses_1.npy", losses_1)
    np.save(out_dir / "losses_2.npy", losses_2)
    # first_order.pt holds the reset initial weights (reference L751 behavior).
    torch.save(trainer.first_order.state_dict(), out_dir / "first_order.pt")
    torch.save(trainer.second_order.state_dict(), out_dir / "second_order.pt")

    summary = {
        "setting": setting.id,
        "seed": seed,
        "n_epochs": int(cfg.train.n_epochs_pretrain),
        "loss_1_final": float(losses_1[-1]),
        "loss_2_final": float(losses_2[-1]),
        "loss_1_min": float(losses_1.min()),
        "loss_2_min": float(losses_2.min()) if setting.second_order else 0.0,
        "elapsed_seconds": train_elapsed,
        "eval": eval_metrics,
        "eval_elapsed_seconds": eval_elapsed,
        "first_order_reset_to_initial": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(
        "[%s | seed=%d] %d epochs in %.1fs (+%.1fs eval), "
        "loss_1[-1]=%.3f, loss_2[-1]=%.3f, classif=%.3f, wager=%.3f",
        setting.id,
        seed,
        cfg.train.n_epochs_pretrain,
        train_elapsed,
        eval_elapsed,
        losses_1[-1],
        losses_2[-1],
        eval_metrics.get("classification_precision", float("nan")),
        eval_metrics.get("wager_accuracy", float("nan")),
    )
    return summary


def _setting_from_cfg(factorial_cfg, setting_id: str) -> AGLSetting:
    for s in factorial_cfg.settings:
        if s.id == setting_id:
            return AGLSetting.from_dict(s)
    valid = [s.id for s in factorial_cfg.settings]
    raise typer.BadParameter(f"Unknown setting {setting_id!r}. Valid: {valid}")


@app.command()
def main(
    setting: str = typer.Option(
        "both",
        help="Factorial setting id: neither | cascade_only | second_order_only | both",
    ),
    all_settings: bool = typer.Option(
        False, "--all-settings", help="Loop over every factorial setting × seed."
    ),
    seed: int | None = typer.Option(None, help="Override seed (single-setting mode only)."),
    seeds: str | None = typer.Option(
        None,
        "--seeds",
        help="Comma-separated seed list overriding factorial.seeds in --all-settings mode (e.g. '42,43,...,51').",
    ),
    override: list[str] = typer.Option(  # noqa: B008
        [],
        "--override",
        "-o",
        help="Hydra-style override, e.g. `-o train.n_epochs_pretrain=10`. Repeatable.",
    ),
    log_level: str = typer.Option("INFO", help="Python logging level"),
) -> None:
    configure_logging(level=log_level)

    cfg = load_config("training/agl", overrides=list(override))
    factorial = load_config("experiments/factorial_2x2")
    paths = get_paths()
    paths.ensure_dirs()

    base_out = paths.scratch_root / "maps" / "outputs" / "agl"

    if all_settings:
        seed_pool = (
            [int(x) for x in seeds.split(",")] if seeds is not None else list(factorial.seeds)
        )
        runs = [(AGLSetting.from_dict(s), s_idx) for s_idx in seed_pool for s in factorial.settings]
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
