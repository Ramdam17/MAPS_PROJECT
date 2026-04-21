"""AGL experiment driver — full 3-phase protocol (D.28.e).

Loads the composed config (config/training/agl.yaml ← config/maps.yaml),
seeds every RNG, builds the networks described by ``cfg``, runs the full
**3-phase AGL protocol** (paper §A.2 + Table 10):

1. **Pretrain** — ``n_epochs_pretrain`` epochs on random-grammar words.
2. **Replication** — ``num_networks`` deep copies of the post-pretrain
   networks (:class:`AGLNetworkPool`).
3. **Training** — cells ``[0:num_networks//2]`` train for
   ``n_epochs_training_high`` epochs (High Awareness, paper 12) on Grammar-A;
   cells ``[num_networks//2:num_networks]`` train for
   ``n_epochs_training_low`` epochs (Low Awareness, paper 3).
4. **Testing** — :meth:`AGLTrainer.evaluate_pool` on the 20-cell pool.

Output goes under
``$SCRATCH/maps/outputs/agl/<setting>/seed-<seed>/``
(or ``./outputs/agl/...`` when ``$SCRATCH`` is unset).

Usage
-----
    uv run python scripts/run_agl.py --setting both
    uv run python scripts/run_agl.py --setting both --seed 43
    uv run python scripts/run_agl.py --setting both -o train.n_epochs_pretrain=30
    uv run python scripts/run_agl.py --all-settings
    uv run python scripts/run_agl.py --setting both --output-dir /scratch/xxx/expA

Notes
-----
AGL's reference ``pre_train`` restores the first-order network to its *initial*
weights at the end of the loop (paper code L751). We preserve that behavior;
the downstream pool is built from the **reset initial** first-order weights
along with the fully-trained second-order circuit (see paper §A.2 and the
:mod:`maps.experiments.agl.pool` docstring).
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

from maps.experiments.agl import AGLNetworkPool, AGLSetting, AGLTrainer
from maps.utils import configure_logging, get_paths, load_config, set_all_seeds

app = typer.Typer(add_completion=False, help=__doc__)
log = logging.getLogger("maps.run_agl")


def _run_one(cfg, setting: AGLSetting, seed: int, out_dir: Path) -> dict:
    """Run one (setting, seed) cell through the full 3-phase protocol.

    Returns a nested summary dict :
    ```
    {
      "setting": ..., "seed": ...,
      "pretrain": {n_epochs, loss_1_final, loss_2_final, elapsed_seconds},
      "training_high": {n_epochs, n_cells, precision_final_mean, elapsed_seconds},
      "training_low":  {n_epochs, n_cells, precision_final_mean, elapsed_seconds},
      "evaluation": {high: {...}, low: {...}, overall: {...}, elapsed_seconds},
      "num_networks": 20,
    }
    ```
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(seed)
    trainer = AGLTrainer(cfg, setting)
    trainer.build()

    # ── Phase 1: Pretrain (random grammar) ──────────────────────────────────
    t0 = time.perf_counter()
    losses_1_pre, losses_2_pre = trainer.pre_train()
    pretrain_elapsed = time.perf_counter() - t0
    np.save(out_dir / "pretrain_losses_1.npy", losses_1_pre)
    np.save(out_dir / "pretrain_losses_2.npy", losses_2_pre)
    # Reset first-order (student L751 behaviour).
    torch.save(trainer.first_order.state_dict(), out_dir / "first_order_reset.pt")
    torch.save(trainer.second_order.state_dict(), out_dir / "second_order_postpre.pt")

    # ── Phase 2: Replicate into num_networks cells ─────────────────────────
    num_networks = int(cfg.train.num_networks)
    n_epochs_high = int(cfg.train.n_epochs_training_high)
    n_epochs_low = int(cfg.train.n_epochs_training_low)
    pool = AGLNetworkPool(trainer, num_networks=num_networks)

    # ── Phase 3: Training (High / Low awareness tiers) ──────────────────────
    t1 = time.perf_counter()
    high_metrics = pool.train_range(
        start=0, end=num_networks // 2, n_epochs=n_epochs_high
    )
    training_high_elapsed = time.perf_counter() - t1

    t2 = time.perf_counter()
    low_metrics = pool.train_range(
        start=num_networks // 2, end=num_networks, n_epochs=n_epochs_low
    )
    training_low_elapsed = time.perf_counter() - t2

    # Persist per-tier training loss arrays for post-hoc analysis.
    np.save(out_dir / "training_high_losses_1.npy", high_metrics["losses_1"])
    np.save(out_dir / "training_high_losses_2.npy", high_metrics["losses_2"])
    np.save(out_dir / "training_high_precision.npy", high_metrics["precision"])
    np.save(out_dir / "training_low_losses_1.npy", low_metrics["losses_1"])
    np.save(out_dir / "training_low_losses_2.npy", low_metrics["losses_2"])
    np.save(out_dir / "training_low_precision.npy", low_metrics["precision"])

    # ── Phase 4: Testing (Grammar A + Grammar B per cell) ──────────────────
    t3 = time.perf_counter()
    eval_metrics = trainer.evaluate_pool(pool)
    eval_elapsed = time.perf_counter() - t3

    summary = {
        "setting": setting.id,
        "seed": seed,
        "num_networks": num_networks,
        "pretrain": {
            "n_epochs": int(cfg.train.n_epochs_pretrain),
            "loss_1_final": float(losses_1_pre[-1]),
            "loss_2_final": float(losses_2_pre[-1]),
            "loss_1_min": float(losses_1_pre.min()),
            "loss_2_min": float(losses_2_pre.min()) if setting.second_order else 0.0,
            "elapsed_seconds": pretrain_elapsed,
        },
        "training_high": {
            "n_epochs": n_epochs_high,
            "n_cells": num_networks // 2,
            "precision_final_mean": float(high_metrics["precision"][:, -1].mean())
            if n_epochs_high > 0
            else 0.0,
            "loss_1_final_mean": float(high_metrics["losses_1"][:, -1].mean())
            if n_epochs_high > 0
            else 0.0,
            "elapsed_seconds": training_high_elapsed,
        },
        "training_low": {
            "n_epochs": n_epochs_low,
            "n_cells": num_networks - num_networks // 2,
            "precision_final_mean": float(low_metrics["precision"][:, -1].mean())
            if n_epochs_low > 0
            else 0.0,
            "loss_1_final_mean": float(low_metrics["losses_1"][:, -1].mean())
            if n_epochs_low > 0
            else 0.0,
            "elapsed_seconds": training_low_elapsed,
        },
        "evaluation": {
            **eval_metrics,
            "elapsed_seconds": eval_elapsed,
        },
        "meta_frozen_in_training": bool(
            cfg.train.get("train_meta_frozen_in_training", True)
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    log.info(
        "[%s | seed=%d] pre %.1fs + train-high %.1fs + train-low %.1fs + eval %.1fs | "
        "high.precision_1st=%.3f high.wager=%.3f | low.precision_1st=%.3f low.wager=%.3f",
        setting.id,
        seed,
        pretrain_elapsed,
        training_high_elapsed,
        training_low_elapsed,
        eval_elapsed,
        float(eval_metrics.get("high", {}).get("precision_1st", float("nan"))),
        float(eval_metrics.get("high", {}).get("wager_accuracy", float("nan"))),
        float(eval_metrics.get("low", {}).get("precision_1st", float("nan"))),
        float(eval_metrics.get("low", {}).get("wager_accuracy", float("nan"))),
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
    output_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        help="Override base output dir. Default: $SCRATCH/maps/outputs/agl/ "
        "(or ./outputs/agl/... when $SCRATCH unset). The <setting>/seed-<seed>/ "
        "tail is appended automatically.",
    ),
    log_level: str = typer.Option("INFO", help="Python logging level"),
) -> None:
    configure_logging(level=log_level)

    cfg = load_config("training/agl", overrides=list(override))
    factorial = load_config("experiments/factorial_2x2")
    paths = get_paths()
    paths.ensure_dirs()

    base_out = (
        output_dir if output_dir is not None else paths.scratch_root / "maps" / "outputs" / "agl"
    )

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
