"""Blindsight hyperparameter sweep orchestrator.

Grid B + n_epochs (per ``docs/benchmarks/blindsight-hyperparam-analysis.md``)
36 cells × 6 settings × 500 seeds = 108 000 runs.

**Skip-if-done.** Each (cell, setting, seed) tuple is committed to disk
as ``summary.json`` ; the script skips any combination that already has
a summary file. This makes the sweep **interruptible and resumable** —
Ctrl+C at any time, re-launch later, picks up exactly where it stopped.

**In-process.** Calls :class:`BlindsightTrainer` directly via Python
imports, **not** via ``subprocess``. Each subprocess launch costs ~1.5s
of Python/torch import overhead ; over 108k runs that's 45 hours wasted.
In-process keeps the overhead at zero.

**CPU only.** MPS was empirically non-deterministic on this workload
(seed 42 gave wager 0.730 on MPS vs 0.820 on CPU). Locked to CPU.

Usage
-----

.. code-block:: bash

    # Full sweep — runs until done (or until Ctrl+C). Re-launch resumes.
    uv run python scripts/bench/blindsight_sweep.py

    # Dry run — print the cells/runs that WOULD execute, don't run.
    uv run python scripts/bench/blindsight_sweep.py --dry-run

    # Probe — small grid for end-to-end validation.
    uv run python scripts/bench/blindsight_sweep.py --probe

    # Run a specific cell only.
    uv run python scripts/bench/blindsight_sweep.py --cell h1-40_h2-100_nwu-1_ne-200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from maps.domains.blindsight.trainer import (
    SETTINGS_REGISTRY,
    BlindsightTrainer,
)
from maps.utils.config import load_config
from maps.utils.logging_setup import configure_logging
from maps.utils.seeding import set_all_seeds

logger = logging.getLogger("bench.blindsight")

# Per-worker cache of resolved cfg (keyed by cell_id). Populated lazily
# inside worker processes ; module-level so it survives across calls in
# the same worker.
_WORKER_CFG_CACHE: dict[str, DictConfig] = {}


# ---------------------------------------------------------------------------
# Grid B + n_epochs (per analysis doc)
# ---------------------------------------------------------------------------

GRID_B = {
    "first_order.hidden_dim": [40, 60, 100],
    "second_order.hidden_dim": [0, 60, 100],
    "second_order.n_wager_units": [1, 2],
    "train.n_epochs": [200, 500],
}

SETTINGS_ORDER = [
    # Fast settings first → quick partial results visible
    "setting-1-baseline",
    "setting-3-second-order-only",
    # Medium (one cascade side)
    "setting-2-cascade-1st",
    "setting-4-maps-1st",
    "setting-5-cascade-2nd",
    # Heaviest last (cascade both sides)
    "setting-6-full-maps",
]

SEEDS = list(range(42, 542))  # 500 seeds, paper convention

OUT_ROOT = Path("outputs/blindsight-benchmark")


# ---------------------------------------------------------------------------
# Cell ID + path helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One hyperparameter combination."""

    first_hidden: int
    second_hidden: int
    n_wager_units: int
    n_epochs: int

    @property
    def cell_id(self) -> str:
        return (
            f"h1-{self.first_hidden}"
            f"_h2-{self.second_hidden}"
            f"_nwu-{self.n_wager_units}"
            f"_ne-{self.n_epochs}"
        )

    def overrides(self) -> list[str]:
        return [
            f"first_order.hidden_dim={self.first_hidden}",
            f"second_order.hidden_dim={self.second_hidden}",
            f"second_order.n_wager_units={self.n_wager_units}",
            f"train.n_epochs={self.n_epochs}",
        ]


def all_cells() -> list[Cell]:
    return [
        Cell(*combo)
        for combo in product(
            GRID_B["first_order.hidden_dim"],
            GRID_B["second_order.hidden_dim"],
            GRID_B["second_order.n_wager_units"],
            GRID_B["train.n_epochs"],
        )
    ]


def cell_dir(cell: Cell) -> Path:
    return OUT_ROOT / cell.cell_id


def run_dir(cell: Cell, setting_id: str, seed: int) -> Path:
    return cell_dir(cell) / setting_id / f"seed-{seed}"


def is_done(cell: Cell, setting_id: str, seed: int) -> bool:
    return (run_dir(cell, setting_id, seed) / "summary.json").exists()


# ---------------------------------------------------------------------------
# Per-run execution (in-process)
# ---------------------------------------------------------------------------


def _build_cell_cfg(cell: Cell) -> DictConfig:
    """Build a fully resolved DictConfig for this cell (overrides applied)."""
    training_cfg = load_config("domains/blindsight/training")
    env_cfg = load_config("domains/blindsight/env", resolve=False)
    merged = OmegaConf.merge(training_cfg, env_cfg)
    merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(cell.overrides()))
    OmegaConf.resolve(merged)
    return merged  # type: ignore[return-value]


def execute_one(cell: Cell, setting_id: str, seed: int, cfg: DictConfig) -> dict:
    """Build → train → evaluate → save. Returns the summary dict."""
    setting = SETTINGS_REGISTRY[setting_id]
    set_all_seeds(seed)

    trainer = BlindsightTrainer(setting=setting, seed=seed, cfg=cfg, device="cpu")
    trainer.build()

    t0 = time.perf_counter()
    train_metrics = trainer.train()
    eval_metrics = trainer.evaluate()
    wall = time.perf_counter() - t0

    out = run_dir(cell, setting_id, seed)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "losses_1.npy", train_metrics.losses_1)
    np.save(out / "losses_2.npy", train_metrics.losses_2)
    # .pt files serve as "this run is done" checkpoints + post-hoc analysis
    assert trainer.first_order is not None
    assert trainer.second_order is not None
    torch.save(trainer.first_order.state_dict(), out / "first_order.pt")
    torch.save(trainer.second_order.state_dict(), out / "second_order.pt")

    summary = {
        "cell_id": cell.cell_id,
        "first_hidden": cell.first_hidden,
        "second_hidden": cell.second_hidden,
        "n_wager_units": cell.n_wager_units,
        "n_epochs": cell.n_epochs,
        "setting": setting_id,
        "seed": seed,
        "batch_size": int(cfg.train.batch_size),
        "first_order_loss": cfg.first_order_loss.kind,
        "final_loss_1": float(train_metrics.losses_1[-1]),
        "final_loss_2": float(train_metrics.losses_2[-1]),
        "discrimination_accuracy": eval_metrics.discrimination_accuracy,
        "wager_accuracy": eval_metrics.wager_accuracy,
        "wall_seconds": wall,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------


def _worker_init() -> None:
    """ProcessPoolExecutor initializer — once per worker subprocess.

    Forces single-threaded PyTorch ops. Without this, every worker
    would try to use 12 perf cores via PyTorch's internal threading,
    causing massive over-subscription (8 workers × 12 threads = 96
    thread requests on 12 cores). Single thread per worker means
    multi-process gives us the parallelism cleanly.
    """
    torch.set_num_threads(1)
    # Per-epoch INFO logging would spam 36k+ lines per process. Mute.
    logging.getLogger("maps.domains.blindsight.trainer").setLevel(logging.WARNING)
    logging.getLogger("maps.utils.seeding").setLevel(logging.WARNING)


def _worker_execute(args: tuple[Cell, str, int]) -> dict | None:
    """Worker entry. Returns summary dict (or None if already done)."""
    cell, setting_id, seed = args
    if is_done(cell, setting_id, seed):
        return None  # race-safe : double-check after queueing
    if cell.cell_id not in _WORKER_CFG_CACHE:
        _WORKER_CFG_CACHE[cell.cell_id] = _build_cell_cfg(cell)
    cfg = _WORKER_CFG_CACHE[cell.cell_id]
    return execute_one(cell, setting_id, seed, cfg)


def sweep(
    cells: list[Cell],
    settings: list[str],
    seeds: list[int],
    *,
    dry_run: bool = False,
    workers: int = 1,
) -> None:
    """Iterate cells × settings × seeds, skipping completed runs.

    When ``workers > 1`` uses :class:`ProcessPoolExecutor` to parallelise
    runs across worker subprocesses. Each worker forces
    ``torch.set_num_threads(1)`` to avoid intra-op × inter-process
    contention.
    """
    total = len(cells) * len(settings) * len(seeds)
    to_do = [
        (cell, setting, seed)
        for cell in cells
        for setting in settings
        for seed in seeds
        if not is_done(cell, setting, seed)
    ]
    done = total - len(to_do)
    logger.info(
        "Sweep plan : %d cells × %d settings × %d seeds = %d runs (%d already done, %d to do)",
        len(cells),
        len(settings),
        len(seeds),
        total,
        done,
        len(to_do),
    )
    logger.info(
        "Parallelism : workers=%d (Mac CPU count : %d)",
        workers,
        os.cpu_count() or 1,
    )

    if dry_run:
        logger.info("Dry run — listing first 20 to-do tuples then exiting")
        for cell, setting, seed in to_do[:20]:
            logger.info("  %s | %s | seed=%d", cell.cell_id, setting, seed)
        return

    if not to_do:
        logger.info("All runs already done — nothing to execute.")
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "plan.json").write_text(
        json.dumps(
            {
                "grid_b": GRID_B,
                "cells": [c.cell_id for c in cells],
                "settings": settings,
                "n_seeds": len(seeds),
                "total_runs": total,
                "workers": workers,
            },
            indent=2,
            sort_keys=True,
        )
    )

    t_start = time.perf_counter()
    n_run = 0

    def _log_progress(cell, setting, seed, summary):
        nonlocal n_run
        if n_run % 50 == 0 or n_run < 5:
            elapsed = time.perf_counter() - t_start
            rate = n_run / elapsed if elapsed > 0 else 0
            eta_s = (len(to_do) - n_run) / rate if rate > 0 else 0
            logger.info(
                "[%d/%d] cell=%s setting=%s seed=%d  discrim=%.3f wager=%.3f  "
                "(rate %.2f/s, ETA %.0f min)",
                n_run,
                len(to_do),
                cell.cell_id,
                setting,
                seed,
                summary["discrimination_accuracy"].get("superthreshold", 0),
                summary["wager_accuracy"].get("superthreshold", 0),
                rate,
                eta_s / 60,
            )

    if workers <= 1:
        # Sequential fallback (debug / single-cell mode)
        cfg_cache: dict[str, DictConfig] = {}
        for cell, setting, seed in to_do:
            if cell.cell_id not in cfg_cache:
                cfg_cache[cell.cell_id] = _build_cell_cfg(cell)
            cfg = cfg_cache[cell.cell_id]
            try:
                summary = execute_one(cell, setting, seed, cfg)
            except KeyboardInterrupt:
                logger.warning(
                    "Interrupted at %s | %s | seed=%d — resumable",
                    cell.cell_id,
                    setting,
                    seed,
                )
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error("FAILED %s | %s | seed=%d : %s", cell.cell_id, setting, seed, exc)
                continue
            n_run += 1
            _log_progress(cell, setting, seed, summary)
    else:
        # Parallel — ProcessPoolExecutor with worker init forcing single-thread torch.
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
            futures = {
                pool.submit(_worker_execute, (cell, setting, seed)): (cell, setting, seed)
                for cell, setting, seed in to_do
            }
            try:
                for fut in as_completed(futures):
                    cell, setting, seed = futures[fut]
                    try:
                        summary = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "FAILED %s | %s | seed=%d : %s", cell.cell_id, setting, seed, exc
                        )
                        continue
                    if summary is None:
                        continue  # was already done
                    n_run += 1
                    _log_progress(cell, setting, seed, summary)
            except KeyboardInterrupt:
                logger.warning(
                    "Interrupted — cancelling pending futures (in-flight runs finish naturally)"
                )
                pool.shutdown(wait=False, cancel_futures=True)
                raise

    elapsed = time.perf_counter() - t_start
    logger.info("✓ Sweep done : %d runs in %.0f min", n_run, elapsed / 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan + 20 to-do tuples, don't run.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Tiny sub-grid for end-to-end validation (2 cells × 2 settings × 3 seeds).",
    )
    parser.add_argument(
        "--cell",
        type=str,
        default=None,
        help="Run only this cell_id (e.g. 'h1-40_h2-100_nwu-1_ne-200').",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel worker processes. Default 1 (sequential). "
            "Recommended : N = number of performance cores. On Mac M4 Max "
            "use 8-12. Each worker forces single-threaded PyTorch so "
            "intra-op × inter-process don't fight."
        ),
    )
    args = parser.parse_args(argv)

    configure_logging(level=args.log_level)

    cells = all_cells()
    settings = SETTINGS_ORDER
    seeds = SEEDS

    if args.probe:
        cells = [
            Cell(40, 0, 1, 200),  # student literal
            Cell(60, 100, 2, 200),  # paper-faithful
        ]
        settings = ["setting-1-baseline", "setting-4-maps-1st"]
        seeds = [42, 43, 44]
        logger.info(
            "PROBE mode : %d cells × %d settings × %d seeds", len(cells), len(settings), len(seeds)
        )

    if args.cell:
        cells = [c for c in cells if c.cell_id == args.cell]
        if not cells:
            logger.error("Unknown cell_id : %s", args.cell)
            return 1

    sweep(cells, settings, seeds, dry_run=args.dry_run, workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
