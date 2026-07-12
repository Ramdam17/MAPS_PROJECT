"""AGL CLI — Typer entry point for the 4-phase protocol (paper §A.2).

Pipeline per ``(setting, seed)``:

1. **pre-train** on random grammar (then reset the 1st-order — D-agl-reset);
2. **replicate** the post-pretrain nets into a ``num_networks``-cell pool;
3. **train** High tier (``[0:n//2]``, 12 epochs) and Low tier (``[n//2:]``,
   3 epochs) on Grammar-A;
4. **evaluate** the pool on Grammar-A + Grammar-B, aggregated by tier.

Usage
-----

.. code-block:: bash

    # Single run, default Full-MAPS, seed 42, paper defaults
    uv run python -m maps.domains.agl.cli --seed 42

    # All 6 settings × seeds
    uv run python -m maps.domains.agl.cli --all-settings --seeds 42,43,44

    # Faithful-source ablation overrides (Sprint-08 D.28)
    uv run python -m maps.domains.agl.cli -o train.n_epochs_pretrain=60

Outputs land in ``$SCRATCH/maps/outputs/agl/<setting>/seed-<N>/`` (HPC) or
``./outputs/agl/<setting>/seed-<N>/`` (dev).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer
from omegaconf import DictConfig, OmegaConf

from maps.domains.agl.pool import AGLNetworkPool
from maps.domains.agl.trainer import SETTINGS_REGISTRY, AGLTrainer
from maps.utils.config import load_config, parse_overrides
from maps.utils.device import get_device
from maps.utils.logging_setup import configure_logging
from maps.utils.paths import get_paths
from maps.utils.seeding import set_all_seeds

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="agl",
    help="AGL Sprint 13 CLI — 4-phase protocol, one or all settings × seeds.",
    no_args_is_help=False,
)


def _load_cfg(overrides: list[str]) -> DictConfig:
    """Load the AGL training config (maps defaults composed), apply overrides."""
    cfg = load_config("domains/agl/training")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg  # type: ignore[return-value]


def _parse_seeds(seeds_arg: str | None) -> list[int] | None:
    if not seeds_arg:
        return None
    return [int(s.strip()) for s in seeds_arg.split(",") if s.strip()]


def _resolve_output_dir(setting_id: str, seed: int, override_dir: Path | None) -> Path:
    if override_dir is not None:
        return override_dir / setting_id / f"seed-{seed}"
    paths = get_paths()
    return paths.scratch_root / "maps" / "outputs" / "agl" / setting_id / f"seed-{seed}"


def _run_one(
    setting_id: str,
    seed: int,
    cfg: DictConfig,
    device: torch.device,
    output_dir: Path | None,
) -> dict:
    """Run the full 4-phase AGL pipeline for one (setting, seed). Returns summary."""
    if setting_id not in SETTINGS_REGISTRY:
        raise typer.BadParameter(
            f"unknown setting {setting_id!r}; expected one of {sorted(SETTINGS_REGISTRY)}"
        )
    setting = SETTINGS_REGISTRY[setting_id]
    set_all_seeds(seed)

    out_dir = _resolve_output_dir(setting_id, seed, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("▶ AGL setting=%s seed=%d → %s", setting_id, seed, out_dir)

    num_networks = int(cfg.train.num_networks)
    half = num_networks // 2
    n_high = int(cfg.train.n_epochs_training_high)
    n_low = int(cfg.train.n_epochs_training_low)

    trainer = AGLTrainer(setting=setting, seed=seed, cfg=cfg, device=device)
    trainer.build()

    # Phase 1 — pre-train (+ reset).
    t0 = time.perf_counter()
    pre_l1, pre_l2, _pre_prec = trainer.pre_train()
    pretrain_seconds = time.perf_counter() - t0
    np.save(out_dir / "pretrain_losses_1.npy", pre_l1)
    np.save(out_dir / "pretrain_losses_2.npy", pre_l2)
    assert trainer.first_order is not None and trainer.second_order is not None
    torch.save(trainer.first_order.state_dict(), out_dir / "first_order_reset.pt")
    torch.save(trainer.second_order.state_dict(), out_dir / "second_order_postpre.pt")

    # Phase 2 — replicate.
    pool = AGLNetworkPool(trainer, num_networks=num_networks)

    # Phase 3 — train High / Low tiers.
    t0 = time.perf_counter()
    high = pool.train_range(0, half, n_high)
    high_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    low = pool.train_range(half, num_networks, n_low)
    low_seconds = time.perf_counter() - t0
    for tier_name, tier in (("high", high), ("low", low)):
        np.save(out_dir / f"training_{tier_name}_losses_1.npy", tier["losses_1"])
        np.save(out_dir / f"training_{tier_name}_losses_2.npy", tier["losses_2"])
        np.save(out_dir / f"training_{tier_name}_precision.npy", tier["precision"])

    # Phase 4 — evaluate.
    t0 = time.perf_counter()
    evaluation = pool.evaluate()
    eval_seconds = time.perf_counter() - t0

    summary = {
        "setting": setting_id,
        "seed": seed,
        "num_networks": num_networks,
        "pretrain": {
            "n_epochs": int(cfg.train.n_epochs_pretrain),
            "loss_1_final": float(pre_l1[-1]),
            "loss_2_final": float(pre_l2[-1]),
            "elapsed_seconds": pretrain_seconds,
        },
        "training_high": {
            "n_epochs": n_high,
            "n_cells": half,
            "precision_final_mean": float(np.mean(high["precision"][:, -1])),
            "loss_1_final_mean": float(np.mean(high["losses_1"][:, -1])),
            "elapsed_seconds": high_seconds,
        },
        "training_low": {
            "n_epochs": n_low,
            "n_cells": num_networks - half,
            "precision_final_mean": float(np.mean(low["precision"][:, -1])),
            "loss_1_final_mean": float(np.mean(low["losses_1"][:, -1])),
            "elapsed_seconds": low_seconds,
        },
        "evaluation": {**evaluation, "elapsed_seconds": eval_seconds},
        "meta_frozen_in_training": bool(cfg.train.train_meta_frozen_in_training),
    }
    with (out_dir / "summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    logger.info(
        "✓ Done setting=%s seed=%d — high wager acc=%.3f | low wager acc=%.3f",
        setting_id,
        seed,
        evaluation["high"]["wager_accuracy"],
        evaluation["low"]["wager_accuracy"],
    )
    return summary


@app.command()
def main(
    setting: Annotated[
        str, typer.Option(help="Setting slug from SETTINGS_REGISTRY.")
    ] = "setting-6-full-maps",
    seed: Annotated[int, typer.Option(help="Single-run seed.")] = 42,
    seeds: Annotated[
        str | None, typer.Option(help="Comma-separated seed list (e.g. '42,43,44').")
    ] = None,
    all_settings: Annotated[
        bool, typer.Option("--all-settings", help="Loop over all 6 settings × seeds.")
    ] = False,
    overrides: Annotated[
        list[str] | None,
        typer.Option("-o", "--override", help="Hydra-style override 'key=value' (repeatable)."),
    ] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Override base output directory.")] = None,
    device: Annotated[str, typer.Option(help="'auto' (default), 'cpu', 'mps', 'cuda'.")] = "auto",
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Run AGL Sprint 13 — single setting or all settings × seeds."""
    configure_logging(level=log_level)
    parsed_overrides = parse_overrides(*(overrides or []))
    cfg = _load_cfg(parsed_overrides)
    torch_device = get_device(device)  # type: ignore[arg-type]
    seed_pool = _parse_seeds(seeds) or [seed]

    setting_ids = list(SETTINGS_REGISTRY.keys()) if all_settings else [setting]
    if not all_settings and not seeds:
        seed_pool = [seed]

    logger.info(
        "AGL CLI : %d setting(s) × %d seed(s) = %d run(s)",
        len(setting_ids),
        len(seed_pool),
        len(setting_ids) * len(seed_pool),
    )
    for s_id in setting_ids:
        for sd in seed_pool:
            _run_one(s_id, sd, cfg, torch_device, output_dir)
    logger.info("✓ All run(s) complete.")


if __name__ == "__main__":
    app()
