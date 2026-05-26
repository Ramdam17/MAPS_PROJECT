"""Blindsight CLI — Typer entry point for single-run or all-settings.

Usage
-----

.. code-block:: bash

    # Single run, default Setting 6 Full MAPS, seed 42, paper defaults
    uv run python -m maps.domains.blindsight.cli --seed 42

    # All settings × multiple seeds
    uv run python -m maps.domains.blindsight.cli --all-settings --seeds 42,43,44

    # Override hyperparams Hydra-style
    uv run python -m maps.domains.blindsight.cli -o train.n_epochs=10 \
        -o first_order.hidden_dim=60

    # SimCLR loss instead of CAE (D11.7 + D12.4)
    uv run python -m maps.domains.blindsight.cli -o first_order_loss.kind=simclr

Outputs land in ``$SCRATCH/maps/outputs/blindsight/<setting>/seed-<N>/``
(HPC) or ``./outputs/blindsight/<setting>/seed-<N>/`` (dev).

Per-run artifacts :

- ``losses_1.npy`` (CAE / SimCLR loss curve)
- ``losses_2.npy`` (BCE wager loss curve)
- ``first_order.pt`` (final state dict)
- ``second_order.pt`` (final state dict)
- ``summary.json`` (setting, seed, n_epochs, final losses, eval metrics)
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

from maps.domains.blindsight.trainer import (
    SETTINGS_REGISTRY,
    BlindsightTrainer,
    EvalMetrics,
    TrainingMetrics,
)
from maps.utils.config import load_config, parse_overrides
from maps.utils.device import get_device
from maps.utils.logging_setup import configure_logging
from maps.utils.paths import get_paths
from maps.utils.seeding import set_all_seeds

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="blindsight",
    help="Blindsight Sprint 12 CLI — run one or all settings × seeds.",
    no_args_is_help=False,
)


def _load_merged_cfg(overrides: list[str]) -> DictConfig:
    """Load and merge training + env YAML, apply CLI overrides, resolve."""
    training_cfg = load_config("domains/blindsight/training")
    env_cfg = load_config("domains/blindsight/env", resolve=False)
    merged = OmegaConf.merge(training_cfg, env_cfg)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(merged)
    return merged  # type: ignore[return-value]


def _parse_seeds(seeds_arg: str | None) -> list[int] | None:
    """Parse '--seeds 42,43,44' → [42, 43, 44]. Returns None if empty."""
    if not seeds_arg:
        return None
    return [int(s.strip()) for s in seeds_arg.split(",") if s.strip()]


def _resolve_output_dir(
    setting_id: str,
    seed: int,
    override_dir: Path | None,
) -> Path:
    """Output layout : $SCRATCH/maps/outputs/blindsight/<setting>/seed-<N>/."""
    if override_dir is not None:
        return override_dir / setting_id / f"seed-{seed}"
    paths = get_paths()
    return paths.scratch_root / "maps" / "outputs" / "blindsight" / setting_id / f"seed-{seed}"


def _save_artifacts(
    out_dir: Path,
    setting_id: str,
    seed: int,
    cfg: DictConfig,
    trainer: BlindsightTrainer,
    train_metrics: TrainingMetrics,
    eval_metrics: EvalMetrics,
    wall_seconds: float,
) -> None:
    """Persist run artefacts : losses .npy, state_dicts .pt, summary.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "losses_1.npy", train_metrics.losses_1)
    np.save(out_dir / "losses_2.npy", train_metrics.losses_2)

    assert trainer.first_order is not None
    assert trainer.second_order is not None
    torch.save(trainer.first_order.state_dict(), out_dir / "first_order.pt")
    torch.save(trainer.second_order.state_dict(), out_dir / "second_order.pt")

    summary = {
        "setting": setting_id,
        "seed": seed,
        "n_epochs": int(cfg.train.n_epochs),
        "batch_size": int(cfg.train.batch_size),
        "first_order_loss": cfg.first_order_loss.kind,
        "final_loss_1": float(train_metrics.losses_1[-1]),
        "final_loss_2": float(train_metrics.losses_2[-1]),
        "discrimination_accuracy": eval_metrics.discrimination_accuracy,
        "wager_accuracy": eval_metrics.wager_accuracy,
        "wall_seconds": wall_seconds,
    }
    with (out_dir / "summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)


def _run_one(
    setting_id: str,
    seed: int,
    cfg: DictConfig,
    device: torch.device,
    output_dir: Path | None,
) -> dict:
    """Run one (setting, seed) combo. Returns the saved summary dict."""
    if setting_id not in SETTINGS_REGISTRY:
        raise typer.BadParameter(
            f"unknown setting {setting_id!r}; expected one of {sorted(SETTINGS_REGISTRY)}"
        )

    setting = SETTINGS_REGISTRY[setting_id]
    set_all_seeds(seed)

    out_dir = _resolve_output_dir(setting_id, seed, output_dir)
    logger.info(
        "▶ Running setting=%s seed=%d → %s",
        setting_id,
        seed,
        out_dir,
    )

    trainer = BlindsightTrainer(setting=setting, seed=seed, cfg=cfg, device=device)
    trainer.build()

    t0 = time.perf_counter()
    train_metrics = trainer.train()
    eval_metrics = trainer.evaluate()
    wall = time.perf_counter() - t0

    _save_artifacts(out_dir, setting_id, seed, cfg, trainer, train_metrics, eval_metrics, wall)
    logger.info(
        "✓ Done setting=%s seed=%d in %.1fs — discrim=%.3f wager=%.3f",
        setting_id,
        seed,
        wall,
        eval_metrics.discrimination_accuracy.get("superthreshold", 0.0),
        eval_metrics.wager_accuracy.get("superthreshold", 0.0),
    )
    return {
        "setting": setting_id,
        "seed": seed,
        "out_dir": str(out_dir),
        "discrimination_accuracy": eval_metrics.discrimination_accuracy,
        "wager_accuracy": eval_metrics.wager_accuracy,
        "wall_seconds": wall,
    }


@app.command()
def main(
    setting: Annotated[
        str,
        typer.Option(help="Setting slug from SETTINGS_REGISTRY."),
    ] = "setting-6-full-maps",
    seed: Annotated[
        int,
        typer.Option(help="Single-run seed (ignored when --all-settings + --seeds is set)."),
    ] = 42,
    seeds: Annotated[
        str | None,
        typer.Option(help="Comma-separated seed list for --all-settings (e.g. '42,43,44')."),
    ] = None,
    all_settings: Annotated[
        bool,
        typer.Option("--all-settings", help="Loop over all 6 settings × seeds."),
    ] = False,
    overrides: Annotated[
        list[str] | None,
        typer.Option(
            "-o",
            "--override",
            help="Hydra-style override 'key=value' (repeatable).",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Override base output directory."),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="'auto' (default), 'cpu', 'mps', 'cuda'."),
    ] = "auto",
    log_level: Annotated[
        str,
        typer.Option(help="Logging level: DEBUG / INFO / WARNING / ERROR."),
    ] = "INFO",
) -> None:
    """Run Blindsight Sprint 12 — single setting or all settings."""
    configure_logging(level=log_level)
    parsed_overrides = parse_overrides(*(overrides or []))
    cfg = _load_merged_cfg(parsed_overrides)
    torch_device = get_device(device)  # type: ignore[arg-type]
    seed_pool = _parse_seeds(seeds) or [seed]

    if all_settings:
        setting_ids = list(SETTINGS_REGISTRY.keys())
    else:
        setting_ids = [setting]
        # single-run path : honour the --seed arg, ignore --seeds if not multi
        seed_pool = [seed] if not seeds else seed_pool

    n_runs = len(setting_ids) * len(seed_pool)
    logger.info(
        "Blindsight CLI : %d settings × %d seeds = %d runs",
        len(setting_ids),
        len(seed_pool),
        n_runs,
    )

    summaries: list[dict] = []
    for s_id in setting_ids:
        for sd in seed_pool:
            summary = _run_one(s_id, sd, cfg, torch_device, output_dir)
            summaries.append(summary)

    logger.info("✓ All %d run(s) complete.", len(summaries))


if __name__ == "__main__":
    app()
