"""Blindsight experiment driver — CLI skeleton.

Loads the composed config (config/training/blindsight.yaml ← config/maps.yaml),
seeds every RNG, and prepares the output directory layout.

Training logic is wired in Sprint 04. For now this script validates the
config pipeline end-to-end so downstream sprints can assume a clean
"load → seed → paths" contract.

Usage
-----
    uv run python scripts/run_blindsight.py                      # defaults
    uv run python scripts/run_blindsight.py --seed 43
    uv run python scripts/run_blindsight.py --override train.n_epochs=10
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from omegaconf import OmegaConf

# Enable `python scripts/run_blindsight.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maps.utils import get_paths, load_config, set_all_seeds

app = typer.Typer(add_completion=False, help=__doc__)
log = logging.getLogger("maps.run_blindsight")


@app.command()
def main(
    seed: int | None = typer.Option(None, help="Override RANDOM_SEED from config"),
    override: list[str] = typer.Option(  # noqa: B008  # Typer idiom
        [],
        "--override",
        "-o",
        help="Hydra-style override, e.g. `-o train.n_epochs=10`. Repeatable.",
    ),
    dry_run: bool = typer.Option(
        True,  # NOTE: True until Sprint 04 wires actual training
        help="Load config + print the resolved tree without training.",
    ),
    log_level: str = typer.Option("INFO", help="Python logging level"),
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    cfg = load_config("training/blindsight", overrides=list(override))
    if seed is not None:
        cfg.seed = seed

    paths = get_paths()
    paths.ensure_dirs()

    set_all_seeds(cfg.seed)

    log.info("Blindsight run — seed=%d", cfg.seed)
    log.info("Paths: root=%s  outputs=%s  models=%s", paths.root, paths.outputs, paths.models)
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    if dry_run:
        log.info("Dry run — exiting before training (Sprint 04 will wire this).")
        return

    raise NotImplementedError("Training loop is wired in Sprint 04.")


if __name__ == "__main__":
    app()
