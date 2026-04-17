"""AGL experiment driver — CLI skeleton.

See scripts/run_blindsight.py for the full docstring; this is the AGL twin.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maps.utils import get_paths, load_config, set_all_seeds

app = typer.Typer(add_completion=False, help=__doc__)
log = logging.getLogger("maps.run_agl")


@app.command()
def main(
    seed: int | None = typer.Option(None, help="Override RANDOM_SEED from config"),
    override: list[str] = typer.Option(  # noqa: B008
        [],
        "--override",
        "-o",
        help="Hydra-style override, e.g. `-o train.n_epochs_pre=10`. Repeatable.",
    ),
    dry_run: bool = typer.Option(
        True,
        help="Load config + print the resolved tree without training.",
    ),
    log_level: str = typer.Option("INFO", help="Python logging level"),
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    cfg = load_config("training/agl", overrides=list(override))
    if seed is not None:
        cfg.seed = seed

    paths = get_paths()
    paths.ensure_dirs()

    set_all_seeds(cfg.seed)

    log.info("AGL run — seed=%d", cfg.seed)
    log.info("Paths: root=%s  outputs=%s  models=%s", paths.root, paths.outputs, paths.models)
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    if dry_run:
        log.info("Dry run — exiting before training (Sprint 04 will wire this).")
        return

    raise NotImplementedError("Training loop is wired in Sprint 04.")


if __name__ == "__main__":
    app()
