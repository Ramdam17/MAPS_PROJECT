"""SARL (MinAtar DQN with MAPS components) experiment driver.

Loads ``config/training/sarl.yaml``, instantiates the MinAtar environment,
and runs one training cell (game × setting × seed) via
:func:`maps.experiments.sarl.training_loop.run_training`. Writes metrics JSON
and final network weights under ``outputs/sarl/<game>/setting-<N>/seed-<seed>/``.

Usage
-----
    uv run python scripts/run_sarl.py                                       # defaults
    uv run python scripts/run_sarl.py --game breakout --setting 4
    uv run python scripts/run_sarl.py --game seaquest --seed 43 --num-frames 500000
    uv run python scripts/run_sarl.py --setting 6 -o training.batch_size=64

Paper settings reminder
-----------------------
    1: vanilla DQN (no cascade, no meta)
    2: cascade on first-order only
    3: meta on, cascade off
    4: meta + cascade on first-order
    5: meta + cascade on second-order
    6: full MAPS (meta + cascade on both)

See ``config/training/sarl.yaml`` and the setting table in
``maps.experiments.sarl.training_loop.setting_to_config``.

Reproduction
------------
Sprint 07 drives this CLI via a SLURM array job on Narval (30 seeds × 6 settings
× 5 games = 900 runs). On a Mac M-series CPU a single 5M-frame run takes
several hours; use ``--num-frames 50000`` for a smoke test.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import typer
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maps.experiments.sarl.training_loop import (
    SarlTrainingConfig,
    run_training,
    setting_to_config,
)
from maps.utils import configure_logging, get_paths, load_config, set_all_seeds

app = typer.Typer(add_completion=False, help=__doc__)
log = logging.getLogger("maps.run_sarl")


_SUPPORTED_GAMES = {"space_invaders", "breakout", "seaquest", "asterix", "freeway"}


def _build_env(game: str) -> object:
    """Instantiate a MinAtar environment.

    Imported locally so the script module doesn't error-import without the
    optional ``sarl`` extra installed.
    """
    try:
        from minatar import Environment
    except ImportError as exc:  # pragma: no cover — environment setup
        raise RuntimeError("MinAtar is not installed. Run: uv sync --extra sarl") from exc

    if game not in _SUPPORTED_GAMES:
        valid = sorted(_SUPPORTED_GAMES)
        raise typer.BadParameter(f"Unknown game {game!r}. Valid: {valid}")

    return Environment(game)


def _build_training_config(
    cfg,
    game: str,
    setting: int,
    seed: int,
    num_frames: int | None,
    output_dir: Path,
) -> SarlTrainingConfig:
    """Translate the OmegaConf YAML into a ``SarlTrainingConfig`` dataclass.

    Setting (1-6) is applied LAST so it overrides any meta/cascade_* values
    that slipped into the YAML by mistake — this keeps the CLI-visible
    setting the source of truth for branch selection.
    """
    base = SarlTrainingConfig(
        game=game,
        seed=seed,
        meta=False,  # overwritten by setting_to_config
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        num_frames=num_frames if num_frames is not None else int(cfg.training.num_frames),
        batch_size=int(cfg.training.batch_size),
        replay_buffer_size=int(cfg.training.replay_buffer_size),
        replay_start_size=int(cfg.training.replay_start_size),
        training_freq=int(cfg.training.training_freq),
        target_update_freq=int(cfg.training.target_update_freq),
        step_size_1=float(cfg.optimizer.lr_first_order),
        step_size_2=float(cfg.optimizer.lr_second_order),
        scheduler_period=int(cfg.scheduler.step_size),
        scheduler_gamma=float(cfg.scheduler.gamma),
        alpha=float(cfg.alpha),
        validation_every_episodes=int(cfg.validation.every_episodes),
        validation_iterations=int(cfg.validation.n_episodes),
        device=str(cfg.device),
        output_dir=output_dir,
    )
    return setting_to_config(setting, base)


@app.command()
def main(
    game: str = typer.Option("space_invaders", help="MinAtar game name."),
    setting: int = typer.Option(
        6,
        help="Paper setting 1-6 (see module docstring).",
        min=1,
        max=6,
    ),
    seed: int | None = typer.Option(None, help="Override seed (falls back to config default 42)."),
    num_frames: int | None = typer.Option(
        None,
        "--num-frames",
        help="Override training length. Useful for smoke tests (e.g. 50000).",
    ),
    output_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        help="Override output directory. Default: outputs/sarl/<game>/setting-<N>/seed-<seed>/",
    ),
    override: list[str] = typer.Option(  # noqa: B008
        [],
        "--override",
        "-o",
        help="OmegaConf override, e.g. `-o training.batch_size=64`. Repeatable.",
    ),
    log_level: str = typer.Option("INFO", help="Python logging level"),
) -> None:
    configure_logging(level=log_level)

    cfg = load_config("training/sarl", overrides=list(override))
    paths = get_paths()
    paths.ensure_dirs()

    effective_seed = seed if seed is not None else int(cfg.seed)
    out_dir = (
        output_dir
        if output_dir is not None
        else paths.outputs / "sarl" / game / f"setting-{setting}" / f"seed-{effective_seed}"
    )

    training_cfg = _build_training_config(
        cfg,
        game=game,
        setting=setting,
        seed=effective_seed,
        num_frames=num_frames,
        output_dir=out_dir,
    )

    log.info(
        "SARL run: game=%s setting=%d seed=%d frames=%d device=%s",
        training_cfg.game,
        setting,
        training_cfg.seed,
        training_cfg.num_frames,
        training_cfg.device,
    )
    log.info("Effective config:\n%s", OmegaConf.to_yaml(cfg))

    set_all_seeds(effective_seed)
    env = _build_env(game)

    t0 = time.perf_counter()
    _, _, metrics = run_training(env, training_cfg)
    elapsed = time.perf_counter() - t0

    log.info(
        "done: total_frames=%d total_updates=%d elapsed=%.1fs mean_last_G=%.2f",
        metrics.total_frames,
        metrics.total_updates,
        elapsed,
        sum(metrics.episode_returns[-10:]) / max(1, min(10, len(metrics.episode_returns))),
    )


if __name__ == "__main__":
    app()
