"""SARL (MinAtar DQN with MAPS components) experiment driver.

Loads ``config/training/sarl.yaml``, instantiates the MinAtar environment,
and runs one training cell (game × setting × seed) via
:func:`maps.experiments.sarl.training_loop.run_training`. Writes metrics JSON
and final network weights under ``$SCRATCH/maps/outputs/sarl/<game>/setting-<N>/seed-<seed>/``
(falls back to the project-local ``outputs/sarl/...`` tree when ``$SCRATCH`` is unset — dev boxes).

Usage
-----
    uv run python scripts/run_sarl.py                                       # defaults
    uv run python scripts/run_sarl.py --game breakout --setting 4
    uv run python scripts/run_sarl.py --game seaquest --seed 43 --num-frames 500000
    uv run python scripts/run_sarl.py --setting 6 -o training.batch_size=64
    uv run python scripts/run_sarl.py --resume                              # auto-detect ckpt in output_dir
    uv run python scripts/run_sarl.py --resume-from /path/to/checkpoint.pt  # explicit path

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
    resume_from: Path | None = None,
) -> SarlTrainingConfig:
    """Translate the OmegaConf YAML into a ``SarlTrainingConfig`` dataclass.

    Setting (1-6) is applied LAST so it overrides any meta/cascade_* values
    that slipped into the YAML by mistake — this keeps the CLI-visible
    setting the source of truth for branch selection.
    """
    # D.7/D.9 additions: plumb paper-faithful gamma + Adam betas from yaml.
    # Both have sensible dataclass defaults (0.999 + (0.95, 0.95)) but CLI
    # overrides via `-o training.gamma=...` / `-o optimizer.betas=[...]`
    # must reach the dataclass — read them defensively with getattr so the
    # script keeps working on legacy yamls that predate these fields.
    gamma_val = float(getattr(cfg.training, "gamma", 0.999))
    betas_cfg = getattr(cfg.optimizer, "betas", (0.95, 0.95))
    betas_val = (float(betas_cfg[0]), float(betas_cfg[1]))
    # Sprint-08 D.22b: pick up the first-order-loss toggle from yaml if
    # present. Legacy yamls (pre-D.22b) default to 'cae'.
    fo_loss_kind = str(
        getattr(getattr(cfg, "first_order_loss", {}), "kind", "cae")
        if hasattr(cfg, "first_order_loss")
        else "cae"
    )

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
        adam_betas=betas_val,
        scheduler_period=int(cfg.scheduler.step_size),
        scheduler_gamma=float(cfg.scheduler.gamma),
        gamma=gamma_val,
        alpha=float(cfg.alpha),
        first_order_loss_kind=fo_loss_kind,
        validation_every_episodes=int(cfg.validation.every_episodes),
        validation_iterations=int(cfg.validation.n_episodes),
        device=str(cfg.device),
        output_dir=output_dir,
        resume_from=resume_from,
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
        help="Override output directory. Default: $SCRATCH/maps/outputs/sarl/<game>/setting-<N>/seed-<seed>/ (or ./outputs/sarl/... when $SCRATCH unset).",
    ),
    override: list[str] = typer.Option(  # noqa: B008
        [],
        "--override",
        "-o",
        help="OmegaConf override, e.g. `-o training.batch_size=64`. Repeatable.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help=(
            "Auto-detect a checkpoint at `<output_dir>/checkpoint.pt` and resume "
            "from it. If the file is missing we log a warning and start fresh. "
            "Mutually coexists with --resume-from (explicit path wins)."
        ),
    ),
    resume_from: Path | None = typer.Option(  # noqa: B008
        None,
        "--resume-from",
        help=(
            "Explicit path to a checkpoint file. If set, overrides --resume "
            "auto-detection. The path must exist — we raise if not, because "
            "an explicit request should not silently fall back to a fresh run."
        ),
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
        else paths.scratch_root
        / "maps"
        / "outputs"
        / "sarl"
        / game
        / f"setting-{setting}"
        / f"seed-{effective_seed}"
    )

    # Resolve resume path (Sprint-08 D.14). Explicit --resume-from wins; then
    # --resume auto-detect; then None (fresh start).
    resolved_resume: Path | None = None
    if resume_from is not None:
        if not resume_from.is_file():
            raise typer.BadParameter(
                f"--resume-from path does not exist: {resume_from}",
                param_hint="--resume-from",
            )
        resolved_resume = resume_from
        log.info("resuming from explicit --resume-from=%s", resolved_resume)
    elif resume:
        candidate = out_dir / "checkpoint.pt"
        if candidate.is_file():
            resolved_resume = candidate
            log.info("--resume auto-detected checkpoint: %s", resolved_resume)
        else:
            log.warning(
                "--resume requested but no checkpoint at %s; starting fresh", candidate
            )

    training_cfg = _build_training_config(
        cfg,
        game=game,
        setting=setting,
        seed=effective_seed,
        num_frames=num_frames,
        output_dir=out_dir,
        resume_from=resolved_resume,
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
