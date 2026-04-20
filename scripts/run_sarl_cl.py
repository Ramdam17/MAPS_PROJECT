"""SARL+CL (MinAtar DQN with continual learning) experiment driver.

Loads ``config/training/sarl_cl.yaml``, instantiates the MinAtar environment,
and runs one training cell (game × setting × seed) via
:func:`maps.experiments.sarl_cl.training_loop.run_training_cl`.

Compared to ``run_sarl.py``, this driver exposes three extra flags that
activate the CL machinery:

* ``--curriculum`` — enable the 3-term CL loss (task + weight-reg + feature).
  Requires ``--teacher-load-path`` to an earlier-task checkpoint.
* ``--adaptive`` — use :class:`AdaptiveQNetwork` for variable in_channels.
  Needed when the curriculum spans games with different input shapes.
* ``--teacher-load-path`` — path to a previous-task ``checkpoint.pt``.
  Loaded via :func:`load_partial_state_dict` so shape mismatches are
  tolerated (cross-game transfer).

Usage
-----
    # Single-task run (no teacher — equivalent to SARL but through CL nets):
    uv run python scripts/run_sarl_cl.py --game breakout --setting 6

    # Second-stage curriculum run with a teacher from Breakout:
    uv run python scripts/run_sarl_cl.py \\
        --game space_invaders --setting 6 \\
        --curriculum --adaptive \\
        --teacher-load-path outputs/sarl_cl/breakout/setting-6/seed-42/checkpoint.pt

    # Smoke test:
    uv run python scripts/run_sarl_cl.py --num-frames 20000 -o validation.every_episodes=10

Paper settings reminder
-----------------------
    1: vanilla DQN (no cascade, no meta)
    2: cascade on first-order only
    3: meta on, cascade off
    4: meta + cascade on first-order
    5: meta + cascade on second-order
    6: full MAPS (meta + cascade on both)

See ``config/training/sarl_cl.yaml`` and the setting table in
:func:`maps.experiments.sarl_cl.training_loop.setting_to_config_cl`.

Reproduction
------------
Multi-game curriculum reproduction (Sprint 06+): chain several invocations
with ``--teacher-load-path`` pointing at the previous stage's checkpoint.
A single 5M-frame run matches standard SARL wall-time on CPU (hours);
use ``--num-frames 50000`` for a smoke test.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import typer
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maps.experiments.sarl_cl.training_loop import (
    SarlCLTrainingConfig,
    run_training_cl,
    setting_to_config_cl,
)
from maps.utils import configure_logging, get_paths, load_config, set_all_seeds

app = typer.Typer(add_completion=False, help=__doc__)
log = logging.getLogger("maps.run_sarl_cl")


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
    curriculum: bool,
    adaptive: bool,
    teacher_load_path: Path | None,
) -> SarlCLTrainingConfig:
    """Translate the OmegaConf YAML + CLI flags into a :class:`SarlCLTrainingConfig`.

    CLI flags win over YAML for the CL-specific toggles — this keeps the
    curriculum/teacher decision visible at the command line rather than
    buried in a config file. Setting (1-6) is applied LAST so it overrides
    any meta/cascade_* values that slipped into the YAML.
    """
    base = SarlCLTrainingConfig(
        game=game,
        seed=seed,
        meta=False,  # overwritten by setting_to_config_cl
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        curriculum=curriculum if curriculum else bool(cfg.cl.curriculum),
        adaptive_backbone=adaptive if adaptive else bool(cfg.cl.adaptive_backbone),
        max_input_channels=int(cfg.cl.max_input_channels),
        teacher_load_path=teacher_load_path
        if teacher_load_path is not None
        else (Path(cfg.cl.teacher_load_path) if cfg.cl.teacher_load_path else None),
        weight_task=float(cfg.cl.weight_task),
        weight_distillation=float(cfg.cl.weight_distillation),
        weight_feature=float(cfg.cl.weight_feature),
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
    return setting_to_config_cl(setting, base)


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
    curriculum: bool = typer.Option(
        False,
        "--curriculum",
        help="Enable 3-term CL loss (requires --teacher-load-path).",
    ),
    adaptive: bool = typer.Option(
        False,
        "--adaptive",
        help="Use AdaptiveQNetwork backbone for variable in_channels.",
    ),
    teacher_load_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--teacher-load-path",
        help="Previous-task checkpoint.pt (required when --curriculum is set).",
    ),
    output_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        help="Override output directory. Default: $SCRATCH/maps/outputs/sarl_cl/<game>/setting-<N>/seed-<seed>/ (or ./outputs/sarl_cl/... when $SCRATCH unset).",
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

    cfg = load_config("training/sarl_cl", overrides=list(override))
    paths = get_paths()
    paths.ensure_dirs()

    effective_seed = seed if seed is not None else int(cfg.seed)
    out_dir = (
        output_dir
        if output_dir is not None
        else paths.scratch_root
        / "maps"
        / "outputs"
        / "sarl_cl"
        / game
        / f"setting-{setting}"
        / f"seed-{effective_seed}"
    )

    training_cfg = _build_training_config(
        cfg,
        game=game,
        setting=setting,
        seed=effective_seed,
        num_frames=num_frames,
        output_dir=out_dir,
        curriculum=curriculum,
        adaptive=adaptive,
        teacher_load_path=teacher_load_path,
    )

    # Sanity check: curriculum mode without a teacher path is a config error.
    if training_cfg.curriculum and training_cfg.teacher_load_path is None:
        raise typer.BadParameter(
            "curriculum=True requires --teacher-load-path (or cl.teacher_load_path in YAML)."
        )
    if (
        training_cfg.teacher_load_path is not None
        and not Path(training_cfg.teacher_load_path).is_file()
    ):
        raise typer.BadParameter(f"teacher checkpoint not found: {training_cfg.teacher_load_path}")

    log.info(
        "SARL+CL run: game=%s setting=%d seed=%d frames=%d device=%s "
        "curriculum=%s adaptive=%s teacher=%s",
        training_cfg.game,
        setting,
        training_cfg.seed,
        training_cfg.num_frames,
        training_cfg.device,
        training_cfg.curriculum,
        training_cfg.adaptive_backbone,
        training_cfg.teacher_load_path,
    )
    log.info("Effective config:\n%s", OmegaConf.to_yaml(cfg))

    set_all_seeds(effective_seed)
    env = _build_env(game)

    t0 = time.perf_counter()
    _, _, metrics = run_training_cl(env, training_cfg)
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
