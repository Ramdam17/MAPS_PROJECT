"""MARL (MeltingPot + MAPPO + MAPS) experiment driver.

Loads ``config/training/marl.yaml`` + ``config/env/marl/<substrate>.yaml`` +
``config/experiments/factorial_marl.yaml``, builds the MeltingPot env and the
separated-MAPPO runner, and trains one cell (substrate × setting × seed).

Writes metrics JSON to
``$SCRATCH/maps/outputs/marl/<substrate>/setting-<id>/seed-<seed>/metrics.json``
(falls back to the project-local ``outputs/marl/...`` tree when ``$SCRATCH``
is unset).

Usage
-----
    # .venv-marl (Python 3.11 + meltingpot) is required to build the env.
    # See docs/install_marl_drac.md.
    .venv-marl/bin/python scripts/run_marl.py --substrate commons_harvest_closed --setting maps --seed 42

    # Smoke test (shorten the run):
    .venv-marl/bin/python scripts/run_marl.py --substrate territory_inside_out \
        --setting baseline --seed 42 --num-env-steps 10000

    # Custom config override:
    .venv-marl/bin/python scripts/run_marl.py --setting baseline \
        --substrate chemistry -o training.n_rollout_threads=1

Paper settings reminder (config/experiments/factorial_marl.yaml)
---------------------------------------------------------------
    baseline              : no meta, no cascade (vanilla MAPPO)
    cascade_1st_no_meta   : cascade on 1st-order only
    meta_no_cascade       : meta wager only (no cascade)
    maps                  : meta + cascade 1st-order (paper §B.4 simple MAPS)
    meta_cascade_2nd      : meta + cascade 2nd-order only
    meta_cascade_both     : full (meta + cascade on both orders)

Paper Table 12 defaults are in ``config/training/marl.yaml`` ; per-substrate
overrides (num_agents, episode_length, max_cycles) live in
``config/env/marl/<substrate>.yaml``.

Reproduction
------------
Sprint 09 drives this CLI via :
``scripts/slurm/marl_array.sh`` — 6 settings × 4 substrates × 3 seeds = 72
cells, submitted with ``--gres=gpu:h100:1`` (or ``h200:1``) and ~24 h wall.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import typer
from gymnasium import spaces
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maps.experiments.marl import (  # noqa: E402
    MarlSetting,
    MeltingpotRunner,
    RunnerConfig,
)
from maps.experiments.marl.env import build_env_from_config  # noqa: E402
from maps.utils import configure_logging, get_paths, load_config, set_all_seeds  # noqa: E402

app = typer.Typer(add_completion=False, help=__doc__)
log = logging.getLogger("maps.run_marl")


_SUPPORTED_SUBSTRATES = (
    "commons_harvest_closed",
    "commons_harvest_partnership",
    "chemistry",
    "territory_inside_out",
)
_SUPPORTED_SETTINGS = (
    "baseline",
    "cascade_1st_no_meta",
    "meta_no_cascade",
    "maps",
    "meta_cascade_2nd",
    "meta_cascade_both",
)


def _resolve_setting(factorial_cfg, setting_id: str) -> MarlSetting:
    """Find the named cell in ``config/experiments/factorial_marl.yaml``."""
    for s in factorial_cfg.settings:
        if s.id == setting_id:
            return MarlSetting.from_dict(s)
    valid = [s.id for s in factorial_cfg.settings]
    raise typer.BadParameter(f"Unknown setting {setting_id!r}. Valid: {valid}")


def _build_runner_config(cfg, env, env_cfg, setting: MarlSetting, device: str) -> RunnerConfig:
    """Bundle the loaded configs + live env into a :class:`RunnerConfig`.

    MeltingPotEnv exposes :
    - ``observation_space["player_0"]["RGB"]`` — per-agent RGB (11×11×3 after 8× downsample).
    - ``share_observation_space["player_0"]`` — centralized WORLD.RGB, whose
      spatial dims are substrate-specific (e.g. 24×18×3 for commons_harvest_closed).
      These do NOT equal the per-agent RGB — query the env directly to get
      the real shape.
    - ``action_space["player_0"]`` — Discrete action space (8 for most substrates).
    """
    obs_shape = tuple(env.observation_space["player_0"]["RGB"].shape)
    share_obs_shape = tuple(env.share_observation_space["player_0"].shape)
    action_space = env.action_space["player_0"]

    return RunnerConfig(
        cfg=cfg,
        setting=setting,
        num_agents=int(env_cfg.num_agents),
        obs_shape=obs_shape,
        share_obs_shape=share_obs_shape,
        action_space=action_space,
        device=device,
    )


def _write_metrics(out_dir: Path, infos: list[dict], meta: dict) -> Path:
    """Persist per-episode train infos + run metadata to ``metrics.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "episodes": infos,
    }
    target = out_dir / "metrics.json"
    with target.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return target


@app.command()
def main(
    substrate: str = typer.Option(
        "commons_harvest_closed",
        help=f"MeltingPot substrate id. One of {_SUPPORTED_SUBSTRATES}.",
    ),
    setting: str = typer.Option(
        "maps",
        help=f"Factorial setting id. One of {_SUPPORTED_SETTINGS}.",
    ),
    seed: int | None = typer.Option(
        None,
        help="Seed override. Falls back to config/training/marl.yaml ``seed`` field.",
    ),
    num_env_steps: int | None = typer.Option(
        None,
        "--num-env-steps",
        help="Override training.num_env_steps. Useful for smoke tests (e.g. 10_000).",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help=(
            "Override output directory. Default : "
            "$SCRATCH/maps/outputs/marl/<substrate>/setting-<id>/seed-<N>/"
            " (or ./outputs/marl/... when $SCRATCH unset)."
        ),
    ),
    device: str = typer.Option(
        "cuda",
        help="Torch device. DRAC compute nodes default to 'cuda' ; 'cpu' for dev.",
    ),
    override: list[str] = typer.Option(  # noqa: B008
        [],
        "--override",
        "-o",
        help="OmegaConf override, e.g. ``-o ppo.ppo_epoch=5``. Repeatable.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help=(
            "Auto-detect ``<output_dir>/checkpoint.pt`` and resume from it. "
            "Silent no-op (with a warning) if the file does not exist — safe "
            "to pass unconditionally for idempotent re-submissions."
        ),
    ),
    resume_from: Path | None = typer.Option(  # noqa: B008
        None,
        "--resume-from",
        help=(
            "Explicit path to a checkpoint file. Raises if the file does not "
            "exist — explicit resume requests should never silently restart."
        ),
    ),
    log_level: str = typer.Option("INFO", help="Python logging level"),
) -> None:
    configure_logging(level=log_level)

    # ── Resolve + load configs ────────────────────────────────────────────
    if substrate not in _SUPPORTED_SUBSTRATES:
        raise typer.BadParameter(
            f"Unknown substrate {substrate!r}. Valid: {_SUPPORTED_SUBSTRATES}"
        )
    if setting not in _SUPPORTED_SETTINGS:
        raise typer.BadParameter(
            f"Unknown setting {setting!r}. Valid: {_SUPPORTED_SETTINGS}"
        )

    cfg = load_config("training/marl", overrides=list(override))
    env_cfg = load_config(f"env/marl/{substrate}")
    factorial_cfg = load_config("experiments/factorial_marl")
    paths = get_paths()
    paths.ensure_dirs()

    effective_seed = seed if seed is not None else int(cfg.seed)
    setting_obj = _resolve_setting(factorial_cfg, setting)

    # ── Wire env-config knobs into the training config ───────────────────
    # Per-substrate episode_length takes precedence over the training default.
    # Expose the runner's ``num_env_steps`` override from the CLI flag.
    cfg.training.episode_length = int(env_cfg.episode_length)
    if num_env_steps is not None:
        cfg.training.num_env_steps = int(num_env_steps)

    out_dir = (
        output_dir
        if output_dir is not None
        else paths.scratch_root
        / "maps"
        / "outputs"
        / "marl"
        / substrate
        / f"setting-{setting}"
        / f"seed-{effective_seed}"
    )

    # ── Resume resolution (E.17a) ─────────────────────────────────────────
    # --resume-from wins over --resume. Both land at a Path that's either
    # None (fresh run) or an existing checkpoint file.
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
                "--resume requested but no checkpoint at %s — starting fresh",
                candidate,
            )

    # Periodic checkpoint path — enabled by default inside the run output dir.
    # One file per cell ; overwritten in place every save_interval episodes.
    checkpoint_path = out_dir / "checkpoint.pt"

    log.info(
        "MARL run : substrate=%s setting=%s seed=%d device=%s num_env_steps=%d",
        substrate,
        setting,
        effective_seed,
        device,
        int(cfg.training.num_env_steps),
    )
    log.info("Effective training config :\n%s", OmegaConf.to_yaml(cfg))

    set_all_seeds(effective_seed)

    # ── Build env ─────────────────────────────────────────────────────────
    env = build_env_from_config(env_cfg)

    runner_cfg = _build_runner_config(cfg, env, env_cfg, setting_obj, device=device)
    runner = MeltingpotRunner(runner_cfg, env)

    # ── Train ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        infos = runner.run(
            checkpoint_path=checkpoint_path,
            resume_from=resolved_resume,
        )
    finally:
        env.close()
    elapsed = time.perf_counter() - t0

    meta = {
        "substrate": substrate,
        "setting": asdict(setting_obj),
        "seed": effective_seed,
        "num_env_steps": int(cfg.training.num_env_steps),
        "episode_length": int(cfg.training.episode_length),
        "n_rollout_threads": int(cfg.training.n_rollout_threads),
        "num_agents": int(env_cfg.num_agents),
        "elapsed_s": elapsed,
    }
    metrics_path = _write_metrics(out_dir, infos, meta)
    log.info(
        "done : episodes=%d elapsed=%.1fs metrics=%s",
        len(infos),
        elapsed,
        metrics_path,
    )


if __name__ == "__main__":
    app()
