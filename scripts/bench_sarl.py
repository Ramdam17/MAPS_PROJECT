"""SARL benchmark driver — 1 cell, writes a machine-readable bench-<mode>.json.

Used by ``scripts/slurm/bench_sarl.sh`` (Sprint 07 Phase 2) to calibrate
SLURM ``--time`` / ``--mem`` and to decide CPU-vs-GPU for the 300-cell array.

Captures, for a single (game, setting, seed) cell of ``--num-frames`` frames:

* ``frames_per_s``      — wall-clock frames/s
* ``updates_per_s``     — wall-clock optimizer updates/s
* ``wall_s``            — total wall-clock seconds of ``run_training``
* ``peak_vram_mb``      — ``torch.cuda.max_memory_allocated`` in MB (0 on CPU)
* ``peak_rss_mb``       — ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` in MB
* ``total_frames``      — echoed from the metrics (sanity)
* ``total_updates``     — idem
* ``rev``               — ``git rev-parse --short HEAD`` (or ``"dirty"``)
* ``seed``, ``game``, ``setting``, ``device``, ``mode`` — run identifiers

Written to ``<output_dir>/bench-<mode>-<rev>.json``.

Usage::

    uv run --offline python scripts/bench_sarl.py \\
        --mode gpu_full --game breakout --setting 6 --seed 42 \\
        --num-frames 500000 --device cuda

``--mode`` is a label (``cpu_4c``, ``gpu_full``, etc.) that ends up in the
filename; it does not change what the script does.
"""

from __future__ import annotations

import json
import logging
import resource
import subprocess
import sys
import time
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maps.experiments.sarl.training_loop import (
    SarlTrainingConfig,
    run_training,
    setting_to_config,
)
from maps.utils import configure_logging, get_paths, load_config, set_all_seeds

app = typer.Typer(add_completion=False, help=__doc__)
log = logging.getLogger("maps.bench_sarl")


def _git_rev(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@app.command()
def main(
    mode: str = typer.Option(
        ..., help="Bench mode label (cpu_4c, gpu_full, ...). Filename suffix only."
    ),
    game: str = typer.Option("breakout"),
    setting: int = typer.Option(6, min=1, max=6),
    seed: int = typer.Option(42),
    num_frames: int = typer.Option(500000, "--num-frames"),
    device: str = typer.Option("cpu", help="cpu | cuda"),
    output_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        help="Where to write bench-<mode>-<rev>.json. Default: paths.outputs / 'bench'",
    ),
) -> None:
    configure_logging(level="INFO")

    paths = get_paths()
    paths.ensure_dirs()
    out_dir = output_dir if output_dir is not None else (paths.outputs / "bench")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config("training/sarl")
    base = SarlTrainingConfig(
        game=game,
        seed=seed,
        meta=False,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        num_frames=num_frames,
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
        device=device,
        output_dir=out_dir / f"_run-{mode}-{seed}",  # throwaway per-bench
    )
    training_cfg = setting_to_config(setting, base)

    set_all_seeds(seed)

    from minatar import Environment

    env = Environment(game)

    # Reset CUDA peak counter iff on GPU.
    peak_vram_mb = 0.0
    if device == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    _, _, metrics = run_training(env, training_cfg)
    wall_s = time.perf_counter() - t0

    if device == "cuda":
        import torch

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

    # ru_maxrss is in KB on Linux (per getrusage(2)).
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    repo_root = Path(__file__).resolve().parents[1]
    rev = _git_rev(repo_root)

    report = {
        "mode": mode,
        "rev": rev,
        "game": game,
        "setting": setting,
        "seed": seed,
        "device": device,
        "num_frames_requested": num_frames,
        "total_frames": metrics.total_frames,
        "total_updates": metrics.total_updates,
        "wall_s": round(wall_s, 3),
        "frames_per_s": round(metrics.total_frames / wall_s, 2) if wall_s > 0 else 0.0,
        "updates_per_s": round(metrics.total_updates / wall_s, 2) if wall_s > 0 else 0.0,
        "peak_vram_mb": round(peak_vram_mb, 1),
        "peak_rss_mb": round(peak_rss_mb, 1),
    }

    out_path = out_dir / f"bench-{mode}-s{setting}-seed{seed}-{rev}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    log.info("bench written: %s", out_path)
    log.info("summary: %s", json.dumps(report))


if __name__ == "__main__":
    app()
