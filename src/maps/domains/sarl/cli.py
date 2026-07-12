"""SARL CLI — Typer entry point (MinAtar DQN, one or all settings × seeds).

Pipeline per ``(game, setting, seed)``: build the MinAtar env → ``run_training``
(episodic DQN) → ``evaluate_greedy`` → save nets + metrics + ``summary.json``.

**Env seeding (D14.3):** ``set_all_seeds`` covers torch/numpy/python (net init,
replay sampling, ε binomial), but the MinAtar env's internal RNG is **not**
seeded — faithful to ``maps_v1.py`` (the source never seeds it). Reproducible
per-seed env trajectories are a separate production decision, not part of this
faithful reproduction.

Usage
-----

.. code-block:: bash

    uv run python -m maps.domains.sarl.cli --game breakout --setting 6 --seed 42
    # v1 paper-canonical full run (Table 6):
    uv run python -m maps.domains.sarl.cli -o training.num_frames=2000000 -o alpha=25

Outputs land in ``$SCRATCH/maps/outputs/sarl/<game>/setting-<N>/seed-<S>/``.
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

from maps.domains.sarl.evaluate import evaluate_greedy
from maps.domains.sarl.trainer import SETTINGS_REGISTRY
from maps.domains.sarl.training_loop import run_training
from maps.utils.config import load_config, parse_overrides
from maps.utils.device import get_device
from maps.utils.logging_setup import configure_logging
from maps.utils.paths import get_paths
from maps.utils.seeding import set_all_seeds

logger = logging.getLogger(__name__)

app = typer.Typer(name="sarl", help="SARL Sprint 14 CLI — MinAtar DQN.", no_args_is_help=False)

_SETTING_BY_NUMBER = {
    1: "setting-1-baseline",
    2: "setting-2-cascade-1st",
    3: "setting-3-second-order-only",
    4: "setting-4-maps-1st",
    5: "setting-5-cascade-2nd",
    6: "setting-6-full-maps",
}


def _load_cfg(overrides: list[str]) -> DictConfig:
    cfg = load_config("domains/sarl/training")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg  # type: ignore[return-value]


def _resolve_output_dir(game: str, setting_id: str, seed: int, override: Path | None) -> Path:
    if override is not None:
        return override / game / setting_id / f"seed-{seed}"
    paths = get_paths()
    return paths.scratch_root / "maps" / "outputs" / "sarl" / game / setting_id / f"seed-{seed}"


@app.command()
def main(
    game: Annotated[str, typer.Option(help="MinAtar game.")] = "breakout",
    setting: Annotated[int, typer.Option(help="Paper setting 1-6.")] = 6,
    seed: Annotated[int, typer.Option(help="Random seed (torch/numpy/python).")] = 42,
    num_frames: Annotated[
        int | None, typer.Option(help="Override cfg.training.num_frames.")
    ] = None,
    eval_episodes: Annotated[int, typer.Option(help="Greedy evaluation episodes.")] = 10,
    overrides: Annotated[
        list[str] | None,
        typer.Option("-o", "--override", help="Hydra-style override 'key=value'."),
    ] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Override base output directory.")] = None,
    device: Annotated[str, typer.Option(help="'auto' (default), 'cpu', 'cuda'.")] = "auto",
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Run one SARL (game, setting, seed) MinAtar DQN training + evaluation."""
    configure_logging(level=log_level)
    if setting not in _SETTING_BY_NUMBER:
        raise typer.BadParameter(f"setting must be 1-6, got {setting}")
    setting_id = _SETTING_BY_NUMBER[setting]
    setting_obj = SETTINGS_REGISTRY[setting_id]

    cfg = _load_cfg(parse_overrides(*(overrides or [])))
    if num_frames is not None:
        cfg.training.num_frames = num_frames
    # The --game arg is authoritative (cfg.game is only the YAML default).
    torch_device = get_device(device)  # type: ignore[arg-type]

    # torch/numpy/python seeded; MinAtar env RNG deliberately NOT (D14.3).
    set_all_seeds(seed)
    from minatar import Environment

    env = Environment(game)

    out_dir = _resolve_output_dir(game, setting_id, seed, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("▶ SARL game=%s setting=%s seed=%d → %s", game, setting_id, seed, out_dir)

    t0 = time.perf_counter()
    policy_net, second_order_net, metrics = run_training(
        setting_obj, cfg, env, num_frames=num_frames, device=torch_device
    )
    train_seconds = time.perf_counter() - t0

    eval_metrics = evaluate_greedy(
        policy_net,
        env,
        n_episodes=eval_episodes,
        cascade_iterations_1=setting_obj.cascade_iterations_1,
        device=torch_device,
    )

    np.save(out_dir / "episode_returns.npy", np.asarray(metrics.episode_returns))
    torch.save(policy_net.state_dict(), out_dir / "policy_net.pt")
    if second_order_net is not None:
        torch.save(second_order_net.state_dict(), out_dir / "second_order_net.pt")

    summary = {
        "game": game,
        "setting": setting_id,
        "seed": seed,
        "meta": setting_obj.meta,
        "cascade_iterations_1": setting_obj.cascade_iterations_1,
        "cascade_iterations_2": setting_obj.cascade_iterations_2,
        "num_frames": metrics.total_frames,
        "total_updates": metrics.total_updates,
        "n_episodes": len(metrics.episode_returns),
        "final_return_mean_last_100": float(np.mean(metrics.episode_returns[-100:]))
        if metrics.episode_returns
        else 0.0,
        "eval": eval_metrics,
        "train_seconds": train_seconds,
    }
    with (out_dir / "summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    logger.info(
        "✓ Done game=%s setting=%s seed=%d in %.1fs — eval mean return=%.3f",
        game,
        setting_id,
        seed,
        train_seconds,
        eval_metrics["mean_return"],
    )


if __name__ == "__main__":
    app()
