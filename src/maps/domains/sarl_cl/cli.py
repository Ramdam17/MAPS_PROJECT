"""SARL+CL CLI — Typer entry point (continual-learning curriculum).

Runs a curriculum over a list of MinAtar games on one channel-adaptive network
(:func:`maps.domains.sarl_cl.training_loop.run_curriculum`) and saves the final
network + per-stage metrics + ``summary.json``.

Scope (D15.6): CL mechanics + small curriculum. Full cross-game forgetting
evaluation (Figure 7) is deferred pending Guillaume's D4. MinAtar env RNG is
unseeded (faithful, D15.2/D15.3).

Usage
-----

.. code-block:: bash

    uv run python -m maps.domains.sarl_cl.cli \
        --games breakout,space_invaders --setting 3 --frames-per-stage 2000
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

from maps.domains.sarl_cl.trainer import SETTINGS_REGISTRY
from maps.domains.sarl_cl.training_loop import run_curriculum
from maps.utils.config import load_config, parse_overrides
from maps.utils.device import get_device
from maps.utils.logging_setup import configure_logging
from maps.utils.paths import get_paths
from maps.utils.seeding import set_all_seeds

logger = logging.getLogger(__name__)

app = typer.Typer(name="sarl_cl", help="SARL+CL Sprint 15 CLI — curriculum.", no_args_is_help=False)

_SETTING_BY_NUMBER = {
    1: "setting-1-baseline",
    2: "setting-2-cascade-1st",
    3: "setting-3-second-order-only",
    4: "setting-4-maps-1st",
    5: "setting-5-cascade-2nd",
    6: "setting-6-full-maps",
}


def _load_cfg(overrides: list[str]) -> DictConfig:
    cfg = load_config("domains/sarl_cl/training")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg  # type: ignore[return-value]


@app.command()
def main(
    games: Annotated[
        str, typer.Option(help="Comma-separated MinAtar games (curriculum order).")
    ] = "breakout,space_invaders",
    setting: Annotated[int, typer.Option(help="Paper setting 1-6.")] = 3,
    seed: Annotated[int, typer.Option(help="Random seed (torch/numpy/python).")] = 42,
    frames_per_stage: Annotated[int, typer.Option(help="Frames per curriculum stage.")] = 2000,
    overrides: Annotated[
        list[str] | None, typer.Option("-o", "--override", help="Hydra-style 'key=value'.")
    ] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Override base output directory.")] = None,
    device: Annotated[str, typer.Option(help="'auto' (default), 'cpu', 'cuda'.")] = "auto",
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Run a SARL+CL curriculum over the given games."""
    configure_logging(level=log_level)
    if setting not in _SETTING_BY_NUMBER:
        raise typer.BadParameter(f"setting must be 1-6, got {setting}")
    setting_id = _SETTING_BY_NUMBER[setting]
    setting_obj = SETTINGS_REGISTRY[setting_id]
    game_list = [g.strip() for g in games.split(",") if g.strip()]

    cfg = _load_cfg(parse_overrides(*(overrides or [])))
    torch_device = get_device(device)  # type: ignore[arg-type]
    set_all_seeds(seed)  # env RNG deliberately unseeded (D15.3)

    if output_dir is not None:
        out_dir = output_dir / ("-".join(game_list)) / setting_id / f"seed-{seed}"
    else:
        out_dir = (
            get_paths().scratch_root
            / "maps"
            / "outputs"
            / "sarl_cl"
            / "-".join(game_list)
            / setting_id
            / f"seed-{seed}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("▶ SARL+CL games=%s setting=%s seed=%d → %s", game_list, setting_id, seed, out_dir)

    t0 = time.perf_counter()
    policy_net, stage_metrics = run_curriculum(
        game_list, cfg, setting_obj, frames_per_stage=frames_per_stage, device=torch_device
    )
    elapsed = time.perf_counter() - t0

    torch.save(policy_net.state_dict(), out_dir / "policy_net_final.pt")
    for m in stage_metrics:
        np.save(out_dir / f"returns_{m.game}.npy", np.asarray(m.episode_returns))

    summary = {
        "games": game_list,
        "setting": setting_id,
        "seed": seed,
        "meta": setting_obj.meta,
        "frames_per_stage": frames_per_stage,
        "stages": [
            {
                "game": m.game,
                "total_frames": m.total_frames,
                "total_updates": m.total_updates,
                "final_return_mean_last_50": float(np.mean(m.episode_returns[-50:]))
                if m.episode_returns
                else 0.0,
            }
            for m in stage_metrics
        ],
        "elapsed_seconds": elapsed,
    }
    with (out_dir / "summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    logger.info("✓ Done — %d stages in %.1fs", len(stage_metrics), elapsed)


if __name__ == "__main__":
    app()
