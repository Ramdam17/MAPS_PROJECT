"""Aggregate SARL (MinAtar DQN) reproduction cells into per-setting stats.

Walks ``outputs/sarl/<game>/setting-<N>/seed-<SEED>/metrics.json``, computes
per-(game, setting) mean/std of final-100-episode returns across seeds, and
the z-score of each setting vs. setting 1 (vanilla DQN baseline).

- ``outputs/reports/sarl_summary.json`` — machine-readable
- ``outputs/reports/sarl_summary.md``   — human-readable table

Exits non-zero if any expected cell is missing (no silent partial reports).

Usage
-----
    uv run python scripts/aggregate_sarl.py --seeds "42,43,44,45,46"
    uv run python scripts/aggregate_sarl.py --games "breakout,seaquest" --seeds "42,43"
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from statistics import mean, stdev

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maps.utils import configure_logging, get_paths

app = typer.Typer(add_completion=False, help=__doc__)
log = logging.getLogger("maps.aggregate_sarl")

BASELINE_SETTING = 1
SETTINGS = (1, 2, 3, 4, 5, 6)
DEFAULT_GAMES = ("breakout", "seaquest", "space_invaders", "asterix", "freeway")
FINAL_N_EPISODES = 100  # paper metric: mean return over final 100 episodes


def _final_return(cell: dict) -> float:
    episodes = cell.get("episode_returns") or []
    if not episodes:
        raise ValueError(f"cell has no episode_returns (game={cell.get('game')!r})")
    window = episodes[-FINAL_N_EPISODES:]
    return float(mean(window))


def _collect_cells(game_dir: Path, settings: tuple[int, ...], seeds: list[int]) -> dict:
    cells: dict[int, dict[int, dict]] = {s: {} for s in settings}
    missing: list[str] = []
    for s in settings:
        for sd in seeds:
            p = game_dir / f"setting-{s}" / f"seed-{sd}" / "metrics.json"
            if not p.exists():
                missing.append(str(p.relative_to(game_dir.parent)))
                continue
            cells[s][sd] = json.loads(p.read_text())
    if missing:
        log.error("Missing %d cells:\n  %s", len(missing), "\n  ".join(missing))
        raise typer.Exit(code=1)
    return cells


def _per_setting_stats(cells: dict) -> dict[int, dict]:
    out = {}
    for s, by_seed in cells.items():
        vals = [_final_return(c) for c in by_seed.values()]
        out[s] = {
            "n": len(vals),
            "mean": mean(vals),
            "std": stdev(vals) if len(vals) > 1 else 0.0,
            "values": vals,
        }
    return out


def _z_vs_baseline(stats: dict, baseline: int) -> dict[int, float | None]:
    """Higher return is better → z = (current − baseline) / σ_baseline."""
    b = stats[baseline]
    z = {}
    for s, row in stats.items():
        if s == baseline:
            z[s] = 0.0
        elif b["std"] == 0.0:
            z[s] = None
        else:
            z[s] = (row["mean"] - b["mean"]) / b["std"]
    return z


def _render_md(game: str, stats: dict, zs: dict) -> str:
    lines = [
        f"### {game} — mean return over final {FINAL_N_EPISODES} episodes",
        "",
        "| Setting | N | Mean | Std | Z vs. baseline (1) |",
        "|--------:|--:|-----:|----:|-------------------:|",
    ]
    for s in sorted(stats.keys()):
        row = stats[s]
        z = zs[s]
        zstr = f"{z:+.2f}" if z is not None else "n/a"
        lines.append(f"| {s} | {row['n']} | {row['mean']:.3f} | {row['std']:.3f} | {zstr} |")
    return "\n".join(lines)


@app.command()
def main(
    games: str = typer.Option(
        ",".join(DEFAULT_GAMES),
        "--games",
        help="Comma-separated game list. Default: all 5 paper games.",
    ),
    seeds: str = typer.Option(..., "--seeds", help="Comma-separated seed list."),
    settings: str = typer.Option(
        ",".join(str(s) for s in SETTINGS),
        "--settings",
        help="Comma-separated setting list (1-6). Default: all 6.",
    ),
    log_level: str = typer.Option("INFO", help="Python logging level."),
) -> None:
    configure_logging(level=log_level)
    paths = get_paths()
    game_list = [g.strip() for g in games.split(",") if g.strip()]
    seed_list = [int(x) for x in seeds.split(",")]
    setting_list = tuple(int(x) for x in settings.split(","))

    base_out = paths.outputs / "sarl"
    reports_dir = paths.outputs / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    payload: dict = {"baseline_setting": BASELINE_SETTING, "games": {}}
    md_sections = []
    for g in game_list:
        game_dir = base_out / g
        if not game_dir.exists():
            log.error("Game dir missing: %s", game_dir)
            raise typer.Exit(code=1)
        log.info("Aggregating %s (%d settings × %d seeds)", g, len(setting_list), len(seed_list))
        cells = _collect_cells(game_dir, setting_list, seed_list)
        stats = _per_setting_stats(cells)
        zs = _z_vs_baseline(stats, BASELINE_SETTING)
        payload["games"][g] = {
            "settings": list(setting_list),
            "seeds": seed_list,
            "stats": {str(k): v for k, v in stats.items()},
            "z_vs_baseline": {str(k): v for k, v in zs.items()},
        }
        md_sections.append(_render_md(g, stats, zs))

    (reports_dir / "sarl_summary.json").write_text(json.dumps(payload, indent=2))
    md = "# SARL reproduction summary\n\n" + "\n\n".join(md_sections) + "\n"
    (reports_dir / "sarl_summary.md").write_text(md)
    log.info("Wrote %s", reports_dir / "sarl_summary.json")
    log.info("Wrote %s", reports_dir / "sarl_summary.md")


if __name__ == "__main__":
    app()
