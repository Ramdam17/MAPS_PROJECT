"""Aggregate Blindsight + AGL reproduction cells into per-setting stats.

Walks ``outputs/{blindsight,agl}/<setting>/seed-<NN>/summary.json``, computes
per-setting mean/std over seeds, z-score of each non-baseline setting vs. the
baseline ("neither") seed distribution, and writes:

- ``outputs/reports/perceptual_summary.json`` — machine-readable
- ``outputs/reports/perceptual_summary.md``   — human-readable table

Exits non-zero if any expected cell is missing (no silent partial reports).

Usage
-----
    uv run python scripts/aggregate_perceptual.py --seeds "42,43,44,45,46,47,48,49,50,51"
    uv run python scripts/aggregate_perceptual.py --domain blindsight --seeds "42,43"
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from statistics import mean, stdev

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maps.utils import configure_logging, get_paths, load_config

app = typer.Typer(add_completion=False, help=__doc__)
log = logging.getLogger("maps.aggregate_perceptual")

BASELINE = "neither"
# Which summary.json field is the "score" for each domain.
# Lower is better for losses — we negate for z-score so "higher = more MAPS gain".
METRIC_FIELD = {
    "blindsight": "loss_1_final",
    "agl": "loss_2_final",
}


def _collect_cells(domain_dir: Path, settings: list[str], seeds: list[int]) -> dict:
    """Load summary.json for every (setting, seed); error if any missing."""
    cells: dict[str, dict[int, dict]] = {s: {} for s in settings}
    missing: list[str] = []
    for s in settings:
        for sd in seeds:
            p = domain_dir / s / f"seed-{sd}" / "summary.json"
            if not p.exists():
                missing.append(str(p.relative_to(domain_dir.parent)))
                continue
            cells[s][sd] = json.loads(p.read_text())
    if missing:
        log.error("Missing %d cells:\n  %s", len(missing), "\n  ".join(missing))
        raise typer.Exit(code=1)
    return cells


def _per_setting_stats(cells: dict, field: str) -> dict[str, dict]:
    """Compute mean/std of `field` across seeds for each setting."""
    out = {}
    for s, by_seed in cells.items():
        vals = [c[field] for c in by_seed.values()]
        out[s] = {
            "n": len(vals),
            "mean": mean(vals),
            "std": stdev(vals) if len(vals) > 1 else 0.0,
            "values": vals,
        }
    return out


def _z_vs_baseline(stats: dict, baseline: str) -> dict[str, float | None]:
    """Z-score of each non-baseline setting's mean vs. baseline's seed distribution.

    Sign convention: negative loss delta (our loss < baseline loss) → positive z.
    """
    b = stats[baseline]
    z = {}
    for s, row in stats.items():
        if s == baseline:
            z[s] = 0.0
            continue
        if b["std"] == 0.0:
            z[s] = None  # undefined
        else:
            # "gain" = baseline − current  (positive means improvement on losses)
            z[s] = (b["mean"] - row["mean"]) / b["std"]
    return z


def _render_md(domain: str, stats: dict, zs: dict, field: str) -> str:
    lines = [
        f"### {domain} — field `{field}`",
        "",
        "| Setting | N | Mean | Std | Z vs. baseline |",
        "|---------|--:|-----:|----:|---------------:|",
    ]
    for s, row in stats.items():
        z = zs[s]
        zstr = f"{z:+.2f}" if z is not None else "n/a"
        lines.append(f"| {s} | {row['n']} | {row['mean']:.4f} | {row['std']:.4f} | {zstr} |")
    return "\n".join(lines)


def _aggregate_domain(domain: str, base_out: Path, settings: list[str], seeds: list[int]) -> dict:
    dom_dir = base_out / domain
    if not dom_dir.exists():
        log.error("Domain dir missing: %s", dom_dir)
        raise typer.Exit(code=1)
    cells = _collect_cells(dom_dir, settings, seeds)
    field = METRIC_FIELD[domain]
    stats = _per_setting_stats(cells, field)
    zs = _z_vs_baseline(stats, BASELINE)
    return {
        "domain": domain,
        "field": field,
        "baseline": BASELINE,
        "seeds": seeds,
        "settings": settings,
        "stats": stats,
        "z_vs_baseline": zs,
    }


@app.command()
def main(
    domain: str = typer.Option("both", help="'blindsight', 'agl', or 'both'."),
    seeds: str = typer.Option(..., "--seeds", help="Comma-separated seed list."),
    log_level: str = typer.Option("INFO", help="Python logging level."),
) -> None:
    configure_logging(level=log_level)
    paths = get_paths()
    factorial = load_config("experiments/factorial_2x2")
    settings = [s.id for s in factorial.settings]
    seed_list = [int(x) for x in seeds.split(",")]

    domains = ["blindsight", "agl"] if domain == "both" else [domain]
    reports_dir = paths.outputs / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    payload = {}
    md_sections = []
    for d in domains:
        log.info("Aggregating %s (%d settings × %d seeds)", d, len(settings), len(seed_list))
        result = _aggregate_domain(d, paths.outputs, settings, seed_list)
        payload[d] = result
        md_sections.append(_render_md(d, result["stats"], result["z_vs_baseline"], result["field"]))

    (reports_dir / "perceptual_summary.json").write_text(json.dumps(payload, indent=2))
    md = "# Perceptual reproduction summary\n\n" + "\n\n".join(md_sections) + "\n"
    (reports_dir / "perceptual_summary.md").write_text(md)
    log.info("Wrote %s", reports_dir / "perceptual_summary.json")
    log.info("Wrote %s", reports_dir / "perceptual_summary.md")


if __name__ == "__main__":
    app()
