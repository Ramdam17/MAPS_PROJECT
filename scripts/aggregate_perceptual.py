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
# Nested paths use dot notation (resolved in `_extract_metric`).
# For Blindsight, the paper's headline is wager accuracy under the
# superthreshold condition (detection task). AGL still uses loss_2_final
# until the awareness-split eval is ported (RG-001).
METRIC_FIELD = {
    # Discrimination accuracy on superthreshold stimuli is defined for all
    # 4 settings (including the baseline without second-order). Wager
    # accuracy only exists when second_order=True, so we pick discrimination
    # as the headline axis and report wager stats separately when available.
    "blindsight": "eval.superthreshold.discrimination_accuracy",
    "agl": "eval.classification_precision",
}
# For accuracies, higher is better → positive z = MAPS improvement.
HIGHER_IS_BETTER = {
    "blindsight": True,
    "agl": True,
}


def _extract_metric(cell: dict, path: str) -> float:
    """Dot-path lookup: 'eval.superthreshold.wager_accuracy'."""
    v: object = cell
    for key in path.split("."):
        if not isinstance(v, dict) or key not in v:
            raise KeyError(f"Missing key {path!r} in summary (stopped at {key!r})")
        v = v[key]
    return float(v)  # type: ignore[arg-type]


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
        vals = [_extract_metric(c, field) for c in by_seed.values()]
        out[s] = {
            "n": len(vals),
            "mean": mean(vals),
            "std": stdev(vals) if len(vals) > 1 else 0.0,
            "values": vals,
        }
    return out


def _z_vs_baseline(stats: dict, baseline: str, higher_is_better: bool) -> dict[str, float | None]:
    """Z-score of each non-baseline setting's mean vs. baseline's seed distribution.

    Sign convention: positive z always means "MAPS gain over baseline".
    When `higher_is_better`, that's (current − baseline) / σ_baseline.
    When losses, it's (baseline − current) / σ_baseline.
    """
    b = stats[baseline]
    z = {}
    for s, row in stats.items():
        if s == baseline:
            z[s] = 0.0
            continue
        if b["std"] == 0.0:
            z[s] = None
        else:
            delta = (row["mean"] - b["mean"]) if higher_is_better else (b["mean"] - row["mean"])
            z[s] = delta / b["std"]
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
    zs = _z_vs_baseline(stats, BASELINE, higher_is_better=HIGHER_IS_BETTER[domain])
    result = {
        "domain": domain,
        "field": field,
        "baseline": BASELINE,
        "seeds": seeds,
        "settings": settings,
        "stats": stats,
        "z_vs_baseline": zs,
    }

    # AGL paper reports High Awareness vs Low Awareness — a post-hoc seed-pool
    # split (reference `metrics_testing_table` L1276: first half = high, second
    # half = low). Replicate that at aggregation time when we have an even
    # number of seeds.
    if domain == "agl" and len(seeds) >= 2 and len(seeds) % 2 == 0:
        mid = len(seeds) // 2
        hi_seeds, lo_seeds = seeds[:mid], seeds[mid:]
        hi_cells = _collect_cells(dom_dir, settings, hi_seeds)
        lo_cells = _collect_cells(dom_dir, settings, lo_seeds)
        result["awareness_split"] = {
            "high": {
                "seeds": hi_seeds,
                "stats": _per_setting_stats(hi_cells, field),
            },
            "low": {
                "seeds": lo_seeds,
                "stats": _per_setting_stats(lo_cells, field),
            },
        }
    return result


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
        if "awareness_split" in result:
            for tier in ("high", "low"):
                sub = result["awareness_split"][tier]
                sub_zs = _z_vs_baseline(
                    sub["stats"], BASELINE, higher_is_better=HIGHER_IS_BETTER[d]
                )
                md_sections.append(
                    _render_md(
                        f"{d} — {tier} awareness (seeds {sub['seeds']})",
                        sub["stats"],
                        sub_zs,
                        result["field"],
                    )
                )

    (reports_dir / "perceptual_summary.json").write_text(json.dumps(payload, indent=2))
    md = "# Perceptual reproduction summary\n\n" + "\n\n".join(md_sections) + "\n"
    (reports_dir / "perceptual_summary.md").write_text(md)
    log.info("Wrote %s", reports_dir / "perceptual_summary.json")
    log.info("Wrote %s", reports_dir / "perceptual_summary.md")


if __name__ == "__main__":
    app()
