"""Status report for SARL+CL cells.

Walks ``$SCRATCH/maps/outputs/sarl_cl/`` (or ``--root <path>``) and reports
per-cell state — DONE / PROGRESS / NEVER — for deciding which cells to
``--resume-from`` after a SLURM timeout.

Default is filesystem-only (fast, ~1 s for 150 cells). Use ``--detail`` to
open each ``checkpoint.pt`` and report exact frame counts (slower, ~30 s
for 150 cells, loads each checkpoint into CPU RAM one at a time).

Usage
-----
    uv run python scripts/slurm/sarl_cl_status.py
    uv run python scripts/slurm/sarl_cl_status.py --detail
    uv run python scripts/slurm/sarl_cl_status.py --root /custom/path
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path


def _fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h{int((seconds % 3600) / 60):02d}m"
    return f"{int(seconds / 86400)}d{int((seconds % 86400) / 3600):02d}h"


def _read_ckpt_progress(p: Path) -> tuple[int | None, int | None, int | None]:
    """Open checkpoint.pt and return (t, episode_idx, num_frames_target)."""
    import torch  # local — only needed in --detail mode

    payload = torch.load(p, map_location="cpu", weights_only=False)
    t = payload.get("t")
    ep = payload.get("episode_idx")
    cfg = payload.get("cfg_snapshot") or {}
    return t, ep, cfg.get("num_frames")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    default_root = ""
    if scratch := os.environ.get("SCRATCH"):
        default_root = f"{scratch}/maps/outputs/sarl_cl"
    ap.add_argument("--root", default=default_root, help="root of sarl_cl outputs")
    ap.add_argument(
        "--detail",
        action="store_true",
        help="open checkpoint.pt to read exact frame counts (slow)",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"root not found: {root!s} (set --root or SCRATCH)")

    now = time.time()
    rows: list[tuple] = []
    for game_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for setting_dir in sorted(p for p in game_dir.iterdir() if p.is_dir()):
            try:
                setting = int(setting_dir.name.split("-")[1])
            except (IndexError, ValueError):
                continue
            for seed_dir in sorted(p for p in setting_dir.iterdir() if p.is_dir()):
                try:
                    seed = int(seed_dir.name.split("-")[1])
                except (IndexError, ValueError):
                    continue

                ckpt = seed_dir / "checkpoint.pt"
                metrics = seed_dir / "metrics.json"

                state = "NEVER"
                frames = "—"
                pct = "—"
                age = "—"

                if metrics.is_file():
                    state = "DONE"
                    try:
                        m = json.loads(metrics.read_text())
                        tot = m.get("total_frames")
                        target = m.get("num_frames", 5_000_000)
                        if isinstance(tot, int):
                            frames = f"{tot:,}"
                            pct = f"{100 * tot / target:.0f}%"
                    except (json.JSONDecodeError, OSError):
                        pass
                elif ckpt.is_file():
                    state = "PROGRESS"
                    age = _fmt_age(now - ckpt.stat().st_mtime)
                    if args.detail:
                        try:
                            t, _ep, target = _read_ckpt_progress(ckpt)
                            if t is not None:
                                frames = f"{t:,}"
                            if t is not None and target:
                                pct = f"{100 * t / target:.0f}%"
                        except Exception as exc:  # noqa: BLE001
                            frames = f"err: {type(exc).__name__}"

                rows.append((game_dir.name, setting, seed, state, frames, pct, age))

    fmt = "{:<16}  {:>2}  {:>4}  {:<8}  {:>13}  {:>5}  {:>8}"
    print(fmt.format("GAME", "S", "SEED", "STATE", "FRAMES", "%", "CKPT_AGE"))
    print("-" * 70)
    for row in rows:
        print(fmt.format(*row))

    counts = Counter(r[3] for r in rows)
    print("-" * 70)
    print("Total:", len(rows), "|", " | ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
