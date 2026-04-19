"""Print the top-N functions from a cProfile dump, sorted by cumulative time.

Usage:
    uv run --offline python scripts/analyze_profile.py prof.out [N=40]

Written for Sprint 07 Phase 2 — identifies SARL bottlenecks without ever
running torch.profiler (which sometimes crashes on H100 + recent CUDA).
"""

from __future__ import annotations

import pstats
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    prof_path = Path(sys.argv[1])
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    stats = pstats.Stats(str(prof_path))
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(top_n)

    # Also a "tottime" slice — catches hotspots masked by deep cumulative trees.
    print(f"\n{'=' * 40} sort: tottime {'=' * 40}\n")
    stats.sort_stats("tottime")
    stats.print_stats(top_n)


if __name__ == "__main__":
    main()
