"""Filesystem path helpers — load ``config/paths.yaml`` and expose typed
accessors.

Project-root resolution priority :

1. ``MAPS_ROOT`` environment variable
2. :data:`maps.utils.config.CONFIG_ROOT` (auto-discovered parent of
   ``config/maps.yaml``)
3. cwd

Scratch-root resolution priority (for HPC vs dev) :

1. ``SCRATCH`` environment variable (DRAC / SLURM convention)
2. ``paths.yaml`` ``scratch_root`` field

The result is cached (lru_cache) — call :func:`get_paths.cache_clear`
to force re-evaluation after changing environment variables.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from maps.utils.config import CONFIG_ROOT, load_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Paths:
    """Typed view of the MAPS filesystem layout.

    Attributes are absolute :class:`pathlib.Path` instances. Anything
    relative in ``paths.yaml`` is resolved against :attr:`root`.

    Attributes
    ----------
    root : pathlib.Path
        Project root (parent of ``config/maps.yaml``, or ``$MAPS_ROOT``).
    data : pathlib.Path
        Raw data directory (gitignored).
    outputs : pathlib.Path
        Run outputs directory (gitignored).
    models : pathlib.Path
        Saved model checkpoints (gitignored).
    logs : pathlib.Path
        Log files directory (gitignored).
    figures : pathlib.Path
        Saved figure output directory.
    reports : pathlib.Path
        Generated reports directory.
    scratch_root : pathlib.Path
        HPC scratch / SLURM workdir. ``$SCRATCH`` wins when set.
    """

    root: Path
    data: Path
    outputs: Path
    models: Path
    logs: Path
    figures: Path
    reports: Path
    scratch_root: Path


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    """Return the resolved :class:`Paths` for this project.

    Uses :data:`maps.utils.config.CONFIG_ROOT` (or ``$MAPS_ROOT`` env
    override) as project root, then loads ``config/paths.yaml`` and
    resolves each path against root. ``$SCRATCH`` env wins over
    ``paths.yaml:scratch_root`` when set.

    Returns
    -------
    Paths
        Frozen dataclass with absolute paths.

    Notes
    -----
    Cached via :func:`functools.lru_cache`. Call
    ``get_paths.cache_clear()`` to force re-evaluation (useful in
    tests).
    """
    root_env = os.environ.get("MAPS_ROOT")
    root = Path(root_env).resolve() if root_env else CONFIG_ROOT

    paths_cfg = load_config("paths")

    def _resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (root / p)

    scratch_env = os.environ.get("SCRATCH")
    scratch_root = Path(scratch_env).resolve() if scratch_env else _resolve(paths_cfg.scratch_root)

    paths = Paths(
        root=root,
        data=_resolve(paths_cfg.data),
        outputs=_resolve(paths_cfg.outputs),
        models=_resolve(paths_cfg.models),
        logs=_resolve(paths_cfg.logs),
        figures=_resolve(paths_cfg.figures),
        reports=_resolve(paths_cfg.reports),
        scratch_root=scratch_root,
    )
    logger.debug("Resolved paths: root=%s, scratch=%s", paths.root, paths.scratch_root)
    return paths
