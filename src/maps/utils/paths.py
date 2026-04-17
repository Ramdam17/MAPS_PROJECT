"""Project filesystem layout — single source of truth for where things live.

The layout is defined declaratively in ``config/paths.yaml`` and resolved here
into a typed ``Paths`` dataclass. This lets scripts write:

    from maps.utils.paths import get_paths
    paths = get_paths()
    torch.save(state, paths.models / "blindsight_seed42.pt")

instead of juggling ``os.path.join(os.environ["MAPS_ROOT"], ...)`` everywhere.

The project root is discovered by (in order):

1. The ``MAPS_ROOT`` environment variable, if set.
2. The directory containing ``config/maps.yaml`` (walks up from cwd).
3. ``cwd`` as a fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from maps.utils.config import load_config

__all__ = ["Paths", "get_paths"]


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem paths. All attributes are absolute :class:`Path`."""

    root: Path
    data: Path
    outputs: Path
    models: Path
    logs: Path
    figures: Path
    reports: Path
    # Internal: track which paths we actually own so ``ensure_dirs`` is safe.
    _owned: tuple[str, ...] = field(
        default=("data", "outputs", "models", "logs", "figures", "reports"),
        repr=False,
    )

    def ensure_dirs(self) -> None:
        """Create every output directory if missing. Never touches ``root``."""
        for name in self._owned:
            getattr(self, name).mkdir(parents=True, exist_ok=True)


def _discover_root() -> Path:
    env = os.environ.get("MAPS_ROOT")
    if env:
        return Path(env).resolve()
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "config" / "paths.yaml").is_file():
            return candidate
    return here


def get_paths(*, root: Path | str | None = None) -> Paths:
    """Load ``config/paths.yaml`` and return a resolved :class:`Paths`.

    Parameters
    ----------
    root : Path | str, optional
        Override the auto-discovered project root. Useful for tests that
        point at a ``tmp_path``.
    """
    resolved_root = Path(root).resolve() if root is not None else _discover_root()

    # Load the raw YAML without resolving interpolations — we substitute root ourselves
    # so callers can override it per-invocation (e.g. for tests).
    cfg = load_config("paths", resolve=False)

    def _substitute(value: str) -> Path:
        # The only interpolation we support is `${.root}/...` or plain paths.
        if value.startswith("${"):
            # Everything from the first "/" onward is the sub-path.
            _, _, tail = value.partition("}/")
            return (resolved_root / tail).resolve() if tail else resolved_root
        p = Path(value)
        return p.resolve() if p.is_absolute() else (resolved_root / p).resolve()

    # Resolve each declared field, preferring the YAML if it's defined there.
    kwargs: dict[str, Path] = {"root": resolved_root}
    for f in fields(Paths):
        if f.name in {"root", "_owned"}:
            continue
        raw = cfg.get(f.name, f.name)  # fallback: use the field name as sub-dir
        kwargs[f.name] = _substitute(str(raw))

    return Paths(**kwargs)
