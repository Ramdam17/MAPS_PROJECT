"""YAML config loading with OmegaConf + lightweight Hydra-style composition.

MAPS configs follow a two-level pattern:

1.  `config/maps.yaml` holds **locked** scientific constants (cascade α,
    hidden dims, wager unit counts, seeds...). These values must not drift
    between runs — any deviation is logged in `docs/reproduction/deviations.md`.
2.  Domain configs (`config/training/{blindsight,agl}.yaml`) extend
    `maps.yaml` with training-loop hyperparameters (optimizer, lr, epochs)
    and override shared dims where needed (Blindsight uses 100-d input).

The composition directive used at the top of each domain YAML is::

    defaults:
      - /maps@_here_

which tells `load_config` to merge `config/maps.yaml` into the root of the
document before applying the rest. This mirrors Hydra's `defaults:` syntax
without pulling in the full Hydra runtime — we only need static composition.

Examples
--------
>>> cfg = load_config("training/blindsight")
>>> cfg.cascade.alpha
0.02
>>> cfg.first_order.input_dim  # overridden by blindsight.yaml
100
>>> cfg = load_config("training/agl", overrides=["train.n_epochs_pre=10"])
>>> cfg.train.n_epochs_pre
10
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["CONFIG_ROOT", "load_config"]


def _find_project_root(start: Path | None = None) -> Path:
    """Walk up until we find a directory with ``config/maps.yaml``.

    Falls back to ``cwd`` if nothing is found — in that case the caller
    will get a clear ``FileNotFoundError`` at load time, which is preferable
    to silently guessing the wrong path.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "config" / "maps.yaml").is_file():
            return candidate
    return Path.cwd()


CONFIG_ROOT: Path = _find_project_root() / "config"


def _resolve_config_path(name: str) -> Path:
    """Accept `"maps"`, `"training/blindsight"`, or an absolute/relative path."""
    p = Path(name)
    if p.is_absolute() and p.is_file():
        return p
    if p.suffix in {".yaml", ".yml"} and p.is_file():
        return p.resolve()
    # Bare name like "training/blindsight" → config/training/blindsight.yaml.
    return CONFIG_ROOT / f"{name}.yaml"


def _apply_defaults(cfg: DictConfig, path: Path) -> DictConfig:
    """Resolve a `defaults:` list at the top of `cfg`.

    Only supports the minimal form we use:
        defaults:
          - /maps@_here_   # merge config/maps.yaml at the document root
          - training/foo   # merge config/training/foo.yaml at the root

    The `@_here_` suffix is optional; we always merge at the root for now
    since that's what every current MAPS config needs. The trailing
    `_self_` sentinel is tolerated but ignored (it's the Hydra convention
    meaning "apply my own keys after the defaults are merged", which is
    also what we do by default).
    """
    defaults = cfg.pop("defaults", None)
    if defaults is None:
        return cfg
    if not isinstance(defaults, ListConfig):
        raise ValueError(f"`defaults:` must be a list in {path}, got {type(defaults).__name__}")

    base = OmegaConf.create({})
    for entry in defaults:
        if entry == "_self_":
            continue
        # Strip leading slash and `@_here_` suffix; we only support root-merge.
        name = str(entry).lstrip("/").split("@", 1)[0]
        child_path = _resolve_config_path(name)
        if not child_path.is_file():
            raise FileNotFoundError(f"`defaults:` references missing config: {child_path}")
        child = OmegaConf.load(child_path)
        child = _apply_defaults(child, child_path)  # recursive: children may compose too
        base = OmegaConf.merge(base, child)

    # The current doc's own keys win over anything it inherits (Hydra semantics).
    return OmegaConf.merge(base, cfg)  # type: ignore[return-value]


def load_config(
    name: str,
    *,
    overrides: list[str] | None = None,
    resolve: bool = True,
) -> DictConfig:
    """Load a YAML config, resolve `defaults:` composition, apply CLI overrides.

    Parameters
    ----------
    name : str
        Config identifier — either a bare name relative to ``config/``
        (e.g. ``"maps"``, ``"training/blindsight"``, ``"experiments/factorial_2x2"``)
        or an absolute path to a `.yaml` file.
    overrides : list[str], optional
        Dot-path overrides in Hydra syntax, e.g.
        ``["train.n_epochs=10", "cascade.alpha=0.05"]``.
    resolve : bool, default True
        If True, resolve OmegaConf interpolations (``${...}``) eagerly.
        Set False when you need to inspect raw interpolations.

    Returns
    -------
    DictConfig
        The merged configuration.

    Raises
    ------
    FileNotFoundError
        If `name` cannot be resolved to an existing YAML file.
    ValueError
        If a `defaults:` entry is malformed.
    """
    path = _resolve_config_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    cfg: Any = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"Top-level config must be a mapping, got {type(cfg).__name__} in {path}")

    cfg = _apply_defaults(cfg, path)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    if resolve:
        # Resolve interpolations in place; raises on missing refs, which is what we want.
        OmegaConf.resolve(cfg)

    return cfg  # type: ignore[return-value]
