"""OmegaConf YAML loader with Hydra-light composition.

Two-level convention :

- ``config/maps.yaml`` — canonical paper constants (locked).
- ``config/domains/<x>/training.yaml`` — domain hyperparams that compose
  from ``maps.yaml`` via ``defaults: [/maps@_here_]``.

API public : :func:`load_config` (file → :class:`DictConfig`) +
:func:`parse_overrides` (CLI helper).

D12.1 — Sprint 12 : we stay OmegaConf-pure. No typed schema layer.
Trade-off : we accept the cost of silent config typos in exchange for
~150 LOC less and zero dep beyond OmegaConf. Revisit post-Phase F if
typos accumulate.

Hydra compat (partial) :

- ``@_here_`` suffix in defaults : accepted, ignored (we always merge
  at root).
- ``_self_`` sentinel in defaults : accepted, ignored (we always apply
  self after defaults).
- ``${...}`` interpolations : resolved at load time (default
  ``resolve=True``) unless ``resolve=False`` is passed.
- Override syntax ``key=value`` via :func:`omegaconf.OmegaConf.from_dotlist`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _find_config_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` (or cwd) until we find a dir
    containing ``config/maps.yaml``. Falls back to cwd if not found.

    The discovery happens at module import time and the result is
    cached in :data:`CONFIG_ROOT`. **Caveat**: if a process changes
    ``cwd`` after import, ``CONFIG_ROOT`` will still point at the
    discovery-time root. In practice this is a non-issue when running
    via ``uv run`` from the repo root.
    """
    current = (start or Path.cwd()).resolve()
    while current != current.parent:
        if (current / "config" / "maps.yaml").is_file():
            return current
        current = current.parent
    return Path.cwd().resolve()


CONFIG_ROOT: Path = _find_config_root()


def _resolve_config_path(name: str | Path) -> Path:
    """Convert a config name to an absolute filesystem path.

    Accepts :

    - Absolute :class:`pathlib.Path` → returned as-is.
    - String ending in ``.yaml`` or ``.yml`` :
      - absolute → returned as :class:`Path`
      - relative → resolved against cwd
    - Bare name (e.g. ``"maps"`` or ``"domains/blindsight/training"``) :
      resolved as ``CONFIG_ROOT / "config" / "{name}.yaml"``.
    """
    if isinstance(name, Path):
        return name.resolve()

    if name.endswith(".yaml") or name.endswith(".yml"):
        p = Path(name)
        return p if p.is_absolute() else p.resolve()

    return CONFIG_ROOT / "config" / f"{name}.yaml"


def _apply_defaults(cfg: DictConfig, visited: set[Path] | None = None) -> DictConfig:
    """Recursively merge any ``defaults:`` list in ``cfg`` into a
    composed :class:`DictConfig`.

    Cycle detection : if a referenced file is already in ``visited``,
    raise :class:`ValueError`. Prevents stack-overflow on circular
    defaults.

    Sentinel handling :

    - ``_self_`` : ignored (we always apply self after defaults).
    - ``@_here_`` suffix : stripped, ignored (we always merge at root).
    """
    if visited is None:
        visited = set()

    if "defaults" not in cfg:
        return cfg

    defaults_raw = cfg.pop("defaults")
    # OmegaConf may give us ListConfig or list — coerce to list
    defaults_list = list(defaults_raw)

    merged: DictConfig = OmegaConf.create({})  # type: ignore[assignment]
    for ref in defaults_list:
        if not isinstance(ref, str):
            # Hydra also supports dict-form defaults; we don't.
            continue
        ref_clean = ref.strip("/")
        ref_clean = ref_clean.split("@")[0]  # drop @_here_ suffix
        if ref_clean == "_self_":
            continue

        ref_path = _resolve_config_path(ref_clean)
        if not ref_path.exists():
            raise FileNotFoundError(
                f"defaults reference not found: {ref_path} (while loading composed config)"
            )
        if ref_path in visited:
            raise ValueError(f"cycle detected in defaults: {ref_path}")
        visited.add(ref_path)

        sub_cfg = OmegaConf.load(ref_path)
        if not isinstance(sub_cfg, DictConfig):
            raise ValueError(
                f"defaults reference {ref_path} is not a mapping; got {type(sub_cfg).__name__}"
            )
        sub_cfg = _apply_defaults(sub_cfg, visited=visited)
        merged = OmegaConf.merge(merged, sub_cfg)  # type: ignore[assignment]

    # Overlay the current cfg (sans defaults) on top of the merged base.
    final = OmegaConf.merge(merged, cfg)
    return final  # type: ignore[return-value]


def load_config(
    name: str | Path,
    *,
    overrides: Sequence[str] | None = None,
    resolve: bool = True,
) -> DictConfig:
    r"""Load a YAML config with composition + CLI-style overrides.

    Parameters
    ----------
    name : str or pathlib.Path
        Bare name relative to ``config/`` (e.g. ``"maps"``,
        ``"domains/blindsight/training"``) **or** an absolute path
        ending in ``.yaml``.
    overrides : sequence of str, optional
        Hydra-style dot-path overrides, e.g.
        ``["train.n_epochs=10", "cascade.alpha=0.05"]``.
    resolve : bool, optional
        If ``True`` (default), resolve ``${...}`` interpolations. Set
        ``False`` when loading a partial config that interpolates
        values defined elsewhere (e.g. ``domains/blindsight/env.yaml``
        references ``${train.noise_level}`` which lives in
        ``training.yaml``).

    Returns
    -------
    omegaconf.DictConfig
        Composed and overridden config.

    Raises
    ------
    FileNotFoundError
        If the requested file does not exist (or any referenced default).
    ValueError
        If top-level YAML is not a mapping, or if a cycle is detected
        in ``defaults``.

    Examples
    --------
    >>> cfg = load_config("maps")
    >>> cfg.cascade.alpha
    0.02

    >>> cfg = load_config("domains/blindsight/training")
    >>> cfg.first_order.input_dim  # Blindsight override
    100
    >>> cfg.cascade.alpha          # inherited from maps.yaml
    0.02

    >>> cfg = load_config("maps", overrides=["cascade.alpha=0.1"])
    >>> cfg.cascade.alpha
    0.1
    """
    path = _resolve_config_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"config top-level must be a mapping; got {type(cfg).__name__} at {path}")

    cfg = _apply_defaults(cfg)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, override_cfg)  # type: ignore[assignment]

    if resolve:
        OmegaConf.resolve(cfg)

    logger.debug("Loaded config %s with %d overrides", name, len(overrides or []))
    return cfg


def parse_overrides(*tokens: str) -> list[str]:
    """Validate and return a list of CLI override tokens.

    Each token must be of the form ``key.path=value``. Trivial
    validation here (presence of ``=``) — full parsing happens in
    :func:`omegaconf.OmegaConf.from_dotlist`.

    Parameters
    ----------
    *tokens : str
        Strings like ``"train.n_epochs=10"``.

    Returns
    -------
    list of str
        The same tokens, validated.

    Raises
    ------
    ValueError
        If any token lacks an ``=`` separator.
    """
    out = []
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"override token must be 'key=value', got {token!r}")
        out.append(token)
    return out
