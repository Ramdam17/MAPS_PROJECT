"""Centralized stdlib logging configuration.

Single entry point — :func:`configure_logging` — that every MAPS CLI,
sweep driver, or notebook should call once at startup. Stdlib backend
(no loguru, no structlog) — convention CLAUDE.md and lab-wide.

Format
------
``YYYY-MM-DD HH:MM:SS module.name [LEVEL] message`` —
parseable, readable, filter-friendly via ``%(name)s``.

Notes
-----
**Lab hard rule (CLAUDE.md)** : never silence console output, never
disable progress bars, never set third-party loggers to WARNING. The
researcher wants to see everything.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_FORMAT = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: LogLevel | int = "INFO",
    log_file: Path | str | None = None,
    *,
    force: bool = True,
) -> logging.Logger:
    """Configure the root logger with the MAPS-standard format.

    Parameters
    ----------
    level : str or int, optional
        Logging level. String form ``"INFO"`` / ``"DEBUG"`` /
        ``"WARNING"`` / etc., or integer level. Default ``"INFO"``.
    log_file : pathlib.Path or str or None, optional
        If given, add a :class:`logging.FileHandler` (append mode,
        UTF-8) in addition to the stderr ``StreamHandler``. Parent
        directories are created if missing.
    force : bool, optional
        Passed through to :func:`logging.basicConfig`. Default
        ``True`` — clears existing handlers so re-configuring from a
        notebook works as expected.

    Returns
    -------
    logging.Logger
        The ``"maps"`` logger (convenience handle for callers that
        want one named logger to play with).

    Raises
    ------
    ValueError
        If ``level`` is a string but not a recognised level name.
    """
    if isinstance(level, str):
        numeric_level = logging.getLevelName(level.upper())
        if not isinstance(numeric_level, int):
            raise ValueError(f"unknown log level: {level!r}")
    else:
        numeric_level = level

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format=DEFAULT_FORMAT,
        datefmt=DEFAULT_DATEFMT,
        force=force,
        handlers=handlers,
    )
    return logging.getLogger("maps")
