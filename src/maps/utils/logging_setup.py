"""Logging configuration helper for MAPS scripts and notebooks.

Centralizes the logging format so every entry point (scripts/run_*.py, sweep
drivers, notebooks) uses the same timestamp / name / level layout, and all of
them accept an optional file handler for run archival.

Usage
-----
    from maps.utils import configure_logging

    configure_logging(level="INFO")                     # stderr only
    configure_logging(level="DEBUG", log_file=path)     # stderr + file

This module deliberately does NOT silence any progress bars or third-party
loggers — keep the firehose visible (lab convention).
"""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    *,
    force: bool = True,
) -> logging.Logger:
    """Configure the root logger with a stderr handler (and optional file handler).

    Parameters
    ----------
    level : str | int, default "INFO"
        Logging level name (``"DEBUG"``, ``"INFO"``, ...) or numeric level.
    log_file : str | Path, optional
        If given, also tee all log records to this file (parent dirs are created).
        The file is opened in append mode so re-runs of a CLI accumulate history.
    force : bool, default True
        Pass-through to ``logging.basicConfig`` — clears any pre-existing handlers
        so calling this from a notebook cell a second time works as expected.

    Returns
    -------
    logging.Logger
        The configured root logger (handy for chained calls).
    """
    if isinstance(level, str):
        level_int = logging.getLevelName(level.upper())
        if not isinstance(level_int, int):
            raise ValueError(f"Unknown log level: {level!r}")
    else:
        level_int = int(level)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=level_int,
        format=_FORMAT,
        datefmt=_DATEFMT,
        handlers=handlers,
        force=force,
    )
    return logging.getLogger()
