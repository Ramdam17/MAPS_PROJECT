"""Unit tests for :mod:`maps.utils.logging_setup`."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from maps.utils.logging_setup import (
    DEFAULT_FORMAT,
    configure_logging,
)


def test_default_level_is_info() -> None:
    logger = configure_logging()
    assert logger.name == "maps"
    assert logging.root.level == logging.INFO


def test_explicit_level_string() -> None:
    configure_logging(level="DEBUG")
    assert logging.root.level == logging.DEBUG


def test_explicit_level_int() -> None:
    configure_logging(level=logging.WARNING)
    assert logging.root.level == logging.WARNING


def test_unknown_level_raises() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging(level="NOT_A_LEVEL")


def test_log_file_handler_writes(tmp_path: Path) -> None:
    """File handler appends to the requested path; parent dirs created."""
    log_path = tmp_path / "nested" / "subdir" / "test.log"
    configure_logging(level="INFO", log_file=log_path)
    logger = logging.getLogger("maps.test.handler")
    logger.info("hello from test")
    # Flush handlers so the file is actually written
    for h in logging.root.handlers:
        h.flush()

    assert log_path.exists()
    contents = log_path.read_text()
    assert "hello from test" in contents
    assert "maps.test.handler" in contents
    assert "INFO" in contents


def test_force_true_is_idempotent() -> None:
    """Calling configure_logging twice with force=True doesn't crash and
    leaves a clean handler set."""
    configure_logging(level="INFO")
    handlers_first = list(logging.root.handlers)
    configure_logging(level="DEBUG")
    # After force=True, the handler list is replaced (not appended to).
    assert len(logging.root.handlers) <= len(handlers_first) + 1


def test_format_matches_default() -> None:
    """The configured format string includes asctime, name, levelname,
    message."""
    assert "%(asctime)s" in DEFAULT_FORMAT
    assert "%(name)s" in DEFAULT_FORMAT
    assert "%(levelname)s" in DEFAULT_FORMAT
    assert "%(message)s" in DEFAULT_FORMAT
