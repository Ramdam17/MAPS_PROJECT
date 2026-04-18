"""Unit tests for `maps.utils.configure_logging`.

Covers:
- accepts string and int log levels
- rejects unknown level names
- attaches a file handler when `log_file` is given
- creates the log file's parent directory
- opens the file in append mode (re-run appends, doesn't truncate)
- `force=True` replaces previously installed handlers
"""

from __future__ import annotations

import logging

import pytest

from maps.utils import configure_logging


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Reset the root logger between tests so handlers don't leak."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    # Close and remove any handlers added during the test.
    for h in root.handlers[:]:
        h.close()
        root.removeHandler(h)
    # Restore originals.
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)


def test_configure_logging_accepts_string_level():
    root = configure_logging(level="DEBUG")
    assert root.level == logging.DEBUG


def test_configure_logging_accepts_int_level():
    root = configure_logging(level=logging.WARNING)
    assert root.level == logging.WARNING


def test_configure_logging_is_case_insensitive():
    root = configure_logging(level="info")
    assert root.level == logging.INFO


def test_configure_logging_rejects_unknown_level():
    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging(level="NOPE")


def test_configure_logging_writes_to_file(tmp_path):
    log_file = tmp_path / "nested" / "run.log"
    configure_logging(level="INFO", log_file=log_file)

    log = logging.getLogger("maps.test_logging_setup")
    log.info("hello from test")

    # Flush the file handler so the message lands on disk before we read.
    for h in logging.getLogger().handlers:
        h.flush()

    assert log_file.exists(), "log file should be created"
    content = log_file.read_text()
    assert "hello from test" in content
    assert "maps.test_logging_setup" in content


def test_configure_logging_appends_to_existing_file(tmp_path):
    log_file = tmp_path / "run.log"
    log_file.write_text("pre-existing line\n")

    configure_logging(level="INFO", log_file=log_file)
    logging.getLogger("maps.test_append").info("new line")
    for h in logging.getLogger().handlers:
        h.flush()

    content = log_file.read_text()
    assert "pre-existing line" in content, "append mode must not truncate"
    assert "new line" in content


def test_configure_logging_force_replaces_handlers(tmp_path):
    # First configuration: one stream handler only.
    configure_logging(level="INFO")
    n_handlers_first = len(logging.getLogger().handlers)

    # Second configuration: stream + file — must not accumulate.
    configure_logging(level="INFO", log_file=tmp_path / "run.log")
    n_handlers_second = len(logging.getLogger().handlers)

    # If `force=True` works, second call has exactly its own handlers
    # (1 stream + 1 file = 2), not first_call + second_call (1 + 2 = 3).
    assert n_handlers_first == 1
    assert n_handlers_second == 2


def test_configure_logging_format_includes_required_fields(tmp_path, caplog):
    log_file = tmp_path / "run.log"
    configure_logging(level="INFO", log_file=log_file)

    logging.getLogger("maps.test_format").info("formatted message")
    for h in logging.getLogger().handlers:
        h.flush()

    content = log_file.read_text()
    # Format: "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    assert "maps.test_format" in content
    assert "[INFO]" in content
    assert "formatted message" in content
