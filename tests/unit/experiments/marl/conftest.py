"""Skip MARL unit tests when meltingpot/dmlab2d deps are unavailable.

MARL ports require gymnasium + meltingpot + dmlab2d, which are Linux-only
per CLAUDE.md. On macOS local dev these are not installed, so importing
``maps.experiments.marl`` raises ``ModuleNotFoundError`` at collection
time. ``pytest.importorskip`` at conftest level causes pytest to skip
the entire directory cleanly, matching the ``linux_only`` marker
registered in ``pyproject.toml``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gymnasium", reason="MARL tests require gymnasium (Linux-only)")
