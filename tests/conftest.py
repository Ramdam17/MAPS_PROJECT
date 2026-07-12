"""Project-wide pytest configuration.

Single autouse fixture that seeds every randomness source before each
test. Lab convention (``seed=42``) matches ``config/maps.yaml`` and the
default of :func:`maps.utils.seeding.set_all_seeds`.

Any test that needs non-default behaviour can either re-call
``set_all_seeds`` explicitly with a different seed, or override this
fixture in a nested ``conftest.py``.
"""

from __future__ import annotations

import pytest
import torch

from maps.utils.seeding import LAB_DEFAULT_SEED, set_all_seeds

# Cap torch to a single thread during tests. On shared HPC nodes the default
# (all cores) causes severe thread-contention slowdowns on the small tensor ops
# used here — a MinAtar CL stage went from ~3s to >15min. Single-thread is also
# more deterministic. Does not affect numerical parity (same config both sides).
torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def seed_everything() -> int:
    """Seed all RNGs before every test and return the seed used.

    Autouse: applies to every test in the suite. Tests that want to
    inspect the seed (e.g. to log it in their assertion message) can
    accept ``seed_everything`` as an explicit fixture argument.
    """
    set_all_seeds(LAB_DEFAULT_SEED)
    return LAB_DEFAULT_SEED
