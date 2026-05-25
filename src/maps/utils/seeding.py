"""Random seeding utilities for reproducible MAPS runs.

Single canonical entry point — :func:`set_all_seeds` — that touches every
randomness source the MAPS pipeline can hit: Python ``random``, NumPy,
PyTorch (CPU + CUDA + MPS), and Python hash randomization.

Lab convention (see ``config/maps.yaml``) : ``seed=42``.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

LAB_DEFAULT_SEED: int = 42


def set_all_seeds(seed: int) -> None:
    """Seed every randomness source MAPS uses.

    Sets the seed for Python's :mod:`random`, NumPy
    (:func:`numpy.random.seed`), PyTorch CPU + CUDA + MPS, and Python's
    hash randomization via ``PYTHONHASHSEED``. This is the canonical
    entry point — call it **once** per training run, **before** any
    data loading, model construction, or trainer setup.

    Parameters
    ----------
    seed : int
        Non-negative integer seed. The lab convention is ``42``
        (:data:`LAB_DEFAULT_SEED`), reflected in ``config/maps.yaml``.

    Raises
    ------
    ValueError
        If ``seed`` is negative. Zero is allowed.

    Notes
    -----
    **``PYTHONHASHSEED`` caveat.** This env var only fully takes effect
    if set *before* the Python interpreter starts. Setting it here is
    a best-effort fallback (covers hashing inside this process from now
    on). For strict reproducibility of dict/set iteration order across
    runs, also set it in your shell:

    .. code-block:: bash

        PYTHONHASHSEED=42 uv run python -m maps.domains.blindsight.cli

    **CUDA / MPS determinism is *not* enforced.** Calling
    ``torch.use_deterministic_algorithms(True)`` would crash on common
    ops (e.g. :func:`torch.nn.functional.interpolate`). Callers that
    need strict device-side determinism should opt in explicitly,
    aware of the trade-off.

    **MPS** (Apple Silicon) seeding is best-effort — the MPS RNG is
    less mature than CUDA's; some ops still pull from a non-seeded
    source. Document any observed non-determinism in
    ``docs/reproduction/deviations.md``.

    Examples
    --------
    >>> from maps.utils.seeding import set_all_seeds, LAB_DEFAULT_SEED
    >>> set_all_seeds(LAB_DEFAULT_SEED)  # at every entry point
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    logger.info("All seeds set to %d (random, numpy, torch, cuda, mps)", seed)
