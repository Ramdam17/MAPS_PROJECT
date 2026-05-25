"""Deterministic seeding across all stochastic libraries used in MAPS.

Seed control is mandatory (CLAUDE.md: "at every entry point"). Calling
``set_all_seeds(seed)`` touches every RNG a MAPS run may consume:

- Python ``random``
- NumPy **legacy global RNG** (``np.random.seed``) — covers all
  ``np.random.normal/randint/...`` calls used in the paper code.
- PyTorch CPU + CUDA (if available) + MPS (if available)
- ``PYTHONHASHSEED`` env var (affects dict iteration in some code paths)

Known gap
---------
``np.random.default_rng()`` (the new PCG64-based Generator API) is **not**
seeded by ``np.random.seed`` — it holds its own internal state. No code in
the current port uses ``default_rng()``; if future work does, that Generator
must be seeded independently (pass ``seed`` to its constructor).

Call this once at the top of every script/notebook — *before* importing
data, building models, or constructing datasets.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

__all__ = ["set_all_seeds"]

_log = logging.getLogger(__name__)


def set_all_seeds(seed: int, *, deterministic_cudnn: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU/CUDA/MPS).

    Parameters
    ----------
    seed : int
        The integer seed to broadcast. MAPS lab convention: 42.
    deterministic_cudnn : bool, default True
        If True and CUDA is available, force cuDNN into deterministic mode.
        Costs some throughput but removes a major source of run-to-run
        variance (cuDNN auto-tuning picks different conv algorithms).

    Notes
    -----
    PyTorch's ``torch.use_deterministic_algorithms(True)`` is *not* enabled
    here because it raises on legitimate non-deterministic ops used inside
    some RL code (e.g., scatter_add on CUDA). Re-enable it in
    domain-specific entry points if full determinism is required.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # MPS (Apple Silicon) inherits from the global torch seed above; no separate API.
    _log.debug("set_all_seeds: seed=%d, deterministic_cudnn=%s", seed, deterministic_cudnn)
