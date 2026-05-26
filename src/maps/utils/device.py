"""Device auto-detection — CPU vs MPS vs CUDA.

Apple-Silicon-friendly: priority is CUDA > MPS > CPU when ``"auto"``
is requested. Explicit preferences fall back gracefully (with a
warning) when the requested backend is unavailable.

D12.2 — Sprint 12 (new module, per structure-decision.md).
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)

DevicePref = Literal["auto", "cpu", "mps", "cuda"]


def pick_best_available() -> torch.device:
    """Return the best torch device available on this host.

    Priority : **CUDA** > **MPS** > **CPU**. This matches the typical
    cluster (Compute Canada → CUDA) and dev (Apple Silicon → MPS)
    scenarios.

    Returns
    -------
    torch.device
        ``cuda``, ``mps``, or ``cpu``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device(prefer: DevicePref = "auto") -> torch.device:
    """Return the requested torch device, falling back to CPU if needed.

    Parameters
    ----------
    prefer : ``"auto"`` | ``"cpu"`` | ``"mps"`` | ``"cuda"``, optional
        - ``"auto"`` (default) : call :func:`pick_best_available`
        - ``"cpu"`` : always CPU
        - ``"mps"`` : MPS if available, else CPU + warning
        - ``"cuda"`` : CUDA if available, else CPU + warning

    Returns
    -------
    torch.device

    Raises
    ------
    ValueError
        If ``prefer`` is none of the four accepted values.
    """
    if prefer == "auto":
        device = pick_best_available()
    elif prefer == "cpu":
        device = torch.device("cpu")
    elif prefer == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            logger.warning("MPS requested but not available; falling back to CPU")
            device = torch.device("cpu")
    elif prefer == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("CUDA requested but not available; falling back to CPU")
            device = torch.device("cpu")
    else:
        raise ValueError(
            f"unknown device preference: {prefer!r} (expected one of 'auto', 'cpu', 'mps', 'cuda')"
        )

    logger.info("Using device: %s", device)
    return device
