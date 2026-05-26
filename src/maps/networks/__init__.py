"""Shared network architectures used across domains.

- :class:`FirstOrderMLP` — domain-agnostic MLP encoder/decoder shared
  by Blindsight (Sprint 12) and AGL (Sprint 13).
- :func:`global_sigmoid`, :func:`make_chunked_sigmoid` — decoder
  activation helpers.

Domain-specific networks (Q-networks for SARL, policy/value networks
for MARL) live under :mod:`maps.domains.<domain>`.
"""

from maps.networks.first_order_mlp import (
    FirstOrderMLP,
    global_sigmoid,
    make_chunked_sigmoid,
)

__all__ = [
    "FirstOrderMLP",
    "global_sigmoid",
    "make_chunked_sigmoid",
]
