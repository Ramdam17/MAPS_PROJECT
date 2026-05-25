"""Per-domain experiment packages.

Each sub-package implements one of the six MAPS evaluation domains:

- ``blindsight`` — perceptual detection under noise (paper §3)
- ``agl`` — Artificial Grammar Learning (paper §4)
- ``sarl`` — single-agent RL on MinAtar (paper §5)
- ``sarl_cl`` — SARL with continual learning + distillation (paper §5.2)
- ``marl`` — multi-agent RL on MeltingPot 2.0 (paper §6)
- ``metta`` — exploratory extension, last in execution order

Each domain contains ``data.py``, ``trainer.py``, and ``cli.py`` modules.

Empty during Sprint 10 — populated in Sprint 11+ (one sprint per domain).
"""
