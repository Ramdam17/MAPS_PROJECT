"""Cross-cutting utilities.

Will hold:

- ``config.py`` — OmegaConf YAML loader + CLI override merger
- ``logging_setup.py`` — stdlib logging configuration helper
- ``seeding.py`` — ``set_all_seeds(seed)`` (random, numpy, torch, cuda)
- ``device.py`` — NEW: auto-detect CPU / MPS (Apple Silicon) / CUDA
- ``energy_tracker.py`` — Codecarbon-style energy wrapping (may move to
  ``maps.domains.marl`` during Sprint 15 per DETTE-4)

Empty during Sprint 10 — populated in Sprint 11+.
"""
