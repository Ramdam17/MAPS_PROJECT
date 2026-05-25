"""AGL (Artificial Grammar Learning) experiment package.

Exports
-------
- `AGLSetting`, `AGLTrainer` — 2×2 factorial training harness
- `AGLNetworkPool`, `NetworkCell` — 20-copy replication (D.28.c)
- `generate_batch`, `generate_random_word`, `generate_grammar_a`, `generate_grammar_b`
- `encode_word`, `target_second`
- `GrammarType`, `TrainingBatch`, `BITS_PER_LETTER`, `NUM_INPUT_UNITS`
"""

from __future__ import annotations

from maps.experiments.agl.data import (
    BITS_PER_LETTER,
    NUM_INPUT_UNITS,
    GrammarType,
    TrainingBatch,
    encode_word,
    generate_batch,
    generate_grammar_a,
    generate_grammar_b,
    generate_random_word,
    target_second,
)
from maps.experiments.agl.pool import AGLNetworkPool, NetworkCell
from maps.experiments.agl.trainer import AGLSetting, AGLTrainer

__all__ = [
    "BITS_PER_LETTER",
    "NUM_INPUT_UNITS",
    "AGLNetworkPool",
    "AGLSetting",
    "AGLTrainer",
    "GrammarType",
    "NetworkCell",
    "TrainingBatch",
    "encode_word",
    "generate_batch",
    "generate_grammar_a",
    "generate_grammar_b",
    "generate_random_word",
    "target_second",
]
