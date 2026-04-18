"""Pattern generation for the Artificial Grammar Learning (AGL) task.

Ported from AGL/AGL_TMLR.py (`Generate_Word_Random`, `Generate_Grammar_A`,
`Generate_Grammar_B`, `encode_word`, `Array_Words`, `target_second`,
lines 268-441). The numerical behavior must stay bit-identical to the
reference — parity with the reference pre_train is asserted in
tests/parity/test_agl_pretrain.py.

Grammar design follows Dienes (1997), "Transfer of implicit knowledge
across domains: How implicit and how abstract?", §5 — a 5-letter alphabet
{x, v, m, t, r} walked through a small finite-state machine.

Pre-training uses **random** words (grammar_type=1). Grammar A/B words are
used for the downstream classification phase (not ported here yet).

Each letter is encoded as a 6-bit one-hot vector with the sixth bit unused
(always 0), producing a 48-d input for an 8-letter maximum word. Shorter
words are zero-padded on the right.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch

__all__ = [
    "BITS_PER_LETTER",
    "NUM_INPUT_UNITS",
    "GrammarType",
    "TrainingBatch",
    "encode_word",
    "generate_batch",
    "generate_grammar_a",
    "generate_grammar_b",
    "generate_random_word",
    "target_second",
]

# Paper constants (Dienes 1997 §5 + AGL_TMLR.py L1676, L361-383).
BITS_PER_LETTER: int = 6
MAX_WORD_LENGTH: int = 8
NUM_INPUT_UNITS: int = BITS_PER_LETTER * MAX_WORD_LENGTH  # 48

# Reference uses a literal mapping: {x, v, m, t, r} → one-hot over 6 bits.
# The 6th bit is never set — kept for legacy compatibility with a version that
# once had a 6-letter alphabet. We preserve the 6-bit chunk so existing code
# keeps working.
_LETTER_TO_ONEHOT: dict[str, tuple[int, int, int, int, int, int]] = {
    "x": (1, 0, 0, 0, 0, 0),
    "v": (0, 1, 0, 0, 0, 0),
    "m": (0, 0, 1, 0, 0, 0),
    "t": (0, 0, 0, 1, 0, 0),
    "r": (0, 0, 0, 0, 1, 0),
}
_ALLOWED_LETTERS: tuple[str, ...] = tuple(_LETTER_TO_ONEHOT.keys())


class GrammarType(IntEnum):
    """Which word generator to use in `generate_batch`.

    Integer values match the reference code's `grammar_type` argument, so
    either the enum or the raw int works.
    """

    RANDOM = 1  # free 5-letter mix, used for pre-training
    A = 2  # Dienes (1997) grammar A
    B = 3  # Dienes (1997) grammar B (control / distractor)


@dataclass
class TrainingBatch:
    """One epoch's worth of AGL training patterns.

    Attributes
    ----------
    patterns : Tensor, shape (N, 48)
        Encoded word batch (1.0 / 0.0 floats).
    """

    patterns: torch.Tensor


def generate_random_word() -> str:
    """Port of `Generate_Word_Random` (AGL_TMLR.py L268-275).

    Picks a word length uniformly in [3, 8] (inclusive) and samples each
    letter independently from the 5-letter alphabet.
    """
    number_letters = random.randint(3, 8)
    return "".join(random.choice(_ALLOWED_LETTERS) for _ in range(number_letters))


def generate_grammar_a() -> str:
    """Port of `Generate_Grammar_A` (AGL_TMLR.py L278-311).

    Traverses Dienes (1997) Grammar A FSM. The reference implementation is
    an obfuscated walrus-ladder; we keep the exact state-transition table
    so the output distribution stays bit-identical. Nodes 1-5 are live;
    node 6 is terminal.
    """
    word = ""
    number_letters = random.randint(3, 8)
    position = 1

    while len(word) < number_letters:
        current_path = random.randint(1, 2)
        # Node 1: emit x→node2 (path 1) or v→node3 (path 2).
        if (position, current_path) == (1, 1):
            word += "x"
            position = 2
        elif (position, current_path) == (1, 2):
            word += "v"
            position = 3

        current_path = random.randint(1, 2)
        if len(word) == number_letters:
            break
        # Node 2: emit m→node2 (path 1) or x→node4 (path 2).
        if (position, current_path) == (2, 1):
            word += "m"
            position = 2
        elif (position, current_path) == (2, 2):
            word += "x"
            position = 4

        current_path = random.randint(1, 2)
        if len(word) == number_letters:
            break
        # Node 3: emit t→node3 (path 1) or v→node5 (path 2).
        if (position, current_path) == (3, 1):
            word += "t"
            position = 3
        elif (position, current_path) == (3, 2):
            word += "v"
            position = 5

        current_path = random.randint(1, 2)
        if len(word) == number_letters:
            break
        # Node 4: emit t→node4 (path 1) or m→node6 (path 2, terminal).
        if (position, current_path) == (4, 1):
            word += "t"
            position = 4
        elif (position, current_path) == (4, 2):
            word += "m"
            position = 6

        current_path = random.randint(1, 2)
        if len(word) == number_letters:
            break
        # Node 5: emit r→node3 (path 1) or m→node6 (path 2, terminal).
        if (position, current_path) == (5, 1):
            word += "r"
            position = 3
        elif (position, current_path) == (5, 2):
            word += "m"
            position = 6

        if position == 6:
            break
    return word


def generate_grammar_b() -> str:
    """Port of `Generate_Grammar_B` (AGL_TMLR.py L315-351).

    Grammar B is the control / distractor FSM in Dienes (1997). Structure is
    symmetric to Grammar A but with different transitions; node 5 has a
    special "emit r or stop" branch at the tail.
    """
    word = ""
    number_letters = random.randint(3, 8)
    position = 1

    while len(word) < number_letters:
        current_path = random.randint(1, 2)
        # Node 1: emit x→node2 or v→node3.
        if (position, current_path) == (1, 1):
            word += "x"
            position = 2
        elif (position, current_path) == (1, 2):
            word += "v"
            position = 3

        current_path = random.randint(1, 2)
        if len(word) == number_letters:
            break
        # Node 2: emit x→node5 or m→node3.
        if (position, current_path) == (2, 1):
            word += "x"
            position = 5
        elif (position, current_path) == (2, 2):
            word += "m"
            position = 3

        current_path = random.randint(1, 2)
        if len(word) == number_letters:
            break
        # Node 3: emit v→node4 or t→node5.
        if (position, current_path) == (3, 1):
            word += "v"
            position = 4
        elif (position, current_path) == (3, 2):
            word += "t"
            position = 5

        current_path = random.randint(1, 2)
        if len(word) == number_letters:
            break
        # Node 4: emit t→node4 or r→node2.
        if (position, current_path) == (4, 1):
            word += "t"
            position = 4
        elif (position, current_path) == (4, 2):
            word += "r"
            position = 2

        current_path = random.randint(1, 2)
        if len(word) == number_letters:
            break

        if position == 5:
            if current_path == 1:
                word += "r"
            elif len(word) > 2:
                break
    return word


def encode_word(word: str) -> list[int]:
    """Port of `encode_word` (AGL_TMLR.py L361-383).

    Returns a 48-element list where each 6-bit chunk is the one-hot code of
    the corresponding letter. Unused positions are zero. An unknown letter
    encodes as all zeros in its chunk (reference behavior — `mapping.get(..., [0]*6)`).
    """
    encoded: list[int] = [0] * NUM_INPUT_UNITS
    for i, letter in enumerate(word):
        start = i * BITS_PER_LETTER
        end = start + BITS_PER_LETTER
        if end > NUM_INPUT_UNITS:
            break
        encoded[start:end] = list(_LETTER_TO_ONEHOT.get(letter, (0,) * BITS_PER_LETTER))
    return encoded


def generate_batch(
    *,
    grammar_type: GrammarType | int,
    number: int,
    device: torch.device | str = "cpu",
) -> TrainingBatch:
    """Generate a batch of encoded words.

    Port of `Array_Words(grammar_type, number, output=False)` (L392-413).
    The `output` flag in the reference only controlled debug prints — we
    drop it.

    Parameters
    ----------
    grammar_type : GrammarType | int
        1 = random (used for pre-training), 2 = Grammar A, 3 = Grammar B.
    number : int
        Number of words in the batch.
    device : torch.device | str
        Where to place the output tensor.

    Returns
    -------
    TrainingBatch
        `patterns` is a (number, 48) float tensor.
    """
    gt = int(grammar_type)
    generators = {
        int(GrammarType.RANDOM): generate_random_word,
        int(GrammarType.A): generate_grammar_a,
        int(GrammarType.B): generate_grammar_b,
    }
    if gt not in generators:
        raise ValueError(f"Unknown grammar_type={grammar_type!r}; expected 1, 2, or 3.")
    generate = generators[gt]

    rows: list[list[int]] = [encode_word(generate()) for _ in range(number)]
    patterns = torch.tensor(np.asarray(rows), dtype=torch.float32, device=device)
    return TrainingBatch(patterns=patterns)


def target_second(input_: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Port of `target_second` (AGL_TMLR.py L421-441).

    For each row, build a binary wager target:

    - Count the number of 1s in the input row (call it *k*).
    - Take the top-*k* positions of the output row.
    - Target is 1.0 if the set of top-*k* positions exactly matches the
      set of 1-positions, else 0.0.

    This is the AGL analog of the Blindsight detection target: a high-wager
    trial is one where the first-order network reconstructed the exact
    active positions.

    Parameters
    ----------
    input_ : Tensor, shape (N, D)
        Ground-truth one-hot encoding.
    output : Tensor, shape (N, D)
        First-order network reconstruction.

    Returns
    -------
    Tensor, shape (N,)
        Float32 wager targets in {0.0, 1.0}, on the same device as `input_`.
    """
    if input_.shape != output.shape:
        raise ValueError(f"shape mismatch: input={input_.shape}, output={output.shape}")

    num_rows = input_.shape[0]
    result = torch.zeros(num_rows, device=input_.device)
    for i in range(num_rows):
        input_indices = (input_[i] == 1).nonzero(as_tuple=True)[0]
        k = int(input_indices.size(0))
        if k == 0:
            # Degenerate row (all zeros); reference `topk(_, 0)` returns
            # an empty tensor; set() equality of two empty sets is True.
            result[i] = 1.0
            continue
        _, output_indices = torch.topk(output[i], k)
        if set(input_indices.tolist()) == set(output_indices.tolist()):
            result[i] = 1.0
    return result
