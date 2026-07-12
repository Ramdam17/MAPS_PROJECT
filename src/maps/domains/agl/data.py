"""AGL (Artificial Grammar Learning) word generation & encoding — MAPS §4.

Implements the Dienes (1997) implicit-learning paradigm: two finite-state
grammars (A and B) over a 5-letter alphabet ``{x, v, m, t, r}`` generate
"grammatical" words; a first-order encoder/decoder learns to reconstruct them
while a second-order network wagers on whether the reconstruction was perfect.

The second-order target (:func:`target_second`) is the AGL analogue of the
Blindsight detection signal: "high wager" (1.0) iff the first-order network
reconstructed **exactly** the active input positions — so the wager learns to
predict the *success of the first-order network*, not an intrinsic property of
the stimulus.

Encoding (paper §4):
- 5 letters, each a 6-bit one-hot (the 6th bit is always 0 — legacy artifact of
  an old 6-letter alphabet; kept so the decoder's ``make_chunked_sigmoid(6)``
  treats each 6-bit chunk independently).
- Up to 8 letters × 6 bits = 48-d input.

Ported **verbatim** from ``external/paper_reference/agl_tmlr.py`` (the Vargas
code that produced paper Tables 5b/5c):
- ``Generate_Word_Random``  L268-275
- ``Generate_Grammar_A``    L278-311
- ``Generate_Grammar_B``    L315-351
- ``encode_word``           L361-383
- ``Array_Words``           L392-413
- ``target_second``         L421-441

The finite-state grammars are kept as the source's walrus-operator ladders:
the **exact order of ``random.randint(1, 2)`` draws** is load-bearing for
Tier-1 bit-exact parity (Sprint 13.G) — a cleaner transition-table rewrite
would consume the RNG in a different order and break parity.

References
----------
Vargas et al. (2025), MAPS, TMLR submission §4.
Dienes, Z. (1997). Transfer of implicit knowledge across domains.
"""

from __future__ import annotations

import logging
import random
from enum import IntEnum

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# ── Constants (paper §4 / agl_tmlr.py) ──────────────────────────────────────
ALLOWED_LETTERS = ["x", "v", "m", "t", "r"]  # Grammar A & B alphabet
BITS_PER_LETTER = 6  # 6-bit one-hot; 6th bit unused (legacy 6-letter alphabet)
MAX_LETTERS = 8  # words padded to 8 letters
NUM_INPUT_UNITS = MAX_LETTERS * BITS_PER_LETTER  # = 48

# One-hot mapping, verbatim agl_tmlr.py:363-367.
_MAPPING: dict[str, list[int]] = {
    "x": [1, 0, 0, 0, 0, 0],
    "v": [0, 1, 0, 0, 0, 0],
    "m": [0, 0, 1, 0, 0, 0],
    "t": [0, 0, 0, 1, 0, 0],
    "r": [0, 0, 0, 0, 1, 0],
}


class GrammarType(IntEnum):
    """Grammar selector — matches the student's integer ``grammar_type`` arg."""

    RANDOM = 1
    A = 2
    B = 3


# ── Word generators (verbatim ladders — do NOT refactor: RNG order parity) ──


def generate_random_word() -> str:
    """Random word over the alphabet. Verbatim ``Generate_Word_Random`` L268-275."""
    grammar_word = ""
    number_letters = random.randint(3, 8)
    allowed_letters = ["x", "v", "m", "t", "r"]  # Letters from Grammar A and B
    while len(grammar_word) < number_letters:
        current_letter = random.choice(allowed_letters)
        grammar_word += current_letter
    return grammar_word


def generate_grammar_a() -> str:
    """Grammar A word (Dienes 1997 FSM). Verbatim ``Generate_Grammar_A`` L278-311."""
    grammar_A_word = ""
    number_letters = random.randint(3, 8)
    position = 1
    while len(grammar_A_word) < number_letters:
        current_path = random.randint(1, 2)

        (grammar_A_word := grammar_A_word + "x", position := 2) if (position, current_path) == (1, 1) else (grammar_A_word := grammar_A_word + "v", position := 3) if (position, current_path) == (1, 2) else None  # fmt: skip

        current_path = random.randint(1, 2)
        if len(grammar_A_word) == number_letters:
            break
        (grammar_A_word := grammar_A_word + "m", position := 2) if (position, current_path) == (2, 1) else (grammar_A_word := grammar_A_word + "x", position := 4) if (position, current_path) == (2, 2) else None  # fmt: skip

        current_path = random.randint(1, 2)
        if len(grammar_A_word) == number_letters:
            break
        (grammar_A_word := grammar_A_word + "t", position := 3) if (position, current_path) == (3, 1) else (grammar_A_word := grammar_A_word + "v", position := 5) if (position, current_path) == (3, 2) else None  # fmt: skip

        current_path = random.randint(1, 2)
        if len(grammar_A_word) == number_letters:
            break
        (grammar_A_word := grammar_A_word + "t", position := 4) if (position, current_path) == (4, 1) else (grammar_A_word := grammar_A_word + "m", position := 6) if (position, current_path) == (4, 2) else None  # fmt: skip

        current_path = random.randint(1, 2)
        if len(grammar_A_word) == number_letters:
            break
        (grammar_A_word := grammar_A_word + "r", position := 3) if (position, current_path) == (5, 1) else (grammar_A_word := grammar_A_word + "m", position := 6) if (position, current_path) == (5, 2) else None  # fmt: skip

        if position == 6:
            break
    return grammar_A_word


def generate_grammar_b() -> str:
    """Grammar B word (Dienes 1997 FSM). Verbatim ``Generate_Grammar_B`` L315-351."""
    grammar_B_word = ""
    number_letters = random.randint(3, 8)
    position = 1
    while len(grammar_B_word) < number_letters:
        current_path = random.randint(1, 2)

        (grammar_B_word := grammar_B_word + "x", position := 2) if (position, current_path) == (1, 1) else (grammar_B_word := grammar_B_word + "v", position := 3) if (position, current_path) == (1, 2) else None  # fmt: skip

        current_path = random.randint(1, 2)
        if len(grammar_B_word) == number_letters:
            break
        (grammar_B_word := grammar_B_word + "x", position := 5) if (position, current_path) == (2, 1) else (grammar_B_word := grammar_B_word + "m", position := 3) if (position, current_path) == (2, 2) else None  # fmt: skip

        current_path = random.randint(1, 2)
        if len(grammar_B_word) == number_letters:
            break
        (grammar_B_word := grammar_B_word + "v", position := 4) if (position, current_path) == (3, 1) else (grammar_B_word := grammar_B_word + "t", position := 5) if (position, current_path) == (3, 2) else None  # fmt: skip

        current_path = random.randint(1, 2)
        if len(grammar_B_word) == number_letters:
            break
        (grammar_B_word := grammar_B_word + "t", position := 4) if (position, current_path) == (4, 1) else (grammar_B_word := grammar_B_word + "r", position := 2) if (position, current_path) == (4, 2) else None  # fmt: skip

        current_path = random.randint(1, 2)
        if len(grammar_B_word) == number_letters:
            break

        if position == 5:
            if current_path == 1:
                grammar_B_word += "r"
            elif len(grammar_B_word) > 2:
                break

    return grammar_B_word


# ── Encoding ────────────────────────────────────────────────────────────────


def encode_word(word: str) -> list[int]:
    """Encode a word into a 48-d 6-bit-chunk vector. Verbatim ``encode_word`` L361-383."""
    mapping = _MAPPING

    # Initialize the output array with zeros
    encoded = [0] * 48  # 48 elements, all zeros

    # Encode each letter and place it in the output array
    for i, letter in enumerate(word):
        start_index = i * BITS_PER_LETTER
        end_index = start_index + BITS_PER_LETTER

        # Ensure we don't exceed the 48 elements limit
        if end_index > 48:
            break

        encoded[start_index:end_index] = mapping.get(letter, [0] * BITS_PER_LETTER)

    return encoded


def array_words(grammar_type: GrammarType | int, number: int, *, device: str = "cpu") -> Tensor:
    """Build a ``(number, 48)`` tensor of encoded words. Verbatim ``Array_Words`` L392-413.

    ``grammar_type``: 1 = random, 2 = Grammar A, 3 = Grammar B.
    """
    list_words: list[list[int]] = []
    while len(list_words) < number:
        if grammar_type == 1:
            generated = generate_random_word()
        if grammar_type == 2:
            generated = generate_grammar_a()
        if grammar_type == 3:
            generated = generate_grammar_b()
        generated_encoded = encode_word(generated)

        list_words.append(generated_encoded)

    return torch.Tensor(list_words).to(device)


# ── Second-order (wager) target ──────────────────────────────────────────────


def target_second(input: Tensor, output: Tensor, *, device: str = "cpu") -> Tensor:
    """High/low wager target per row. Verbatim ``target_second`` L421-441.

    ``1.0`` iff the top-k output positions (k = number of active input bits)
    exactly match the active input positions — i.e. the first-order network
    reconstructed the word perfectly.
    """
    if input.shape != output.shape:
        raise ValueError("Input and output must have the same shape")

    num_rows, _ = input.shape
    result = torch.zeros(num_rows)

    for i in range(num_rows):
        # Count the number of 1s in the input row
        input_indexes = (input[i] == 1).nonzero(as_tuple=True)[0]
        num_ones = input_indexes.size(0)

        # Get the indexes of the top x values in the output row
        _, output_indexes = torch.topk(output[i], num_ones)

        # Compare and set the result
        if set(input_indexes.tolist()) == set(output_indexes.tolist()):
            result[i] = 1.0

    return torch.Tensor(result).to(device)
