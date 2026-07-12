"""Verbatim extracts from ``external/paper_reference/agl_tmlr.py``.

These are the reference implementations that produced the paper Table 5b/5c
numbers. The Tier-1 parity test compares our port (Sprint 13,
``maps.domains.agl.data``) against these extracts bit-by-bit under an
identical RNG seed.

Each extract carries its source line range. Only cosmetic deltas are applied
(module-level ``bits_per_letter``/``device`` turned into locals/params); the
maths and — critically — the exact order of ``random.randint`` draws are
byte-identical to the source.
"""

# ruff: noqa  (verbatim student extract — do not lint/refactor)
from __future__ import annotations

import random

import torch

# agl_tmlr.py L1676 (module global in the source).
bits_per_letter = 6


# ── Verbatim Generate_Word_Random (agl_tmlr.py:268-275) ─────────────────────
def Generate_Word_Random():
    grammar_word = ""
    number_letters = random.randint(3, 8)
    allowed_letters = ["x", "v", "m", "t", "r"]  # Letters from Grammar A and B
    while len(grammar_word) < number_letters:
        current_letter = random.choice(allowed_letters)
        grammar_word += current_letter
    return grammar_word


# ── Verbatim Generate_Grammar_A (agl_tmlr.py:278-311) ───────────────────────
def Generate_Grammar_A():
    grammar_A_word = ""
    number_letters = random.randint(3, 8)
    position = 1
    i = 0
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


# ── Verbatim Generate_Grammar_B (agl_tmlr.py:315-351) ───────────────────────
def Generate_Grammar_B():
    grammar_B_word = ""
    number_letters = random.randint(3, 8)
    position = 1
    i = 0
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


# ── Verbatim encode_word (agl_tmlr.py:361-383) ──────────────────────────────
def encode_word(word):
    mapping = {
        "x": [1, 0, 0, 0, 0, 0],
        "v": [0, 1, 0, 0, 0, 0],
        "m": [0, 0, 1, 0, 0, 0],
        "t": [0, 0, 0, 1, 0, 0],
        "r": [0, 0, 0, 0, 1, 0],
    }
    encoded = [0] * 48
    for i, letter in enumerate(word):
        start_index = i * bits_per_letter
        end_index = start_index + bits_per_letter
        if end_index > 48:
            break
        encoded[start_index:end_index] = mapping.get(letter, [0] * bits_per_letter)
    return encoded


# ── Verbatim Array_Words (agl_tmlr.py:392-413) ──────────────────────────────
def Array_Words(grammar_type, number, device="cpu", output=False):
    list_words = []
    while len(list_words) < number:
        if grammar_type == 1:
            generated = Generate_Word_Random()
        if grammar_type == 2:
            generated = Generate_Grammar_A()
        if grammar_type == 3:
            generated = Generate_Grammar_B()
        generated_encoded = encode_word(generated)
        list_words.append(generated_encoded)
    list_words = torch.Tensor(list_words).to(device)
    return list_words


# ── Verbatim target_second (agl_tmlr.py:421-441) ────────────────────────────
def target_second(input, output, device="cpu"):
    if input.shape != output.shape:
        raise ValueError("Input and output must have the same shape")
    num_rows, num_cols = input.shape
    result = torch.zeros(num_rows)
    for i in range(num_rows):
        input_indexes = (input[i] == 1).nonzero(as_tuple=True)[0]
        num_ones = input_indexes.size(0)
        _, output_indexes = torch.topk(output[i], num_ones)
        if set(input_indexes.tolist()) == set(output_indexes.tolist()):
            result[i] = 1.0
    wager = torch.Tensor(result).to(device)
    return wager
