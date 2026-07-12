"""Tier-1 parity: AGL data generation bit-exact vs paper_reference.

Sprint 13.G / D13.4. Seeds Python's ``random`` identically for the student
extract and our port, then asserts the generated words / encodings / tensors /
wager targets are identical. The generators consume ``random.randint`` in a
specific order (the FSM walrus ladders); any divergence in RNG consumption
would surface here.
"""

from __future__ import annotations

import random

import torch

from maps.domains.agl import data
from tests.parity.agl import _student_extracts as ref


def _both(seed, fn_ours, fn_ref, n):
    """Run n draws of each generator under an identical seed; return (ours, ref)."""
    random.seed(seed)
    ours = [fn_ours() for _ in range(n)]
    random.seed(seed)
    theirs = [fn_ref() for _ in range(n)]
    return ours, theirs


def test_generate_random_word_bit_exact():
    ours, theirs = _both(42, data.generate_random_word, ref.Generate_Word_Random, 200)
    assert ours == theirs


def test_generate_grammar_a_bit_exact():
    ours, theirs = _both(42, data.generate_grammar_a, ref.Generate_Grammar_A, 200)
    assert ours == theirs


def test_generate_grammar_b_bit_exact():
    ours, theirs = _both(7, data.generate_grammar_b, ref.Generate_Grammar_B, 200)
    assert ours == theirs


def test_encode_word_matches_reference():
    # Cover normal words, max length, unknown letter (defensive all-zeros), empty.
    for word in ["xvmtr", "xxxxxxxx", "xvmtrxvm", "zzz", "", "xvmtrxvmtr"]:
        assert data.encode_word(word) == ref.encode_word(word)


def test_encode_word_shape_and_chunks():
    enc = data.encode_word("xv")
    assert len(enc) == data.NUM_INPUT_UNITS == 48
    # First letter 'x' -> chunk 0, second 'v' -> chunk 1; rest zeros.
    assert enc[0:6] == [1, 0, 0, 0, 0, 0]
    assert enc[6:12] == [0, 1, 0, 0, 0, 0]
    assert enc[12:] == [0] * 36


def test_array_words_bit_exact_all_grammars():
    for gtype in (1, 2, 3):
        random.seed(123)
        ours = data.array_words(gtype, 64)
        random.seed(123)
        theirs = ref.Array_Words(gtype, 64)
        assert torch.equal(ours, theirs), f"grammar_type={gtype} tensor mismatch"


def test_target_second_bit_exact():
    torch.manual_seed(0)
    # Binary-ish inputs (0/1) and arbitrary float outputs, several rows.
    inp = (torch.rand(32, 48) > 0.6).float()
    out = torch.rand(32, 48)
    ours = data.target_second(inp, out)
    theirs = ref.target_second(inp, out)
    assert torch.equal(ours, theirs)


def test_target_second_perfect_reconstruction_gives_one():
    # If output ranks the active input positions highest, wager == 1.
    inp = torch.zeros(1, 48)
    inp[0, [0, 6, 12]] = 1.0
    out = torch.zeros(1, 48)
    out[0, [0, 6, 12]] = 10.0  # top-3 == active positions
    assert data.target_second(inp, out)[0].item() == 1.0


def test_target_second_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError, match="same shape"):
        data.target_second(torch.zeros(2, 48), torch.zeros(2, 47))
