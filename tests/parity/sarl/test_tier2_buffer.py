"""Tier 2 parity tests — SARL replay buffer sampling equivalence.

Asserts that ``maps.experiments.sarl.data.SarlReplayBuffer`` reproduces the
paper reference (``tests/parity/sarl/_reference_sarl.replay_buffer``)
transition-for-transition under identical seeding.

Why this matters: the replay buffer determines which transitions the learner
sees and in what order. If our sampling diverges from the paper's — even by
one index — the training trajectory diverges, and all downstream z-scores
become incomparable to the reported numbers.

Test strategy
-------------
1. **Identical insertion → identical storage.** Feed the same stream of
   synthetic transitions into both buffers; assert the internal lists
   compare equal after each batch and after wrap-around.
2. **Seeded sampling → identical draws.** Seed Python's ``random`` once,
   then call ``.sample(B)`` on both buffers N times from the same post-seed
   state; assert each draw's transition identities match.
3. **Wrap-around preserves cyclic overwrite.** Fill past capacity, verify
   the overwrite index advances cyclically and the two buffers agree on
   which transitions are now evicted.
4. **Sampling without replacement.** Paper-faithful guardrail: ensure the
   returned batch has no duplicate transitions (``random.sample`` semantics,
   not ``random.choices``).

Notes on RNG coupling
---------------------
Both buffers share Python's global ``random`` state (``random.sample`` reads
it). So to compare them fairly, we MUST reseed between draws — if we draw
from ref first, the global state advances and the second draw from ours
starts from a different position. The tests interleave ``random.seed`` calls
carefully.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

import pytest
import torch

from maps.experiments.sarl.data import SarlReplayBuffer, Transition, get_state
from tests.parity.sarl._reference_sarl import (
    replay_buffer as RefReplayBuffer,
)
from tests.parity.sarl._reference_sarl import (
    transition as ref_transition,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

BUFFER_SIZE = 1000
BATCH_SIZE = 128
N_DRAWS = 100  # number of independent sampling calls to compare
IN_CHANNELS = 4
STATE_SHAPE = (1, IN_CHANNELS, 10, 10)


def _make_transition(i: int) -> tuple[torch.Tensor, torch.Tensor, int, float, bool]:
    """Produce a reproducible synthetic transition tagged by index i.

    Using ``i`` in the tensor values gives us a cheap equality check: two
    transitions are "the same transition" iff their tensors match element-wise.
    """
    state = torch.full(STATE_SHAPE, float(i))
    next_state = torch.full(STATE_SHAPE, float(i) + 0.5)
    action = i % 6  # 6 is max MinAtar action space
    reward = float(i) * 0.01
    is_terminal = bool(i % 50 == 0)  # terminal every 50 steps
    return state, next_state, action, reward, is_terminal


def _insert_stream(
    buf_ref: RefReplayBuffer, buf_ours: SarlReplayBuffer, indices: Iterable[int]
) -> None:
    """Insert the same transition stream into both buffers."""
    for i in indices:
        args = _make_transition(i)
        buf_ref.add(*args)
        buf_ours.add(*args)


def _transitions_equal(ref_t: Any, our_t: Transition) -> bool:
    """Compare a reference namedtuple with ours. The name differs
    (``transition`` vs ``Transition``) but field values must match."""
    return (
        torch.equal(ref_t.state, our_t.state)
        and torch.equal(ref_t.next_state, our_t.next_state)
        and ref_t.action == our_t.action
        and ref_t.reward == our_t.reward
        and ref_t.is_terminal == our_t.is_terminal
    )


# ─── Storage-level parity ────────────────────────────────────────────────────


def test_insertion_below_capacity_produces_matching_buffers() -> None:
    """Below capacity: both buffers grow identically, location advances alike."""
    ref = RefReplayBuffer(BUFFER_SIZE)
    ours = SarlReplayBuffer(BUFFER_SIZE)
    _insert_stream(ref, ours, range(500))

    assert len(ref.buffer) == len(ours.buffer) == 500
    assert ref.location == ours.location == 500
    for r, o in zip(ref.buffer, ours.buffer, strict=True):
        assert _transitions_equal(r, o)


def test_wrap_around_overwrites_oldest_slot() -> None:
    """At capacity + 1 insertion, slot 0 must be overwritten in both buffers."""
    ref = RefReplayBuffer(BUFFER_SIZE)
    ours = SarlReplayBuffer(BUFFER_SIZE)
    # Fill to exactly capacity
    _insert_stream(ref, ours, range(BUFFER_SIZE))
    assert ref.location == ours.location == 0  # wrapped back to 0
    assert len(ref.buffer) == len(ours.buffer) == BUFFER_SIZE

    # One more insertion should overwrite slot 0, not append
    _insert_stream(ref, ours, [BUFFER_SIZE])
    assert len(ref.buffer) == len(ours.buffer) == BUFFER_SIZE
    assert ref.location == ours.location == 1
    # The old transition (index 0) must be gone from both
    assert _transitions_equal(ref.buffer[0], ours.buffer[0])
    # And it should be the new one (index == BUFFER_SIZE)
    expected = Transition(*_make_transition(BUFFER_SIZE))
    assert _transitions_equal(expected, ours.buffer[0])


def test_cyclic_overwrite_over_multiple_wraps() -> None:
    """After 3× capacity insertions, only the last BUFFER_SIZE transitions
    survive, and both buffers agree on which ones."""
    ref = RefReplayBuffer(BUFFER_SIZE)
    ours = SarlReplayBuffer(BUFFER_SIZE)
    _insert_stream(ref, ours, range(3 * BUFFER_SIZE))

    assert len(ref.buffer) == len(ours.buffer) == BUFFER_SIZE
    assert ref.location == ours.location == 0
    for r, o in zip(ref.buffer, ours.buffer, strict=True):
        assert _transitions_equal(r, o)


# ─── Sampling-level parity ───────────────────────────────────────────────────


def _draw_both_same_seed(
    ref: RefReplayBuffer, ours: SarlReplayBuffer, seed: int, batch_size: int
) -> tuple[list[Any], list[Transition]]:
    """Draw one batch from each buffer, reseeding between draws.

    Both use ``random.sample`` from Python's global RNG. Calling it once
    advances the state. To give both a fair chance at producing the same
    batch, we reseed immediately before each call.
    """
    random.seed(seed)
    ref_batch = ref.sample(batch_size)
    random.seed(seed)
    our_batch = ours.sample(batch_size)
    return ref_batch, our_batch


def test_single_draw_matches_at_fixed_seed() -> None:
    """One seeded draw: ref and ours return identical transitions in order."""
    ref = RefReplayBuffer(BUFFER_SIZE)
    ours = SarlReplayBuffer(BUFFER_SIZE)
    _insert_stream(ref, ours, range(BUFFER_SIZE))

    ref_batch, our_batch = _draw_both_same_seed(ref, ours, seed=12345, batch_size=BATCH_SIZE)
    assert len(ref_batch) == len(our_batch) == BATCH_SIZE
    for r, o in zip(ref_batch, our_batch, strict=True):
        assert _transitions_equal(r, o)


def test_many_draws_match_at_different_seeds() -> None:
    """N independent draws at distinct seeds — every draw must match."""
    ref = RefReplayBuffer(BUFFER_SIZE)
    ours = SarlReplayBuffer(BUFFER_SIZE)
    _insert_stream(ref, ours, range(BUFFER_SIZE))

    for k in range(N_DRAWS):
        seed = 1000 + k
        ref_batch, our_batch = _draw_both_same_seed(ref, ours, seed, BATCH_SIZE)
        for j, (r, o) in enumerate(zip(ref_batch, our_batch, strict=True)):
            assert _transitions_equal(r, o), (
                f"draw {k} at seed {seed}, index {j}: ref vs ours differ"
            )


def test_sampling_is_without_replacement() -> None:
    """Guardrail: ``random.sample`` semantics (no dupes). If someone swaps to
    ``random.choices`` (with replacement), this test fires."""
    ref = RefReplayBuffer(BUFFER_SIZE)
    ours = SarlReplayBuffer(BUFFER_SIZE)
    _insert_stream(ref, ours, range(BUFFER_SIZE))

    random.seed(42)
    batch = ours.sample(BATCH_SIZE)
    action_indices = [int(t.state[0, 0, 0, 0].item()) for t in batch]
    assert len(set(action_indices)) == BATCH_SIZE, "duplicates detected in sample"


def test_sequential_draws_consume_rng_identically() -> None:
    """Two back-to-back draws without reseeding: both buffers consume the
    RNG stream at the same rate, so their *second* draw also matches."""
    ref = RefReplayBuffer(BUFFER_SIZE)
    ours = SarlReplayBuffer(BUFFER_SIZE)
    _insert_stream(ref, ours, range(BUFFER_SIZE))

    random.seed(99)
    ref_1 = ref.sample(BATCH_SIZE)
    ref_2 = ref.sample(BATCH_SIZE)

    random.seed(99)
    our_1 = ours.sample(BATCH_SIZE)
    our_2 = ours.sample(BATCH_SIZE)

    for r, o in zip(ref_1, our_1, strict=True):
        assert _transitions_equal(r, o), "draw #1 diverged"
    for r, o in zip(ref_2, our_2, strict=True):
        assert _transitions_equal(r, o), "draw #2 diverged"


# ─── Transition-layout parity ────────────────────────────────────────────────


def test_transition_field_order_matches_reference() -> None:
    """Guardrail: if someone reorders Transition fields, unpacking order in
    the trainer silently flips state↔next_state or action↔reward. Catch it."""
    assert Transition._fields == ref_transition._fields, (
        f"Transition field order diverged: ours={Transition._fields} ref={ref_transition._fields}"
    )


def test_get_state_shape_and_permutation() -> None:
    """``get_state`` contracts: input (10, 10, C) numpy → output (1, C, 10, 10)
    float tensor with dims permuted via (2, 0, 1)."""
    import numpy as np

    # Paper uses HWC-layout numpy arrays from MinAtar's env.state()
    np_state = np.arange(4 * 10 * 10, dtype=np.float32).reshape(10, 10, 4)
    out = get_state(np_state, device="cpu")

    assert out.shape == (1, 4, 10, 10)
    assert out.dtype == torch.float32
    # Verify permutation: out[0, c, h, w] == np_state[h, w, c]
    for c in range(4):
        for h in range(10):
            for w in range(10):
                assert out[0, c, h, w].item() == np_state[h, w, c]


@pytest.mark.parametrize("buffer_size", [10, 100, 10_000])
def test_len_matches_buffer_contents(buffer_size: int) -> None:
    """``len(buffer)`` must reflect filled slots, not ``buffer_size``."""
    ours = SarlReplayBuffer(buffer_size)
    _insert_stream(RefReplayBuffer(buffer_size), ours, range(buffer_size // 2))
    assert len(ours) == buffer_size // 2
