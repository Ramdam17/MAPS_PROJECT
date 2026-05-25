"""Tier-1 parity tests for the **v1** SARL model port.

Asserts that ``maps.experiments.sarl.model_v1`` reproduces the v1 reference
(``tests/parity/sarl/_reference_sarl_v1.py``, extracted from
``external/paper_reference/sarl/maps_v1.py``) bit-for-bit at ``atol=1e-6``
under:

1. Zero-cascade pass (``prev=None``, ``cascade_rate=1.0``) — pure
   feed-forward with no accumulation.
2. Single-step cascade (``prev`` seeded, ``cascade_rate=0.02``) — tests
   the ``cascade_update`` arithmetic.
3. Multi-iteration cascade rollout (50 iterations at α=0.02) — tests
   the iteration-level accumulation matches.
4. Reproduction residual ``Input - Output`` (1024-dim) — tests that v1's
   dedicated decoder reproduction is bit-identical.

This is the v1 counterpart of ``test_tier1_forward.py`` (which targets v2).
Both must stay green simultaneously after Sprint-09 — when the config
toggle ``training.sarl_model_variant`` selects which port is wired in.

Shared spec: in_channels=4 (Breakout default), num_actions=6, batch=32.
Dropout is disabled via ``.eval()`` to make the comparison deterministic
— dropout parity belongs in Tier 2/3.

References
----------
- ``docs/reproduction/sarl-v1-vs-v2.md`` (v1 vs v2 structural diff).
- ``docs/sprints/sprint-09-sarl-v1-port-and-ruff-sweep.md`` (Phase 9.1).
- ``docs/reproduction/deviations.md`` (``D-sarl-wrong-variant``).
"""

from __future__ import annotations

import pytest
import torch

from maps.experiments.sarl.model_v1 import SarlQNetworkV1, SarlSecondOrderNetworkV1
from maps.utils.seeding import set_all_seeds
from tests.parity.sarl._reference_sarl_v1 import (
    QNetwork as RefQNetworkV1,
)
from tests.parity.sarl._reference_sarl_v1 import (
    SecondOrderNetwork as RefSecondOrderNetworkV1,
)

# ─── Shared fixtures ──────────────────────────────────────────────────────────

IN_CHANNELS = 4
NUM_ACTIONS = 6
BATCH = 32
STATE_SHAPE = (BATCH, IN_CHANNELS, 10, 10)
COMPARISON_SHAPE = (BATCH, 1024)  # NUM_LINEAR_UNITS — v1's SecondOrder input
CASCADE_RATE_PAPER = 0.02
N_ITER_PAPER = 50
ATOL = 1e-6


@pytest.fixture
def paired_q_networks() -> tuple[RefQNetworkV1, SarlQNetworkV1]:
    """Reference + ported v1 Q-networks with identical weights.

    Cross-loads the reference state_dict into the port so layer-order
    differences (if any) don't masquerade as forward-pass divergence.
    """
    set_all_seeds(42)
    ref = RefQNetworkV1(IN_CHANNELS, NUM_ACTIONS).eval()
    set_all_seeds(42)
    ours = SarlQNetworkV1(IN_CHANNELS, NUM_ACTIONS).eval()
    # Sanity: identical state_dict by construction (same layer order, same seed).
    # Cross-load as a belt-and-braces guarantee.
    ours.load_state_dict(ref.state_dict())
    return ref, ours


@pytest.fixture
def paired_second_order_networks() -> tuple[RefSecondOrderNetworkV1, SarlSecondOrderNetworkV1]:
    """Reference + ported v1 SecondOrder networks with identical weights."""
    set_all_seeds(42)
    ref = RefSecondOrderNetworkV1(IN_CHANNELS).eval()
    set_all_seeds(42)
    ours = SarlSecondOrderNetworkV1(IN_CHANNELS).eval()
    # SecondOrderNetwork v1 has a Softmax module that has no parameters and is
    # not present in our port (we don't use it in forward). load_state_dict
    # is happy because softmax contributes nothing to state_dict.
    ours.load_state_dict(ref.state_dict())
    return ref, ours


# ─── QNetwork v1 forward-pass parity ─────────────────────────────────────────


def test_qnet_v1_zero_cascade(paired_q_networks):
    """Pure feed-forward path (prev_h2=None, cascade_rate=1.0) is bit-exact."""
    ref, ours = paired_q_networks
    torch.manual_seed(123)
    x = torch.randn(STATE_SHAPE)

    with torch.no_grad():
        ref_out = ref(x, None, 1.0)
        our_out = ours(x, None, 1.0)

    # 4-tuple: (q_values, hidden, comparison, output)
    for ref_t, our_t, name in zip(
        ref_out, our_out, ["q_values", "hidden", "comparison", "output"], strict=True
    ):
        assert torch.allclose(ref_t, our_t, atol=ATOL), (
            f"v1 QNetwork divergence on '{name}' at zero-cascade: "
            f"max diff = {(ref_t - our_t).abs().max().item():.2e}"
        )


def test_qnet_v1_single_step_cascade(paired_q_networks):
    """Single cascade iteration with seeded prev_h2 is bit-exact.

    In v1, ``prev_h2`` is a previous 1024-dim Output (NOT 128-dim Hidden) —
    cascade is applied to Output, not Hidden. The test feeds a seeded 1024-dim
    tensor as ``prev_h2``.
    """
    ref, ours = paired_q_networks
    torch.manual_seed(456)
    x = torch.randn(STATE_SHAPE)
    # v1 cascade input is 1024-dim (Output), unlike v2 which is 128-dim (Hidden).
    prev_output = torch.randn(BATCH, 1024)

    with torch.no_grad():
        ref_out = ref(x, prev_output, CASCADE_RATE_PAPER)
        our_out = ours(x, prev_output, CASCADE_RATE_PAPER)

    for ref_t, our_t, name in zip(
        ref_out, our_out, ["q_values", "hidden", "comparison", "output"], strict=True
    ):
        assert torch.allclose(ref_t, our_t, atol=ATOL), (
            f"v1 QNetwork divergence on '{name}' at single-step cascade: "
            f"max diff = {(ref_t - our_t).abs().max().item():.2e}"
        )


def test_qnet_v1_50_iter_cascade(paired_q_networks):
    """50-iteration cascade rollout at α=0.02 (paper §2.1 eq.6) is bit-exact.

    Mirrors the production cascade loop: feed previous Output back as
    ``prev_h2`` for each iteration. With no dropout in the forward path,
    this is mathematically a no-op (each iter produces the same Output),
    but we still verify the arithmetic matches.
    """
    ref, ours = paired_q_networks
    torch.manual_seed(789)
    x = torch.randn(STATE_SHAPE)

    with torch.no_grad():
        ref_prev = None
        our_prev = None
        for _ in range(N_ITER_PAPER):
            ref_out = ref(x, ref_prev, CASCADE_RATE_PAPER)
            our_out = ours(x, our_prev, CASCADE_RATE_PAPER)
            # 4th slot is Output in v1 (the cascade-integrated 1024-dim tensor).
            ref_prev = ref_out[3]
            our_prev = our_out[3]

    for ref_t, our_t, name in zip(
        ref_out, our_out, ["q_values", "hidden", "comparison", "output"], strict=True
    ):
        assert torch.allclose(ref_t, our_t, atol=ATOL), (
            f"v1 QNetwork divergence on '{name}' after {N_ITER_PAPER} cascade iters: "
            f"max diff = {(ref_t - our_t).abs().max().item():.2e}"
        )


def test_qnet_v1_comparison_shape_is_1024(paired_q_networks):
    """v1 reproduction residual is 1024-dim (Input - Output), not 128-dim.

    Distinguishes v1 from v2 at the output level: v2's comparison is also
    1024-dim (Input - reconstruction), but v2 computes reconstruction via
    tied weights. v1 uses a dedicated decoder. This test pins the shape
    so any accidental refactor that drops to 128-dim is caught.
    """
    _, ours = paired_q_networks
    x = torch.randn(STATE_SHAPE)
    with torch.no_grad():
        _, _, comparison, output = ours(x, None, 1.0)
    assert comparison.shape == (BATCH, 1024), f"expected (B, 1024), got {comparison.shape}"
    assert output.shape == (BATCH, 1024), f"expected (B, 1024), got {output.shape}"


# ─── SecondOrderNetwork v1 forward-pass parity ───────────────────────────────


def test_so_v1_zero_cascade(paired_second_order_networks):
    """v1 SecondOrder: linear+ReLU+dropout pipeline matches reference, no cascade."""
    ref, ours = paired_second_order_networks
    torch.manual_seed(321)
    comparison_matrix = torch.randn(COMPARISON_SHAPE)

    with torch.no_grad():
        ref_wager, ref_out = ref(comparison_matrix, None, 1.0)
        our_wager, our_out = ours(comparison_matrix, None, 1.0)

    assert torch.allclose(ref_wager, our_wager, atol=ATOL), (
        f"v1 SecondOrder wager divergence: max diff = {(ref_wager - our_wager).abs().max():.2e}"
    )
    assert torch.allclose(ref_out, our_out, atol=ATOL), (
        f"v1 SecondOrder comparison_out divergence: "
        f"max diff = {(ref_out - our_out).abs().max():.2e}"
    )


def test_so_v1_single_step_cascade(paired_second_order_networks):
    """Single cascade iteration on comparison_out at α=0.02 is bit-exact."""
    ref, ours = paired_second_order_networks
    torch.manual_seed(654)
    comparison_matrix = torch.randn(COMPARISON_SHAPE)
    prev = torch.randn(BATCH, 1024)

    with torch.no_grad():
        ref_wager, ref_out = ref(comparison_matrix, prev, CASCADE_RATE_PAPER)
        our_wager, our_out = ours(comparison_matrix, prev, CASCADE_RATE_PAPER)

    assert torch.allclose(ref_wager, our_wager, atol=ATOL)
    assert torch.allclose(ref_out, our_out, atol=ATOL)


def test_so_v1_50_iter_cascade(paired_second_order_networks):
    """50-iteration cascade rollout: each iter feeds prev_comparison forward.

    Unlike v1 QNetwork (which is deterministic = cascade no-op), v1
    SecondOrder uses dropout, so cascade DOES average across iterations
    in train mode. We test in ``.eval()`` mode here, so dropout is off
    and the rollout is deterministic — same shape of test as QNet.
    """
    ref, ours = paired_second_order_networks
    torch.manual_seed(987)
    comparison_matrix = torch.randn(COMPARISON_SHAPE)

    with torch.no_grad():
        ref_prev = None
        our_prev = None
        for _ in range(N_ITER_PAPER):
            ref_wager, ref_out = ref(comparison_matrix, ref_prev, CASCADE_RATE_PAPER)
            our_wager, our_out = ours(comparison_matrix, our_prev, CASCADE_RATE_PAPER)
            ref_prev = ref_out
            our_prev = our_out

    assert torch.allclose(ref_wager, our_wager, atol=ATOL), (
        f"v1 SecondOrder wager divergence after {N_ITER_PAPER} iters: "
        f"max diff = {(ref_wager - our_wager).abs().max():.2e}"
    )
    assert torch.allclose(ref_out, our_out, atol=ATOL), (
        f"v1 SecondOrder comparison_out divergence after {N_ITER_PAPER} iters: "
        f"max diff = {(ref_out - our_out).abs().max():.2e}"
    )


# ─── State-dict shape parity (catches accidental refactors) ──────────────────


def test_qnet_v1_state_dict_has_dedicated_decoder():
    """v1 must have ``fc_output`` (dedicated decoder) — distinguishes from v2.

    v2 has fc_hidden + actions + b_recon (tied-weight reconstruction).
    v1 has fc_hidden + fc_output + actions (dedicated decoder, no b_recon).
    """
    net = SarlQNetworkV1(IN_CHANNELS, NUM_ACTIONS)
    keys = set(net.state_dict().keys())
    expected_v1 = {
        "conv.weight",
        "conv.bias",
        "fc_hidden.weight",
        "fc_hidden.bias",
        "fc_output.weight",  # <-- dedicated decoder, unique to v1
        "fc_output.bias",
        "actions.weight",
        "actions.bias",
    }
    assert keys == expected_v1, (
        f"v1 QNetwork state_dict mismatch.\n"
        f"  Expected: {sorted(expected_v1)}\n"
        f"  Actual:   {sorted(keys)}\n"
        f"  Missing:  {sorted(expected_v1 - keys)}\n"
        f"  Extra:    {sorted(keys - expected_v1)}"
    )


def test_so_v1_state_dict_has_comparison_layer():
    """v1 SecondOrder must have ``comparison_layer`` — distinguishes from v2.

    v2 has only ``wager`` (comparison_layer commented out).
    v1 has ``comparison_layer`` (1024×1024 active) + ``wager``.
    """
    net = SarlSecondOrderNetworkV1(IN_CHANNELS)
    keys = set(net.state_dict().keys())
    expected_v1 = {
        "comparison_layer.weight",  # <-- active, unique to v1
        "comparison_layer.bias",
        "wager.weight",
        "wager.bias",
    }
    assert keys == expected_v1, (
        f"v1 SecondOrderNetwork state_dict mismatch.\n"
        f"  Expected: {sorted(expected_v1)}\n"
        f"  Actual:   {sorted(keys)}\n"
        f"  Missing:  {sorted(expected_v1 - keys)}\n"
        f"  Extra:    {sorted(keys - expected_v1)}"
    )


def test_qnet_v1_param_count_matches_paper():
    """v1 should have ~131K more params than v2 (the dedicated decoder).

    Paper Table 11 doesn't explicitly state per-network param counts, but
    the v1 decoder is ``Linear(128, 1024)`` = 128*1024 + 1024 = 132,096 params.
    v2 reuses fc_hidden.weight.t() (0 new) + b_recon (1024 new) = 1024 params.
    Difference : 132,096 - 1,024 = 131,072 = paper's b_recon → fc_output gap.
    """
    net = SarlQNetworkV1(IN_CHANNELS, NUM_ACTIONS)
    fc_output_params = net.fc_output.weight.numel() + net.fc_output.bias.numel()
    assert fc_output_params == 128 * 1024 + 1024, (
        f"fc_output should have 128*1024 + 1024 = 132,096 params, got {fc_output_params}"
    )
