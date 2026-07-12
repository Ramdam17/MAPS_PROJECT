"""Tier-2 parity: SARL v1 networks bit-exact vs maps_v1.py (Sprint 14.D).

Same seed → identical init → identical forward (the port reuses core
cascade_update, numerically identical to the source's inline blend). Also pins
the 4 v1 structural properties that distinguish v1 from the v2 wrong-variant
(D14.1).
"""

from __future__ import annotations

import torch

from maps.domains.sarl.model_v1 import (
    NUM_LINEAR_UNITS,
    SarlQNetworkV1,
    SarlSecondOrderNetworkV1,
)
from maps.utils.seeding import set_all_seeds
from tests.parity.sarl import _student_extracts as ref

IN_CHANNELS = 4
NUM_ACTIONS = 6


def test_num_linear_units_is_1024():
    assert NUM_LINEAR_UNITS == 1024 == ref.num_linear_units


def test_qnetwork_v1_structural_shapes():
    """v1: Q-head reads 1024-d Output; dedicated decoder Linear(128, 1024)."""
    net = SarlQNetworkV1(IN_CHANNELS, NUM_ACTIONS)
    assert net.fc_hidden.weight.shape == (128, 1024)
    assert net.fc_output.weight.shape == (1024, 128)  # dedicated decoder (not tied)
    assert net.actions.weight.shape == (NUM_ACTIONS, 1024)  # reads Output, not Hidden


def test_second_order_v1_comparison_layer_active():
    """v1: comparison_layer is an active 1024×1024 Linear (v2 commented it out)."""
    net = SarlSecondOrderNetworkV1()
    assert net.comparison_layer.weight.shape == (1024, 1024)
    assert net.wager.weight.shape == (2, 1024)


def test_qnetwork_v1_forward_bit_exact():
    set_all_seeds(42)
    ours = SarlQNetworkV1(IN_CHANNELS, NUM_ACTIONS).eval()
    set_all_seeds(42)
    theirs = ref.QNetwork(IN_CHANNELS, NUM_ACTIONS).eval()

    x = torch.rand(8, IN_CHANNELS, 10, 10)
    q_o, h_o, c_o, out_o = ours(x, None, 1.0)
    q_t, h_t, c_t, out_t = theirs(x, None, 1.0)
    assert torch.equal(q_o, q_t)
    assert torch.equal(h_o, h_t)
    assert torch.equal(c_o, c_t)
    assert torch.equal(out_o, out_t)


def test_qnetwork_v1_cascade_bit_exact():
    """Cascade blend on Output matches the source inline expression."""
    set_all_seeds(1)
    ours = SarlQNetworkV1(IN_CHANNELS, NUM_ACTIONS).eval()
    set_all_seeds(1)
    theirs = ref.QNetwork(IN_CHANNELS, NUM_ACTIONS).eval()

    x = torch.rand(8, IN_CHANNELS, 10, 10)
    prev = torch.rand(8, NUM_LINEAR_UNITS)
    _, _, _, out_o = ours(x, prev, 0.02)
    _, _, _, out_t = theirs(x, prev, 0.02)
    assert torch.allclose(out_o, out_t, atol=1e-7)


def test_second_order_v1_forward_bit_exact():
    set_all_seeds(7)
    ours = SarlSecondOrderNetworkV1().eval()  # eval → dropout off → deterministic
    set_all_seeds(7)
    theirs = ref.SecondOrderNetwork(None).eval()

    comparison = torch.rand(8, NUM_LINEAR_UNITS)
    w_o, c_o = ours(comparison, None, 1.0)
    w_t, c_t = theirs(comparison, None, 1.0)
    assert torch.equal(w_o, w_t)
    assert torch.equal(c_o, c_t)
    assert w_o.shape == (8, 2)  # raw 2-logit wager
