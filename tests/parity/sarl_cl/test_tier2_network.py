"""Tier-2 parity: SARL+CL networks bit-exact vs sarl_cl_maps.py (Sprint 15.D).

Same seed → identical init → identical forward (port reuses core cascade_update,
numerically identical to the source inline blend). Covers the CL innovation
(AdaptiveQNetwork: 1×1 adapter + channel zero-pad) and the v1 first/second-order
nets.
"""

from __future__ import annotations

import torch

from maps.domains.sarl_cl.model import (
    NUM_LINEAR_UNITS,
    AdaptiveQNetwork,
    SarlCLQNetwork,
    SarlCLSecondOrderNetwork,
)
from maps.utils.seeding import set_all_seeds
from tests.parity.sarl_cl import _student_extracts as ref

IN_CHANNELS = 4
MAX_CHANNELS = 10
NUM_ACTIONS = 6


def test_num_linear_units_is_1024():
    assert NUM_LINEAR_UNITS == 1024


def test_qnetwork_forward_bit_exact():
    set_all_seeds(42)
    ours = SarlCLQNetwork(IN_CHANNELS, NUM_ACTIONS).eval()
    set_all_seeds(42)
    theirs = ref.QNetwork(IN_CHANNELS, NUM_ACTIONS).eval()
    x = torch.rand(8, IN_CHANNELS, 10, 10)
    for o, t in zip(ours(x, None, 1.0), theirs(x, None, 1.0), strict=True):
        assert torch.equal(o, t)


def test_second_order_forward_bit_exact():
    set_all_seeds(7)
    ours = SarlCLSecondOrderNetwork().eval()
    set_all_seeds(7)
    theirs = ref.SecondOrderNetwork(None).eval()
    comparison = torch.rand(8, NUM_LINEAR_UNITS)
    w_o, c_o = ours(comparison, None, 1.0)
    w_t, c_t = theirs(comparison, None, 1.0)
    assert torch.equal(w_o, w_t) and torch.equal(c_o, c_t)


def test_adaptive_qnetwork_forward_bit_exact():
    """AdaptiveQNetwork (incl. the torch.rand init probe) matches the source."""
    set_all_seeds(1)
    ours = AdaptiveQNetwork(MAX_CHANNELS, NUM_ACTIONS).eval()
    set_all_seeds(1)
    theirs = ref.AdaptiveQNetwork(MAX_CHANNELS, NUM_ACTIONS).eval()
    x = torch.rand(8, MAX_CHANNELS, 10, 10)
    for o, t in zip(ours(x, None, 1.0), theirs(x, None, 1.0), strict=True):
        assert torch.allclose(o, t, atol=1e-6)


def test_adaptive_qnetwork_channel_zero_pad():
    """Fewer-channel input is zero-padded to max_input_channels, forward runs."""
    set_all_seeds(2)
    net = AdaptiveQNetwork(MAX_CHANNELS, NUM_ACTIONS).eval()
    x_small = torch.rand(4, IN_CHANNELS, 10, 10)  # 4 < 10 channels
    padded = net.adapt_input(x_small)
    assert padded.shape == (4, MAX_CHANNELS, 10, 10)
    assert torch.equal(padded[:, IN_CHANNELS:], torch.zeros(4, MAX_CHANNELS - IN_CHANNELS, 10, 10))
    q, _hidden, _comparison, output = net(x_small, None, 1.0)
    assert q.shape == (4, NUM_ACTIONS)
    assert output.shape == (4, NUM_LINEAR_UNITS)


def test_adaptive_qnetwork_conv_output_is_1024():
    net = AdaptiveQNetwork(MAX_CHANNELS, NUM_ACTIONS)
    assert net.actions.in_features == 1024
    assert net.fc_hidden.in_features == 1024
