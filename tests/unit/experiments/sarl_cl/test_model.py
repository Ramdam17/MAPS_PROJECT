"""Unit tests for the SARL+CL network architectures.

Covers the three networks from ``src/maps/experiments/sarl_cl/model.py``:
``SarlCLQNetwork``, ``SarlCLSecondOrderNetwork``, ``AdaptiveQNetwork``. The
architectural deltas against standard SARL are LOAD-BEARING (cascade
applied to Output not Hidden; explicit ``fc_output`` decoder; explicit
``comparison_layer`` before the wager head; variable-channel padding),
so these tests pin them precisely.
"""

from __future__ import annotations

import pytest
import torch

from maps.experiments.sarl_cl.model import (
    NUM_LINEAR_UNITS,
    AdaptiveQNetwork,
    SarlCLQNetwork,
    SarlCLSecondOrderNetwork,
)

# ── SarlCLQNetwork ──────────────────────────────────────────────────────────


def test_sarlcl_q_network_forward_shapes():
    """Forward returns (q_values, hidden, comparison, output) with paper shapes."""
    net = SarlCLQNetwork(in_channels=4, num_actions=6)
    x = torch.randn(2, 4, 10, 10)
    q, hidden, comparison, output = net(x, prev_h2=None, cascade_rate=1.0)
    assert q.shape == (2, 6)
    assert hidden.shape == (2, 128)
    # Cascade runs on the 1024-dim output, and comparison = flat_input - output.
    assert comparison.shape == (2, NUM_LINEAR_UNITS)
    assert output.shape == (2, NUM_LINEAR_UNITS)


def test_sarlcl_q_network_actions_consume_output_not_hidden():
    """`self.actions` takes the 1024-dim output (NOT the 128-dim hidden)."""
    net = SarlCLQNetwork(in_channels=4, num_actions=6)
    # The standard SARL variant has actions: 128 → num_actions. The CL variant
    # must be 1024 → num_actions — if this assertion ever flips we've silently
    # regressed to the standard SARL topology.
    assert net.actions.in_features == NUM_LINEAR_UNITS
    assert net.fc_output.in_features == 128
    assert net.fc_output.out_features == NUM_LINEAR_UNITS


def test_sarlcl_q_network_cascade_is_applied_to_output():
    """Cascade blends ``fc_output(hidden)`` with ``prev_h2`` (NOT hidden with prev_h2)."""
    net = SarlCLQNetwork(in_channels=4, num_actions=3)
    net.eval()
    x = torch.randn(2, 4, 10, 10)
    prev = torch.full((2, NUM_LINEAR_UNITS), 7.0)
    alpha = 0.1
    # cascade_update(new, prev, rate) = rate * new + (1 - rate) * prev.
    _, hidden, comparison, output = net(x, prev_h2=prev, cascade_rate=alpha)

    # The un-cascaded output would be relu(fc_output(hidden)). The cascade step
    # should produce exactly α·that + (1−α)·prev.
    expected_new_output = torch.nn.functional.relu(net.fc_output(hidden))
    expected_cascaded = alpha * expected_new_output + (1.0 - alpha) * prev
    assert torch.allclose(output, expected_cascaded, atol=1e-5)

    # And the comparison residual = flat_input - cascaded_output.
    conv_out = torch.nn.functional.relu(net.conv(x))
    flat_input = conv_out.view(conv_out.size(0), -1)
    assert torch.allclose(comparison, flat_input - output, atol=1e-5)


def test_sarlcl_q_network_cascade_rate_1_disables_integration():
    """With ``cascade_rate=1``, prev_h2 is ignored (pure feed-forward)."""
    net = SarlCLQNetwork(in_channels=4, num_actions=3)
    net.eval()
    x = torch.randn(2, 4, 10, 10)
    prev = torch.full((2, NUM_LINEAR_UNITS), 99.0)
    _, hidden, _, output = net(x, prev_h2=prev, cascade_rate=1.0)
    expected = torch.nn.functional.relu(net.fc_output(hidden))
    assert torch.allclose(output, expected, atol=1e-5)


def test_sarlcl_q_network_gradients_flow():
    net = SarlCLQNetwork(in_channels=4, num_actions=3)
    x = torch.randn(2, 4, 10, 10, requires_grad=False)
    q, _, _, _ = net(x, prev_h2=None, cascade_rate=1.0)
    q.sum().backward()
    # Every parameter receives a non-None gradient tensor.
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None for g in grads)


def test_sarlcl_q_network_decoder_is_not_tied():
    """``fc_output`` is an independent Linear, NOT ``fc_hidden.weight.T``."""
    net = SarlCLQNetwork(in_channels=4, num_actions=3)
    # Paper line 143: fc_output = nn.Linear(128, 1024). Standard SARL uses
    # tied weights — this network does not.
    assert net.fc_output.weight.shape == (NUM_LINEAR_UNITS, 128)
    # fc_hidden.weight.T has the same shape, but should be a different tensor.
    assert not torch.equal(net.fc_output.weight, net.fc_hidden.weight.T)


# ── SarlCLSecondOrderNetwork ────────────────────────────────────────────────


def test_sarlcl_second_order_forward_shapes():
    net = SarlCLSecondOrderNetwork(in_channels=4)
    comparison = torch.randn(2, NUM_LINEAR_UNITS)
    wager, comp_out = net(comparison, prev_comparison=None, cascade_rate=1.0)
    assert wager.shape == (2, 2)
    assert comp_out.shape == (2, NUM_LINEAR_UNITS)


def test_sarlcl_second_order_returns_raw_logits():
    """Wager output is NOT softmaxed, despite the paper defining softmax/sigmoid attrs."""
    net = SarlCLSecondOrderNetwork(in_channels=4)
    net.eval()  # drop dropout stochasticity
    comparison = torch.randn(2, NUM_LINEAR_UNITS)
    wager, _ = net(comparison, prev_comparison=None, cascade_rate=1.0)
    # A softmax output would have rows summing to 1; raw logits do not.
    row_sums = wager.softmax(dim=-1).sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(2), atol=1e-5)  # softmaxed *would* sum to 1
    # but the raw wager itself should not (except by vanishing chance) —
    # we test that it's not close to a probability simplex.
    assert not torch.allclose(wager.sum(dim=-1), torch.ones(2), atol=1e-3)


def test_sarlcl_second_order_init_ranges():
    """comparison_layer weights in [-1, 1], wager weights in [0, 0.1]."""
    net = SarlCLSecondOrderNetwork(in_channels=4)
    w_cmp = net.comparison_layer.weight
    w_wag = net.wager.weight
    assert w_cmp.min() >= -1.0 and w_cmp.max() <= 1.0
    assert w_wag.min() >= 0.0 and w_wag.max() <= 0.1


def test_sarlcl_second_order_cascade_blends_comparison_output():
    """Cascade blends ``dropout(relu(comparison_layer(x)))`` with ``prev_comparison``."""
    net = SarlCLSecondOrderNetwork(in_channels=4)
    net.eval()  # deterministic (no dropout)
    comparison = torch.randn(2, NUM_LINEAR_UNITS)
    prev = torch.full((2, NUM_LINEAR_UNITS), 3.0)
    alpha = 0.1

    # Reproduce the pre-cascade activation manually.
    pre_cascade = torch.nn.functional.relu(net.comparison_layer(comparison))

    _, comp_out = net(comparison, prev_comparison=prev, cascade_rate=alpha)
    expected = alpha * pre_cascade + (1.0 - alpha) * prev
    assert torch.allclose(comp_out, expected, atol=1e-5)


def test_sarlcl_second_order_cascade_rate_1_disables_integration():
    net = SarlCLSecondOrderNetwork(in_channels=4)
    net.eval()
    comparison = torch.randn(2, NUM_LINEAR_UNITS)
    prev = torch.full((2, NUM_LINEAR_UNITS), 42.0)
    _, comp_out = net(comparison, prev_comparison=prev, cascade_rate=1.0)
    expected = torch.nn.functional.relu(net.comparison_layer(comparison))
    assert torch.allclose(comp_out, expected, atol=1e-5)


def test_sarlcl_second_order_unused_sigmoid_softmax_present():
    """Paper-parity attributes exist but are never invoked in forward."""
    net = SarlCLSecondOrderNetwork(in_channels=4)
    assert hasattr(net, "softmax") and hasattr(net, "sigmoid")


# ── AdaptiveQNetwork ────────────────────────────────────────────────────────


def test_adaptive_q_network_forward_with_max_channels():
    """Input with the maximum channel count works without padding."""
    net = AdaptiveQNetwork(max_input_channels=7, num_actions=6)
    x = torch.randn(2, 7, 10, 10)
    q, hidden, comparison, output = net(x, prev_h2=None, cascade_rate=1.0)
    assert q.shape == (2, 6)
    assert hidden.shape == (2, 128)
    assert comparison.shape == (2, NUM_LINEAR_UNITS)
    assert output.shape == (2, NUM_LINEAR_UNITS)


def test_adaptive_q_network_zero_pads_fewer_channels():
    """Input with fewer channels is zero-padded on the right; forward still runs."""
    net = AdaptiveQNetwork(max_input_channels=7, num_actions=6)
    x = torch.randn(2, 4, 10, 10)  # Breakout-style 4-channel
    padded = net.adapt_input(x)
    assert padded.shape == (2, 7, 10, 10)
    # First 4 channels preserved, last 3 zero.
    assert torch.allclose(padded[:, :4], x)
    assert torch.allclose(padded[:, 4:], torch.zeros(2, 3, 10, 10))
    # Full forward also works.
    q, _, _, _ = net(x, prev_h2=None, cascade_rate=1.0)
    assert q.shape == (2, 6)


def test_adaptive_q_network_adapt_input_passthrough_when_equal():
    """When channels already match, adapt_input is a no-op."""
    net = AdaptiveQNetwork(max_input_channels=4, num_actions=3)
    x = torch.randn(2, 4, 10, 10)
    assert torch.allclose(net.adapt_input(x), x)


def test_adaptive_q_network_has_input_adapter():
    """1×1 input-adapter conv is prepended (paper lines 200-203)."""
    net = AdaptiveQNetwork(max_input_channels=7, num_actions=6)
    first = next(iter(net.input_adapter.children()))
    assert isinstance(first, torch.nn.Conv2d)
    assert first.kernel_size == (1, 1)
    assert first.in_channels == 7 and first.out_channels == 7


def test_adaptive_q_network_gradients_flow_through_adapter():
    """Gradients reach the input-adapter, not just the main conv."""
    net = AdaptiveQNetwork(max_input_channels=7, num_actions=3)
    x = torch.randn(2, 4, 10, 10)
    q, _, _, _ = net(x, prev_h2=None, cascade_rate=1.0)
    q.sum().backward()
    adapter_conv = next(iter(net.input_adapter.children()))
    assert adapter_conv.weight.grad is not None
    assert adapter_conv.weight.grad.abs().sum() > 0


# ── Construction-order / edge cases ─────────────────────────────────────────


@pytest.mark.parametrize("in_channels,num_actions", [(4, 3), (6, 6), (7, 3)])
def test_sarlcl_q_network_various_games(in_channels: int, num_actions: int):
    """Works for representative MinAtar games (Breakout=4/3, SpaceInvaders=6/6, Freeway=7/3)."""
    net = SarlCLQNetwork(in_channels=in_channels, num_actions=num_actions)
    x = torch.randn(2, in_channels, 10, 10)
    q, _, _, _ = net(x, prev_h2=None, cascade_rate=1.0)
    assert q.shape == (2, num_actions)
