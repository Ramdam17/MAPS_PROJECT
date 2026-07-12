"""SecondOrderNetwork (the confidence judge) — fix (a) keeps this and attaches it inside the
acting R_Actor; the paper's ghost R_Actor_Meta/R_Critic_Meta/RNNLayer_Meta are removed."""

from __future__ import annotations

import torch

from maps.domains.marl.policy_meta import SecondOrderNetwork
from maps.utils.seeding import set_all_seeds

HIDDEN = 64
B = 8


def test_second_order_forward_shapes():
    set_all_seeds(42)
    so = SecondOrderNetwork(HIDDEN).eval()
    comparison = torch.randn(B, HIDDEN)
    wager, comparison_out = so(comparison, None, 1.0)
    assert wager.shape == (B, 2)  # two raw wager logits (BCE-with-logits target)
    assert comparison_out.shape == (B, HIDDEN)  # threads the graded cascade
    assert torch.isfinite(wager).all()


def test_second_order_cascade_threads_prev():
    set_all_seeds(1)
    so = SecondOrderNetwork(HIDDEN).eval()
    comparison = torch.randn(B, HIDDEN)
    _w0, out_none = so(comparison, None, 0.5)
    prev = torch.randn(B, HIDDEN)
    _w1, out_blend = so(comparison, prev, 0.5)
    assert not torch.allclose(out_none, out_blend)  # prev blends into the cascade


def test_ghost_meta_nets_removed():
    """Fix (a): the separate meta actor/critic ghost networks no longer exist."""
    import maps.domains.marl.policy_meta as pm

    assert not hasattr(pm, "R_Actor_Meta")
    assert not hasattr(pm, "R_Critic_Meta")
    assert not hasattr(pm, "RNNLayer_Meta")
