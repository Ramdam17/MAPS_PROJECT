"""Unit tests for SARL+CL dynamic loss weighting (Sprint 15.E)."""

from __future__ import annotations

import torch

from maps.domains.sarl_cl.loss_weighting import DynamicLossWeighter, LossMixingWeights


def test_mixing_weights_faithful_default():
    w = LossMixingWeights()
    assert (w.task, w.distillation, w.feature) == (0.4, 0.4, 0.2)


def test_update_tracks_historical_max():
    dw = DynamicLossWeighter()
    dw.update(
        {"task": torch.tensor(2.0), "distillation": torch.tensor(1.0), "feature": torch.tensor(0.5)}
    )
    dw.update(
        {"task": torch.tensor(1.0), "distillation": torch.tensor(3.0), "feature": torch.tensor(0.1)}
    )
    assert dw.historical_max["task"] == 2.0
    assert dw.historical_max["distillation"] == 3.0
    assert dw.historical_max["feature"] == 0.5


def test_weight_losses_normalizes_by_historical_max():
    dw = DynamicLossWeighter()
    dw.update(
        {"task": torch.tensor(4.0), "distillation": torch.tensor(2.0), "feature": torch.tensor(1.0)}
    )
    weighted = dw.weight_losses(
        {"task": torch.tensor(4.0), "distillation": torch.tensor(1.0), "feature": torch.tensor(0.5)}
    )
    # value / (historical_max + eps): 4/4≈1, 1/2≈0.5, 0.5/1≈0.5
    assert torch.allclose(weighted["task"], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(weighted["distillation"], torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(weighted["feature"], torch.tensor(0.5), atol=1e-6)


def test_weight_losses_keeps_gradients():
    dw = DynamicLossWeighter()
    dw.update(
        {"task": torch.tensor(2.0), "distillation": torch.tensor(2.0), "feature": torch.tensor(2.0)}
    )
    live = torch.tensor(1.0, requires_grad=True)
    weighted = dw.weight_losses({"task": live, "distillation": live, "feature": live})
    # normalisation is by a detached scalar → gradient still flows
    weighted["task"].backward()
    assert live.grad is not None
