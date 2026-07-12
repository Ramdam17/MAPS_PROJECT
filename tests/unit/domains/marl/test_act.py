"""Unit tests for MARL ACTLayer (Discrete) — Sprint 16.F."""

from __future__ import annotations

import torch
from gymnasium import spaces

from maps.domains.marl.act import ACTLayer
from maps.utils.seeding import set_all_seeds

INPUTS = 32
N_ACT = 6


def _layer():
    return ACTLayer(spaces.Discrete(N_ACT), INPUTS, use_orthogonal=True, gain=0.01)


def test_forward_returns_valid_actions():
    set_all_seeds(42)
    layer = _layer()
    x = torch.randn(8, INPUTS)
    actions, log_probs = layer(x)
    assert actions.shape == (8, 1)
    assert (actions >= 0).all() and (actions < N_ACT).all()
    assert torch.isfinite(log_probs).all()


def test_deterministic_is_argmax():
    set_all_seeds(1)
    layer = _layer()
    x = torch.randn(4, INPUTS)
    actions, _ = layer(x, deterministic=True)
    probs = layer.get_probs(x)
    assert torch.equal(actions.squeeze(-1), probs.argmax(dim=-1))


def test_available_actions_masking():
    set_all_seeds(2)
    layer = _layer()
    x = torch.randn(16, INPUTS)
    avail = torch.ones(16, N_ACT)
    avail[:, 3:] = 0  # only actions 0,1,2 available
    actions, _ = layer(x, available_actions=avail, deterministic=True)
    assert (actions < 3).all()


def test_evaluate_actions_shapes():
    set_all_seeds(3)
    layer = _layer()
    x = torch.randn(8, INPUTS)
    action = torch.randint(0, N_ACT, (8, 1))
    log_probs, entropy = layer.evaluate_actions(x, action)
    assert log_probs.shape == (8, 1)
    assert entropy.dim() == 0  # scalar mean entropy


def test_non_discrete_raises():
    import pytest

    with pytest.raises(NotImplementedError, match="Discrete"):
        ACTLayer(spaces.Box(low=-1, high=1, shape=(2,)), INPUTS, True, 0.01)
