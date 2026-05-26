"""Tier 1+2 parity: :class:`maps.networks.FirstOrderMLP` forward AND
backward bit-exact vs verbatim student
:class:`StudentFirstOrderNetwork`.

When both modules are constructed with the same RNG state and given
the same input, their forward outputs and backward gradients must
match within float32 noise (1e-5 atol over 50 cascade steps).
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from maps.networks.first_order_mlp import FirstOrderMLP, global_sigmoid
from tests.parity.blindsight._student_extracts import StudentFirstOrderNetwork


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _make_paired_mlps(
    hidden: int = 40,
) -> tuple[FirstOrderMLP, StudentFirstOrderNetwork]:
    """Build both modules with identical init by re-seeding before each
    construction. Same constructor sequence (Linear(100, hidden) then
    Linear(hidden, 100), both bias=False, uniform(-1, 1) init)."""
    _seed_all(0)
    ours = FirstOrderMLP(input_dim=100, hidden_dim=hidden, decoder_activation=global_sigmoid)
    _seed_all(0)
    theirs = StudentFirstOrderNetwork(hidden=hidden, dropout_p=0.1)
    return ours, theirs


def test_init_weights_bit_identical() -> None:
    """Same seed → identical fc1.weight and fc2.weight."""
    ours, theirs = _make_paired_mlps()
    assert torch.equal(ours.fc1.weight, theirs.fc1.weight)
    assert torch.equal(ours.fc2.weight, theirs.fc2.weight)


def test_forward_eval_bit_exact() -> None:
    """Single forward in eval mode (dropout off) → bit-exact."""
    ours, theirs = _make_paired_mlps()
    ours.eval()
    theirs.eval()

    x = torch.randn(8, 100)
    h1_ours, h2_ours = ours(x, cascade_rate=1.0)
    h1_their, h2_their = theirs(x, cascade_rate=1.0)
    assert torch.allclose(h1_ours, h1_their, atol=1e-7)
    assert torch.allclose(h2_ours, h2_their, atol=1e-7)


def test_50_cascade_steps_forward_parity() -> None:
    """50 cascade steps in eval mode — both implementations converge
    to the same h2 (1e-6 atol after 50 unrolls)."""
    ours, theirs = _make_paired_mlps()
    ours.eval()
    theirs.eval()

    x = torch.randn(4, 100)
    prev_h2_ours: torch.Tensor | None = None
    prev_h2_theirs: torch.Tensor | None = None
    for _ in range(50):
        _, prev_h2_ours = ours(x, prev_h2=prev_h2_ours, cascade_rate=0.02)
        _, prev_h2_theirs = theirs(x, prev_h2=prev_h2_theirs, cascade_rate=0.02)
    assert prev_h2_ours is not None and prev_h2_theirs is not None
    assert torch.allclose(prev_h2_ours, prev_h2_theirs, atol=1e-6)


def test_backward_eval_bit_exact() -> None:
    """Single forward+backward in eval — gradients on fc1.weight and
    fc2.weight match bit-exact."""
    ours, theirs = _make_paired_mlps()
    ours.eval()
    theirs.eval()

    x = torch.randn(4, 100)
    _, h2_ours = ours(x, cascade_rate=1.0)
    _, h2_their = theirs(x, cascade_rate=1.0)
    h2_ours.sum().backward()
    h2_their.sum().backward()

    assert ours.fc1.weight.grad is not None
    assert theirs.fc1.weight.grad is not None
    assert torch.allclose(ours.fc1.weight.grad, theirs.fc1.weight.grad, atol=1e-7)
    assert torch.allclose(ours.fc2.weight.grad, theirs.fc2.weight.grad, atol=1e-7)


@pytest.mark.parametrize("hidden", [40, 60, 100])
def test_init_parity_across_hidden_dims(hidden: int) -> None:
    """Init parity holds for all Blindsight hidden_dim candidates (40
    student, 60 paper Table 9, 100 legacy port)."""
    _seed_all(0)
    ours = FirstOrderMLP(input_dim=100, hidden_dim=hidden, decoder_activation=global_sigmoid)
    _seed_all(0)
    theirs = StudentFirstOrderNetwork(hidden=hidden, dropout_p=0.1)
    assert torch.equal(ours.fc1.weight, theirs.fc1.weight)
    assert torch.equal(ours.fc2.weight, theirs.fc2.weight)
