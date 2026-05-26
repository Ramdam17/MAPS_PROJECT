"""Tier 3 parity (D12.7 user choice): one isolated update step
(forward + backward + optimizer.step) bit-exact vs student.

Pinpoints any drift at the optimizer level that Tier 2 (forward+backward)
alone would miss.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import optim

from maps.core.losses import cae_loss
from maps.networks.first_order_mlp import FirstOrderMLP, global_sigmoid
from tests.parity.blindsight._student_extracts import (
    StudentFirstOrderNetwork,
    student_cae_loss,
)


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _make_paired_mlps_with_optimizers(
    hidden: int = 40, lr: float = 0.5
) -> tuple[FirstOrderMLP, StudentFirstOrderNetwork, optim.Optimizer, optim.Optimizer]:
    _seed_all(0)
    ours = FirstOrderMLP(input_dim=100, hidden_dim=hidden, decoder_activation=global_sigmoid)
    opt_ours = optim.Adamax(ours.parameters(), lr=lr)

    _seed_all(0)
    theirs = StudentFirstOrderNetwork(hidden=hidden, dropout_p=0.1)
    opt_theirs = optim.Adamax(theirs.parameters(), lr=lr)

    return ours, theirs, opt_ours, opt_theirs


def test_cae_loss_value_parity_eval() -> None:
    """Single forward + CAE loss : scalar must match bit-exact in eval."""
    ours, theirs, _, _ = _make_paired_mlps_with_optimizers()
    ours.eval()
    theirs.eval()

    x = torch.rand(20, 100)  # [0, 1] for BCE target validity
    target = (torch.rand(20, 100) > 0.5).float()

    _, h2_ours = ours(x, cascade_rate=1.0)
    _, h2_their = theirs(x, cascade_rate=1.0)
    h1_ours, _ = ours(x, cascade_rate=1.0)
    h1_their, _ = theirs(x, cascade_rate=1.0)

    loss_ours = cae_loss(
        weight=ours.fc1.weight,
        x=target,
        recons_x=h2_ours,
        hidden=h1_ours,
        lam=1e-4,
        recon="bce_sum",
    )
    loss_their = student_cae_loss(
        W=theirs.fc1.weight, x=target, recons_x=h2_their, h=h1_their, lam=1e-4
    )
    assert torch.allclose(loss_ours, loss_their, atol=1e-6), (
        f"CAE drift : ours={loss_ours.item()}, theirs={loss_their.item()}"
    )


def test_one_update_step_weights_parity_eval() -> None:
    """forward + cae_loss + backward + optim.step in eval mode →
    post-update weights match to 1e-6."""
    ours, theirs, opt_ours, opt_theirs = _make_paired_mlps_with_optimizers(lr=0.5)
    ours.eval()
    theirs.eval()

    x = torch.rand(20, 100)
    target = (torch.rand(20, 100) > 0.5).float()

    h1_ours, h2_ours = ours(x, cascade_rate=1.0)
    h1_their, h2_their = theirs(x, cascade_rate=1.0)

    loss_ours = cae_loss(
        weight=ours.fc1.weight,
        x=target,
        recons_x=h2_ours,
        hidden=h1_ours,
        lam=1e-4,
        recon="bce_sum",
    )
    loss_their = student_cae_loss(
        W=theirs.fc1.weight, x=target, recons_x=h2_their, h=h1_their, lam=1e-4
    )

    opt_ours.zero_grad()
    opt_theirs.zero_grad()
    loss_ours.backward()
    loss_their.backward()
    opt_ours.step()
    opt_theirs.step()

    # Post-update weights must match
    assert torch.allclose(ours.fc1.weight, theirs.fc1.weight, atol=1e-6)
    assert torch.allclose(ours.fc2.weight, theirs.fc2.weight, atol=1e-6)


def test_optimizer_step_with_scheduler_no_drift() -> None:
    """Multiple consecutive steps + StepLR scheduler → weights still match
    after N steps (covers scheduler arithmetic parity too)."""
    from torch.optim.lr_scheduler import StepLR

    ours, theirs, opt_ours, opt_theirs = _make_paired_mlps_with_optimizers(lr=0.5)
    ours.eval()
    theirs.eval()
    sched_ours = StepLR(opt_ours, step_size=2, gamma=0.98)
    sched_theirs = StepLR(opt_theirs, step_size=2, gamma=0.98)

    for _step in range(5):
        x = torch.rand(20, 100)
        target = (torch.rand(20, 100) > 0.5).float()

        h1_ours, h2_ours = ours(x, cascade_rate=1.0)
        h1_their, h2_their = theirs(x, cascade_rate=1.0)
        loss_ours = cae_loss(ours.fc1.weight, target, h2_ours, h1_ours, 1e-4, recon="bce_sum")
        loss_their = student_cae_loss(theirs.fc1.weight, target, h2_their, h1_their, 1e-4)
        opt_ours.zero_grad()
        opt_theirs.zero_grad()
        loss_ours.backward()
        loss_their.backward()
        opt_ours.step()
        opt_theirs.step()
        sched_ours.step()
        sched_theirs.step()

    # After 5 steps + 2 LR decays : weights still match within 1e-5
    assert torch.allclose(ours.fc1.weight, theirs.fc1.weight, atol=1e-5)
    assert torch.allclose(ours.fc2.weight, theirs.fc2.weight, atol=1e-5)
    # LR scheduler state should agree too
    assert opt_ours.param_groups[0]["lr"] == opt_theirs.param_groups[0]["lr"]
