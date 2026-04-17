"""Unit tests for cae_loss, wagering_bce_loss, distillation_loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from maps.components import cae_loss, distillation_loss, wagering_bce_loss


def test_cae_loss_bce_sum_matches_reference():
    """Default `recon='bce_sum'` matches the reference nn.BCELoss(size_average=False)."""
    W = torch.randn(40, 100)
    x = torch.randint(0, 2, (8, 100)).float()  # binary targets
    recons = torch.sigmoid(torch.randn(8, 100))  # sigmoid outputs
    h = torch.sigmoid(torch.randn(8, 40))
    loss = cae_loss(W, x, recons, h, lam=0.0)
    expected = F.binary_cross_entropy(recons, x, reduction="sum")
    assert torch.allclose(loss, expected)


def test_cae_loss_mse_variant():
    """`recon='mse_mean'` falls back to F.mse_loss for non-binary targets."""
    W = torch.randn(40, 100)
    x = torch.randn(8, 100)
    recons = torch.randn(8, 100)
    h = torch.sigmoid(torch.randn(8, 40))
    loss = cae_loss(W, x, recons, h, lam=0.0, recon="mse_mean")
    assert torch.allclose(loss, F.mse_loss(recons, x))


def test_cae_loss_grows_with_lambda():
    W = torch.randn(40, 100)
    x = torch.randint(0, 2, (8, 100)).float()
    recons = x.clone().clamp(1e-6, 1 - 1e-6)  # near-perfect reconstruction
    h = torch.sigmoid(torch.randn(8, 40))
    loss_small = cae_loss(W, x, recons, h, lam=0.01)
    loss_big = cae_loss(W, x, recons, h, lam=1.0)
    assert loss_big > loss_small


def test_wagering_bce_matches_Ftorch_bce():
    wager = torch.rand(16, 1)
    target = torch.randint(0, 2, (16, 1)).float()
    ours = wagering_bce_loss(wager, target)
    theirs = F.binary_cross_entropy(wager, target)
    assert torch.allclose(ours, theirs)


def test_distillation_soft_only_matches_manual_kl():
    student = torch.randn(4, 5)
    teacher = torch.randn(4, 5)
    T = 2.0
    alpha = 0.5

    soft_targets = F.softmax(teacher / T, dim=-1)
    log_probs = F.log_softmax(student / T, dim=-1)
    expected = (
        alpha
        * torch.sum(soft_targets * (soft_targets.clamp_min(1e-12).log() - log_probs), dim=-1).mean()
    )

    out = distillation_loss(student, teacher, hard_labels=None, alpha=alpha, temperature=T)
    assert torch.allclose(out, expected)


def test_distillation_with_hard_labels_is_mixture():
    student = torch.randn(4, 5, requires_grad=True)
    teacher = torch.randn(4, 5)
    hard = torch.randint(0, 5, (4,))
    out = distillation_loss(student, teacher, hard_labels=hard, alpha=0.5, temperature=2.0)
    out.backward()
    assert student.grad is not None
