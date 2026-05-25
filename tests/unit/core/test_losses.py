"""Unit tests for :mod:`maps.core.losses`.

- ``cae_loss``: parity vs ``external/paper_reference/blindsight_tmlr.py``
  inline ``CAE_loss`` (extracted verbatim into ``_student_cae_loss``
  below), plus quirk preservation tests.
- ``wagering_bce_loss``: equivalence to ``nn.BCELoss``.
- ``weight_regularization``: identity → 0, drift → exact, topology
  mismatch raises.
- ``simclr_loss``: sanity (monotonicity in similarity, temperature
  scaling, gradient flow, reduction modes, shape validation).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from maps.core.losses import (
    cae_loss,
    simclr_loss,
    wagering_bce_loss,
    weight_regularization,
)

# ---------------------------------------------------------------------------
# Student CAE_loss — extracted verbatim from blindsight_tmlr.py:91-129
# ---------------------------------------------------------------------------


def _student_cae_loss(
    W: torch.Tensor,
    x: torch.Tensor,
    recons_x: torch.Tensor,
    h: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Verbatim port of external/paper_reference/blindsight_tmlr.py:91-129.

    Two cosmetic modernizations: ``size_average=False`` → ``reduction='sum'``
    (the former is deprecated PyTorch API but mathematically identical),
    and dropped the ``torch.autograd.Variable`` wrapper which is a no-op
    in PyTorch ≥ 0.4. The numerical result is bit-for-bit identical.
    """
    mse_loss = nn.BCELoss(reduction="sum")  # student: BCELoss(size_average=False)
    mse = mse_loss(recons_x, x)
    dh = h * (1 - h)
    w_sum = torch.sum(W**2, dim=1)
    w_sum = w_sum.unsqueeze(1)
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)


# ---------------------------------------------------------------------------
# cae_loss
# ---------------------------------------------------------------------------


def _random_cae_inputs(
    batch_size: int = 8, n_input: int = 100, n_hidden: int = 40
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random but valid (B, hidden, input) for CAE inputs."""
    x = torch.rand(batch_size, n_input)  # ∈ [0, 1]
    recons_x = torch.rand(batch_size, n_input)  # ∈ [0, 1] for BCE
    hidden = torch.rand(batch_size, n_hidden)  # ∈ [0, 1] so h(1-h) is positive
    weight = torch.randn(n_hidden, n_input)
    return weight, x, recons_x, hidden


def test_cae_loss_parity_with_student() -> None:
    """Our cae_loss must match the verbatim student CAE_loss bit-exact
    (modulo float32 noise) at the same inputs and λ."""
    weight, x, recons_x, hidden = _random_cae_inputs()
    lam = 1e-4

    ours = cae_loss(weight, x, recons_x, hidden, lam, recon="bce_sum")
    theirs = _student_cae_loss(weight, x, recons_x, hidden, lam)

    assert torch.allclose(ours, theirs, atol=1e-6), (
        f"CAE drift: ours={ours.item()}, theirs={theirs.item()}"
    )


def test_cae_loss_recon_variants_differ() -> None:
    """bce_sum vs mse_sum vs mse_mean produce different scalars."""
    weight, x, recons_x, hidden = _random_cae_inputs()
    lam = 1e-4

    bce_sum = cae_loss(weight, x, recons_x, hidden, lam, recon="bce_sum")
    mse_sum = cae_loss(weight, x, recons_x, hidden, lam, recon="mse_sum")
    mse_mean = cae_loss(weight, x, recons_x, hidden, lam, recon="mse_mean")

    assert not torch.allclose(bce_sum, mse_sum, atol=1e-3)
    assert not torch.allclose(mse_sum, mse_mean, atol=1e-3)


def test_cae_loss_invalid_recon_raises() -> None:
    weight, x, recons_x, hidden = _random_cae_inputs()
    with pytest.raises(ValueError, match="recon must be one of"):
        cae_loss(weight, x, recons_x, hidden, 1e-4, recon="huber")


def test_cae_loss_weight_is_detached() -> None:
    """Gradient must NOT flow directly into `weight` via the
    contractive term (matches student state_dict() detach behavior)."""
    weight, x, recons_x, hidden = _random_cae_inputs()
    weight.requires_grad_(True)
    hidden.requires_grad_(True)

    loss = cae_loss(weight, x, recons_x, hidden, 1e-4, recon="bce_sum")
    loss.backward()

    assert weight.grad is None, (
        "weight.grad should be None — gradient must NOT flow through "
        "weight directly (only through hidden)."
    )
    assert hidden.grad is not None
    assert hidden.grad.abs().sum() > 0


def test_cae_loss_h_one_minus_h_quirk_preserved() -> None:
    """The h(1-h) formula is preserved (sigmoid-derivative form), even
    when `hidden` comes from ReLU. Test: feed a hidden with values >1
    (ReLU output can be unbounded) and check the formula still applies
    without correction. h(1-h) for h=2 = 2*(1-2) = -2, which would NOT
    happen with a sigmoid-correct formula (sigmoid outputs ∈ [0,1])."""
    weight, x, recons_x, _ = _random_cae_inputs()
    # ReLU-like hidden: values > 1, so h(1-h) goes negative
    hidden = torch.tensor([[2.0, 0.5, 3.0]])
    x = torch.rand(1, 10)
    recons_x = torch.rand(1, 10)
    weight = torch.randn(3, 10)

    ours = cae_loss(weight, x, recons_x, hidden, 1e-4, recon="bce_sum")
    theirs = _student_cae_loss(weight, x, recons_x, hidden, 1e-4)
    # Bit-exact match — confirms the quirk is byte-for-byte preserved.
    assert torch.allclose(ours, theirs, atol=1e-6)


# ---------------------------------------------------------------------------
# wagering_bce_loss
# ---------------------------------------------------------------------------


def test_wagering_bce_matches_nn_bceloss_sum() -> None:
    """`reduction='sum'` is the student trainer's setting — must match
    nn.BCELoss(reduction='sum') exactly."""
    wager = torch.rand(16, 1)
    target = torch.randint(0, 2, (16, 1)).float()

    ours = wagering_bce_loss(wager, target, reduction="sum")
    theirs = nn.BCELoss(reduction="sum")(wager, target)
    assert torch.allclose(ours, theirs, atol=1e-7)


def test_wagering_bce_default_reduction_is_mean() -> None:
    """Default reduction is 'mean' (modern PyTorch convention)."""
    wager = torch.rand(16, 1)
    target = torch.randint(0, 2, (16, 1)).float()

    default = wagering_bce_loss(wager, target)
    explicit_mean = wagering_bce_loss(wager, target, reduction="mean")
    assert torch.allclose(default, explicit_mean, atol=1e-7)


def test_wagering_bce_none_returns_per_element_tensor() -> None:
    """`reduction='none'` returns per-element tensor (same shape)."""
    wager = torch.rand(4, 1)
    target = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    out = wagering_bce_loss(wager, target, reduction="none")
    assert out.shape == wager.shape


# ---------------------------------------------------------------------------
# weight_regularization
# ---------------------------------------------------------------------------


def test_weight_reg_identical_models_is_zero() -> None:
    """L2 drift between a model and itself is exactly 0."""
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    import copy

    teacher = copy.deepcopy(model)
    teacher.requires_grad_(False)

    loss = weight_regularization(model, teacher)
    assert torch.equal(loss, torch.tensor(0.0))


def test_weight_reg_single_param_diff_is_exact() -> None:
    """Drift of a known magnitude on one param produces the exact L2."""
    student = nn.Linear(3, 2, bias=False)
    teacher = nn.Linear(3, 2, bias=False)
    teacher.requires_grad_(False)

    with torch.no_grad():
        student.weight.copy_(teacher.weight)
        # Add a controlled drift of magnitude 2 on a single param
        student.weight[0, 0] += 2.0

    loss = weight_regularization(student, teacher)
    assert torch.allclose(loss, torch.tensor(4.0), atol=1e-6)  # 2² = 4


def test_weight_reg_mismatched_topology_raises() -> None:
    """zip(strict=True) must fail-fast on different parameter counts."""
    student = nn.Linear(10, 5)
    teacher = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
    teacher.requires_grad_(False)

    with pytest.raises(ValueError):
        weight_regularization(student, teacher)


# ---------------------------------------------------------------------------
# simclr_loss (NT-Xent)
# ---------------------------------------------------------------------------


def test_simclr_monotonic_in_similarity() -> None:
    """A 'good' positive pair (z_i ≈ z_j) yields a LOWER loss than a
    'bad' positive pair (z_i orthogonal to z_j)."""
    n, d = 8, 32
    z_i = torch.randn(n, d)
    z_j_good = z_i + 0.01 * torch.randn(n, d)  # nearly identical → strong positives
    z_j_bad = torch.randn(n, d)  # random → orthogonal-ish positives

    loss_good = simclr_loss(z_i, z_j_good, temperature=0.5)
    loss_bad = simclr_loss(z_i, z_j_bad, temperature=0.5)

    assert loss_good < loss_bad


def test_simclr_temperature_scaling() -> None:
    """Lower τ → sharper softmax → larger gap between loss_good and
    loss_bad. The ordering loss_bad > loss_good must hold for both,
    and the gap should grow as τ shrinks."""
    n, d = 16, 64
    z_i = torch.randn(n, d)
    z_j_good = z_i + 0.01 * torch.randn(n, d)
    z_j_bad = torch.randn(n, d)

    gap_high_temp = simclr_loss(z_i, z_j_bad, temperature=1.0) - simclr_loss(
        z_i, z_j_good, temperature=1.0
    )
    gap_low_temp = simclr_loss(z_i, z_j_bad, temperature=0.1) - simclr_loss(
        z_i, z_j_good, temperature=0.1
    )

    assert gap_low_temp > gap_high_temp


def test_simclr_gradient_flow() -> None:
    """backward() must populate gradients on both z_i and z_j."""
    z_i = torch.randn(8, 16, requires_grad=True)
    z_j = torch.randn(8, 16, requires_grad=True)

    loss = simclr_loss(z_i, z_j, temperature=0.5)
    loss.backward()

    assert z_i.grad is not None and z_i.grad.abs().sum() > 0
    assert z_j.grad is not None and z_j.grad.abs().sum() > 0


def test_simclr_reduction_mean_vs_sum() -> None:
    """mean == sum / (2N) for any inputs."""
    n, d = 12, 32
    z_i = torch.randn(n, d)
    z_j = torch.randn(n, d)

    loss_mean = simclr_loss(z_i, z_j, temperature=0.5, reduction="mean")
    loss_sum = simclr_loss(z_i, z_j, temperature=0.5, reduction="sum")
    assert torch.allclose(loss_mean, loss_sum / (2 * n), atol=1e-5)


def test_simclr_reduction_none_returns_per_row() -> None:
    """reduction='none' returns a (2N,) tensor — one loss per row."""
    n, d = 6, 8
    z_i = torch.randn(n, d)
    z_j = torch.randn(n, d)

    loss = simclr_loss(z_i, z_j, temperature=0.5, reduction="none")
    assert loss.shape == (2 * n,)


def test_simclr_shape_mismatch_raises() -> None:
    z_i = torch.randn(8, 16)
    z_j = torch.randn(8, 32)  # different D
    with pytest.raises(ValueError, match="same shape"):
        simclr_loss(z_i, z_j)


def test_simclr_non_2d_raises() -> None:
    z_i = torch.randn(8)  # 1D
    z_j = torch.randn(8)
    with pytest.raises(ValueError, match="2D"):
        simclr_loss(z_i, z_j)


def test_simclr_invalid_temperature_raises() -> None:
    z_i = torch.randn(4, 8)
    z_j = torch.randn(4, 8)
    with pytest.raises(ValueError, match="temperature"):
        simclr_loss(z_i, z_j, temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        simclr_loss(z_i, z_j, temperature=-0.1)


def test_simclr_invariant_to_l2_rescaling() -> None:
    """Cosine similarity is scale-invariant: multiplying z_i by 5
    should not change the loss (within float tolerance)."""
    z_i = torch.randn(8, 16)
    z_j = torch.randn(8, 16)

    loss_raw = simclr_loss(z_i, z_j, temperature=0.5)
    loss_scaled = simclr_loss(z_i * 5.0, z_j * 5.0, temperature=0.5)
    assert torch.allclose(loss_raw, loss_scaled, atol=1e-5)


def test_simclr_cross_check_manual_cross_entropy() -> None:
    """Hand-compute NT-Xent for N=2 against the function output.

    This is the R4 mitigation from the Sprint 11 spec: cross-check
    the NT-Xent maths against an independent computation using
    F.cross_entropy on a manually built similarity matrix.
    """
    n, d = 2, 4
    torch.manual_seed(7)
    z_i = torch.randn(n, d)
    z_j = torch.randn(n, d)
    temperature = 0.5

    # Manual reference computation
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    sim = (z @ z.T) / temperature
    mask = torch.eye(2 * n, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))
    positives = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    ref = F.cross_entropy(sim, positives, reduction="mean")

    ours = simclr_loss(z_i, z_j, temperature=temperature)
    assert torch.allclose(ours, ref, atol=1e-7)
