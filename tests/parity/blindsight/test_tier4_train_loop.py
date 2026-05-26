"""Tier 4-light parity (D12.7) : full ``BlindsightTrainer.train()``
loop bit-exact vs student-equivalent inline reference, on 2 epochs.

This is the ultimate parity guard : if our refactored trainer
produces a different loss sequence from the verbatim student logic
at the same seeds, something in the port has drifted.

Important caveat
----------------
The student ``SecondOrderNetwork`` never instantiates the Pasquali
hidden layer (D.25 bug). To compare apples-to-apples, the test
overrides ``second_order.hidden_dim = 0`` (None) in cfg, disabling
our restored hidden path. **Production runs use Pasquali hidden
(D.25)** ; this override is parity-test-only.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

from maps.domains.blindsight.trainer import (
    SETTINGS_REGISTRY,
    BlindsightTrainer,
)
from maps.utils.config import load_config
from tests.parity.blindsight._student_extracts import (
    StudentFirstOrderNetwork,
    StudentSecondOrderNetwork,
    student_cae_loss,
    student_generate_patterns,
)


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _small_cfg() -> DictConfig:
    """Composed cfg with second_order.hidden_dim=0 (disables Pasquali)."""
    training_cfg = load_config("domains/blindsight/training")
    env_cfg = load_config("domains/blindsight/env", resolve=False)
    merged = OmegaConf.merge(training_cfg, env_cfg)
    OmegaConf.resolve(merged)
    # Shrink for unit-test speed
    merged.train.n_epochs = 2
    merged.train.batch_size = 50  # student factor=1 * patterns_number ~= 100 ; use 50 here
    # PARITY OVERRIDE : disable Pasquali hidden to match student D.25 bug
    merged.second_order.hidden_dim = 0
    return merged  # type: ignore[return-value]


def _student_pre_train_2_epochs(
    seed: int, n_epochs: int = 2, batch_size: int = 50, cascade_rate: float = 0.02
) -> tuple[list[float], list[float]]:
    """Verbatim student pre_train logic for n_epochs (Setting 6 Full MAPS)."""
    _seed_all(seed)

    fo = StudentFirstOrderNetwork(hidden=40, dropout_p=0.1)
    so = StudentSecondOrderNetwork(input_dim=100, dropout_p=0.5)

    opt_1 = optim.Adamax(fo.parameters(), lr=0.5)
    opt_2 = optim.Adamax(so.parameters(), lr=0.1)
    sched_1 = StepLR(opt_1, step_size=25, gamma=0.98)
    sched_2 = StepLR(opt_2, step_size=25, gamma=0.98)

    iters = int(1.0 / cascade_rate)  # = 50
    losses_1, losses_2 = [], []

    for _ in range(n_epochs):
        # Generate data : SUPERTHRESHOLD condition (matches default
        # pre_training_condition).
        patterns, stim_present, _, order_2 = student_generate_patterns(
            patterns_number=batch_size,
            num_units=100,
            factor=1,
            condition=0,
            noise_level=0.1,
        )

        # Cascade 1st-order
        h1: torch.Tensor | None = None
        h2: torch.Tensor | None = None
        for _ in range(iters):
            h1, h2 = fo(patterns, h1, h2, cascade_rate)
        assert h1 is not None and h2 is not None

        opt_1.zero_grad()

        # Cascade 2nd-order (Full MAPS : meta=True)
        wager: torch.Tensor | None = None
        comparison: torch.Tensor | None = None
        for _ in range(iters):
            wager, comparison = so(patterns, h2, comparison, cascade_rate)
        assert wager is not None

        # Wager loss (BCE sum on probs) — eq.5
        criterion_2 = nn.BCELoss(reduction="sum")
        loss_2 = criterion_2(wager.squeeze(-1), order_2[:, 0])
        opt_2.zero_grad()
        loss_2.backward(retain_graph=True)
        opt_2.step()
        sched_2.step()
        losses_2.append(loss_2.item())

        # CAE loss — eq.4 (Rifai), with W from state_dict (already
        # detached). Pass stim_present.view(-1, 100) as the recon target.
        loss_1 = student_cae_loss(
            W=fo.state_dict()["fc1.weight"],
            x=stim_present.view(-1, 100),
            recons_x=h2,
            h=h1,
            lam=1e-4,
        )
        loss_1.backward(retain_graph=True)
        opt_1.step()
        sched_1.step()
        losses_1.append(loss_1.item())

    return losses_1, losses_2


def test_train_2_epochs_loss_parity_full_maps() -> None:
    """Setting 6 Full MAPS, 2 epochs, same seed → loss sequences match
    student-equivalent inline reference to 1e-4 (multi-epoch float
    accumulation tolerance, D12.7 spec)."""
    cfg = _small_cfg()
    seed = 42

    # --- Our trainer ---
    _seed_all(seed)
    trainer = BlindsightTrainer(
        setting=SETTINGS_REGISTRY["setting-6-full-maps"],
        seed=seed,
        cfg=cfg,
        device="cpu",
    )
    trainer.build()
    ours = trainer.train(n_epochs=2, condition_name="superthreshold")

    # --- Student-equivalent inline reference ---
    theirs_loss_1, theirs_loss_2 = _student_pre_train_2_epochs(
        seed=seed, n_epochs=2, batch_size=int(cfg.train.batch_size)
    )

    # Compare loss sequences
    drift_1 = max(
        abs(float(o) - float(t)) for o, t in zip(ours.losses_1, theirs_loss_1, strict=True)
    )
    drift_2 = max(
        abs(float(o) - float(t)) for o, t in zip(ours.losses_2, theirs_loss_2, strict=True)
    )
    assert drift_1 < 1e-4, (
        f"loss_1 drift {drift_1:.6f} > 1e-4 — ours={list(ours.losses_1)}, theirs={theirs_loss_1}"
    )
    assert drift_2 < 1e-4, (
        f"loss_2 drift {drift_2:.6f} > 1e-4 — ours={list(ours.losses_2)}, theirs={theirs_loss_2}"
    )
