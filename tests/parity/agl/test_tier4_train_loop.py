"""Tier-4-light + Tier-5 parity: AGL pre-train loop & reset.

Sprint 13.G / D13.4:
- **Tier 4-light** — ``AGLTrainer.pre_train`` is (a) deterministic under a fixed
  seed and (b) matches a hand-written inline replica of the student ``pre_train``
  control flow bit-exact — guarding the two-loss orchestration against drift.
- **Tier 5** — the reset (D-agl-reset) is deterministic and returns the
  1st-order to its initial weights bit-exact.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from maps.core.losses import cae_loss, wagering_bce_loss
from maps.domains.agl.data import array_words, target_second
from maps.domains.agl.trainer import SETTINGS_REGISTRY, AGLTrainer
from maps.utils.config import load_config
from maps.utils.seeding import set_all_seeds

N_EPOCHS = 3


def _cfg():
    return load_config(
        "domains/agl/training",
        overrides=[
            f"train.n_epochs_pretrain={N_EPOCHS}",
            "train.batch_size_pretrain=8",
        ],
    )


def _run_trainer(setting_id="setting-3-second-order-only"):
    set_all_seeds(42)
    t = AGLTrainer(SETTINGS_REGISTRY[setting_id], seed=42, cfg=_cfg())
    t.build()
    l1, l2, prec = t.pre_train()
    return t, l1, l2, prec


def test_pretrain_is_deterministic():
    """Same seed → identical loss curves (reproducibility guarantee)."""
    _, l1_a, l2_a, _ = _run_trainer()
    _, l1_b, l2_b, _ = _run_trainer()
    assert np.array_equal(l1_a, l1_b)
    assert np.array_equal(l2_a, l2_b)


def test_reset_restores_initial_weights_bit_exact():
    """Tier 5: after pre_train the 1st-order equals its build-time weights."""
    set_all_seeds(42)
    t = AGLTrainer(SETTINGS_REGISTRY["setting-3-second-order-only"], seed=42, cfg=_cfg())
    t.build()
    initial = copy.deepcopy(t.first_order.state_dict())
    t.pre_train()
    after = t.first_order.state_dict()
    assert all(torch.equal(initial[k], after[k]) for k in initial)


def test_pretrain_matches_inline_student_flow():
    """Tier 4-light: trainer.pre_train == inline replica of the student loop.

    Both start from ``set_all_seeds(42)`` and build identical networks (same
    seed + construction order), then run the same two-loss control flow, so the
    loss curves must coincide to floating-point tolerance. Guards the trainer's
    orchestration (op order, RNG consumption, optimizer stepping) against drift.
    """
    _, ref_l1, ref_l2, _ = _run_trainer("setting-3-second-order-only")

    # Inline replica, identical seed / build order.
    set_all_seeds(42)
    cfg = _cfg()
    t = AGLTrainer(SETTINGS_REGISTRY["setting-3-second-order-only"], seed=42, cfg=cfg)
    t.build()
    fo, so = t.first_order, t.second_order
    opt1, opt2 = t.optimizer_1, t.optimizer_2
    sched1, sched2 = t.scheduler_1, t.scheduler_2
    initial = copy.deepcopy(fo.state_dict())
    rate_1, iters_1, rate_2, iters_2 = t._cascade_params()
    batch = cfg.train.batch_size_pretrain
    lam = cfg.losses.cae_lambda

    inl_l1 = np.zeros(N_EPOCHS)
    inl_l2 = np.zeros(N_EPOCHS)
    for epoch in range(N_EPOCHS):
        patterns = array_words(1, batch, device="cpu")
        h1 = h2 = None
        for _ in range(iters_1):
            h1, h2 = fo(patterns, h1, h2, rate_1)
        opt1.zero_grad()
        comparison = wager = None
        for _ in range(iters_2):
            wager, comparison = so(patterns, h2, comparison, rate_2)
        target = target_second(patterns, h2)
        loss_2 = wagering_bce_loss(wager.squeeze(), target, reduction="sum")
        loss_2.backward(retain_graph=True)
        opt2.step()
        sched2.step()
        opt2.zero_grad()
        loss_1 = cae_loss(
            weight=fo.fc1.weight, x=patterns, recons_x=h2, hidden=h1, lam=lam, recon="bce_sum"
        )
        loss_1.backward()
        opt1.step()
        sched1.step()
        inl_l1[epoch] = loss_1.item()
        inl_l2[epoch] = loss_2.item()
    fo.load_state_dict(initial)

    assert np.allclose(ref_l1, inl_l1, atol=1e-5), f"{ref_l1} != {inl_l1}"
    assert np.allclose(ref_l2, inl_l2, atol=1e-5), f"{ref_l2} != {inl_l2}"
