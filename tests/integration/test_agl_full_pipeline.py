"""End-to-end smoke for the AGL 3-phase protocol (D.29).

Verifies that ``run_agl.py``'s orchestration pattern works programmatically :
pretrain → replicate to pool → train cells by tier → evaluate_pool.

Small problem size (num_networks=4, n_epochs_pretrain=5, train-high=3,
train-low=2) to keep the test fast. Asserts structural invariants (tier
dict keys, array shapes, metric ranges) rather than bit-exact parity — the
parity side is handled by ``test_agl_pretrain.py`` and ``test_agl_training.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from maps.experiments.agl import (
    AGLNetworkPool,
    AGLSetting,
    AGLTrainer,
)
from maps.utils import load_config, set_all_seeds


# All 4 factorial settings.
SETTINGS = [
    AGLSetting(id="neither", label="", cascade=False, second_order=False),
    AGLSetting(id="cascade_only", label="", cascade=True, second_order=False),
    AGLSetting(id="second_order_only", label="", cascade=False, second_order=True),
    AGLSetting(id="both", label="", cascade=True, second_order=True),
]


@pytest.fixture
def tiny_cfg():
    """Config sized for fast end-to-end test."""
    return load_config(
        "training/agl",
        overrides=[
            "train.n_epochs_pretrain=5",
            "train.batch_size_pretrain=10",
            "train.n_epochs_training_high=3",
            "train.n_epochs_training_low=2",
            "train.batch_size_training=10",
            "train.num_networks=4",
            "cascade.n_iterations=3",
            "cascade.alpha=0.33",
            "optimizer.name=ADAMAX",  # faster than RangerVA for tests
        ],
    )


@pytest.mark.parametrize("setting", SETTINGS, ids=lambda s: s.id)
def test_full_pipeline_runs_end_to_end(tiny_cfg, setting):
    """Pretrain → pool → train tiers → evaluate_pool must complete for every
    factorial setting and return a well-structured metrics dict."""
    set_all_seeds(42)
    trainer = AGLTrainer(tiny_cfg, setting)
    trainer.build()

    # Phase 1: pretrain.
    l1_pre, l2_pre = trainer.pre_train()
    assert l1_pre.shape == (5,)
    assert l2_pre.shape == (5,)
    assert np.all(np.isfinite(l1_pre))

    # Phase 2: replicate.
    pool = AGLNetworkPool(trainer, num_networks=4)
    assert len(pool) == 4

    # Phase 3: train high (cells 0:2) + low (cells 2:4).
    high_result = pool.train_range(0, 2, n_epochs=3)
    low_result = pool.train_range(2, 4, n_epochs=2)
    assert high_result["losses_1"].shape == (2, 3)
    assert low_result["losses_1"].shape == (2, 2)

    # Phase 4: evaluate.
    eval_metrics = trainer.evaluate_pool(pool)
    assert set(eval_metrics.keys()) == {"high", "low", "overall"}
    for tier in ("high", "low", "overall"):
        assert "precision_1st" in eval_metrics[tier]
        p = eval_metrics[tier]["precision_1st"]
        assert 0.0 <= p <= 1.0, f"precision_1st out of range on {setting.id}/{tier}: {p}"
        if setting.second_order:
            wa = eval_metrics[tier].get("wager_accuracy")
            assert wa is not None, f"wager_accuracy missing on {setting.id}/{tier}"
            assert 0.0 <= wa <= 1.0, (
                f"wager_accuracy out of range on {setting.id}/{tier}: {wa}"
            )


def test_full_pipeline_tier_split_respects_num_networks_over_2(tiny_cfg):
    """High tier must cover cells [0:N//2], low [N//2:N]. Overall averages all."""
    set_all_seeds(42)
    trainer = AGLTrainer(tiny_cfg, SETTINGS[-1])  # both
    trainer.build()
    trainer.pre_train()

    pool = AGLNetworkPool(trainer, num_networks=4)
    # Train everyone, identical n_epochs, to make tier means comparable.
    pool.train_range(0, 4, n_epochs=2)

    result = trainer.evaluate_pool(pool)
    # overall precision_1st should be mean of (high, low) precisions when same
    # training — not exactly because per-cell test batches differ, but close.
    p_hi = result["high"]["precision_1st"]
    p_lo = result["low"]["precision_1st"]
    p_all = result["overall"]["precision_1st"]
    assert abs(p_all - (p_hi + p_lo) / 2) < 0.2, (
        f"overall mean {p_all} not close to (high+low)/2 = {(p_hi+p_lo)/2}"
    )


def test_full_pipeline_preserves_pretrain_after_pool_training(tiny_cfg):
    """The trainer's own (first_order, second_order) must NOT be modified
    when cells in the pool are trained — the pool owns deep-copied weights."""
    set_all_seeds(42)
    trainer = AGLTrainer(tiny_cfg, SETTINGS[-1])
    trainer.build()
    trainer.pre_train()

    # Snapshot trainer's weights post-pretrain (note: pre_train resets fo to
    # initial weights per student L751, so this is the RESET weights).
    fo_snapshot = trainer.first_order.fc1.weight.detach().clone()
    so_snapshot = trainer.second_order.wagering_head.wager.weight.detach().clone()

    pool = AGLNetworkPool(trainer, num_networks=3)
    pool.train_range(0, 3, n_epochs=3)

    # Trainer's own weights must be unchanged.
    assert torch.allclose(fo_snapshot, trainer.first_order.fc1.weight), (
        "trainer.first_order leaked after pool training"
    )
    assert torch.allclose(so_snapshot, trainer.second_order.wagering_head.wager.weight), (
        "trainer.second_order leaked after pool training"
    )
