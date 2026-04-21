"""Unit tests for :mod:`maps.experiments.agl.pool`.

Covers :
- Pool initialization : cells have identical post-pretrain weights.
- Cell independence : training one cell does not affect others.
- ``train_range`` : decreases loss_1 and produces precision metrics.
- ``len(pool)`` matches ``num_networks``.
"""

from __future__ import annotations

import pytest
import torch

from maps.experiments.agl import AGLNetworkPool, AGLSetting, AGLTrainer
from maps.utils import load_config, set_all_seeds


SETTINGS = [
    AGLSetting(id="neither", label="", cascade=False, second_order=False),
    AGLSetting(id="both", label="", cascade=True, second_order=True),
]


@pytest.fixture
def tiny_cfg():
    """Small config for fast pool tests (5 pretrain epochs, batch 10)."""
    return load_config(
        "training/agl",
        overrides=[
            "train.n_epochs_pretrain=5",
            "train.batch_size_pretrain=10",
            "train.batch_size_training=10",
            "train.num_networks=3",  # enough to test [0:1], [1:3] split
            "cascade.n_iterations=3",
            "cascade.alpha=0.33",
            "optimizer.name=ADAMAX",  # faster than RangerVA for tests
        ],
    )


def _build_pool(cfg, setting: AGLSetting, num_networks: int = 3) -> AGLNetworkPool:
    set_all_seeds(42)
    trainer = AGLTrainer(cfg, setting)
    trainer.build()
    trainer.pre_train()
    return AGLNetworkPool(trainer, num_networks=num_networks)


@pytest.mark.parametrize("setting", SETTINGS, ids=lambda s: s.id)
def test_pool_init_cells_start_identical(tiny_cfg, setting):
    """All cells must start with the same weights (post-pretrain snapshot)."""
    pool = _build_pool(tiny_cfg, setting, num_networks=3)
    assert len(pool) == 3

    w0_fo = pool.cells[0].first_order.fc1.weight.detach()
    w1_fo = pool.cells[1].first_order.fc1.weight.detach()
    w2_fo = pool.cells[2].first_order.fc1.weight.detach()
    assert torch.allclose(w0_fo, w1_fo)
    assert torch.allclose(w1_fo, w2_fo)

    w0_so = pool.cells[0].second_order.wagering_head.wager.weight.detach()
    w1_so = pool.cells[1].second_order.wagering_head.wager.weight.detach()
    assert torch.allclose(w0_so, w1_so)


def test_pool_cells_are_independent_after_training(tiny_cfg):
    """Training cell 0 must NOT change cell 1 or 2 weights."""
    pool = _build_pool(tiny_cfg, SETTINGS[1], num_networks=3)  # setting=both

    # Snapshot cells 1, 2 before training cell 0.
    w1_fo_before = pool.cells[1].first_order.fc1.weight.detach().clone()
    w2_fo_before = pool.cells[2].first_order.fc1.weight.detach().clone()

    # Train ONLY cell 0.
    pool.train_range(start=0, end=1, n_epochs=3)

    # Cell 0 should have moved.
    w0_fo_after = pool.cells[0].first_order.fc1.weight.detach()
    w0_fo_init = pool.cells[1].first_order.fc1.weight.detach()  # unchanged from init
    assert not torch.allclose(w0_fo_after, w0_fo_init), "cell 0 didn't train"

    # Cells 1, 2 must not have moved.
    w1_fo_after = pool.cells[1].first_order.fc1.weight.detach()
    w2_fo_after = pool.cells[2].first_order.fc1.weight.detach()
    assert torch.allclose(w1_fo_before, w1_fo_after), "cell 1 leaked"
    assert torch.allclose(w2_fo_before, w2_fo_after), "cell 2 leaked"


def test_pool_train_range_decreases_loss_1(tiny_cfg):
    """``train_range`` should produce decreasing loss_1 trajectories."""
    pool = _build_pool(tiny_cfg, SETTINGS[1], num_networks=3)
    result = pool.train_range(start=0, end=3, n_epochs=5)

    assert result["losses_1"].shape == (3, 5)
    assert result["losses_2"].shape == (3, 5)
    assert result["precision"].shape == (3, 5)

    # Loss 1 should be lower at end than at start on average.
    for row in range(3):
        assert result["losses_1"][row, -1] < result["losses_1"][row, 0] * 2.0, (
            "loss_1 trajectory worrying (end not much better than start, row=%d)" % row
        )


def test_pool_train_range_rejects_bad_ranges(tiny_cfg):
    pool = _build_pool(tiny_cfg, SETTINGS[0], num_networks=3)
    with pytest.raises(ValueError):
        pool.train_range(start=0, end=4, n_epochs=1)  # end > num_networks
    with pytest.raises(ValueError):
        pool.train_range(start=-1, end=2, n_epochs=1)
    with pytest.raises(ValueError):
        pool.train_range(start=2, end=2, n_epochs=1)  # end == start


def test_evaluate_pool_returns_high_low_overall_tiers(tiny_cfg):
    """evaluate_pool aggregates per-cell metrics into high / low / overall tiers."""
    set_all_seeds(42)
    trainer = AGLTrainer(tiny_cfg, SETTINGS[1])  # setting=both
    trainer.build()
    trainer.pre_train()
    # num_networks=4 → high=[0:2], low=[2:4].
    pool = AGLNetworkPool(trainer, num_networks=4)
    pool.train_range(0, 2, n_epochs=3)
    pool.train_range(2, 4, n_epochs=1)

    result = trainer.evaluate_pool(pool)

    assert set(result.keys()) == {"high", "low", "overall"}
    for tier in ("high", "low", "overall"):
        assert "precision_1st" in result[tier]
        assert "precision_1st_std" in result[tier]
        assert "wager_accuracy" in result[tier]  # setting.second_order=True
        assert "wager_accuracy_std" in result[tier]
        # f1 / precision_2nd / recall_2nd also present.
        assert "f1_2nd" in result[tier]


def test_evaluate_pool_without_second_order_has_no_wager_keys(tiny_cfg):
    """When setting.second_order=False, wager keys must be absent from tiers."""
    set_all_seeds(42)
    trainer = AGLTrainer(tiny_cfg, SETTINGS[0])  # setting=neither (second_order=False)
    trainer.build()
    trainer.pre_train()
    pool = AGLNetworkPool(trainer, num_networks=3)
    pool.train_range(0, 3, n_epochs=2)

    result = trainer.evaluate_pool(pool)
    for tier in ("high", "low", "overall"):
        assert "precision_1st" in result[tier]
        assert "wager_accuracy" not in result[tier]
        assert "f1_2nd" not in result[tier]


def test_evaluate_pool_rejects_empty_pool(tiny_cfg):
    """evaluate_pool must raise on empty pool."""
    set_all_seeds(42)
    trainer = AGLTrainer(tiny_cfg, SETTINGS[0])
    trainer.build()

    class _EmptyPool:
        cells = []

        def __len__(self):
            return 0

    with pytest.raises(ValueError):
        trainer.evaluate_pool(_EmptyPool())
