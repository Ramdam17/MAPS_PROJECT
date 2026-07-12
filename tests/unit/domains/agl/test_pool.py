"""Unit tests for :class:`maps.domains.agl.pool.AGLNetworkPool` (Sprint 13.E).

Covers the 20-cell replication mechanics:
- the pool holds ``num_networks`` cells;
- cells are independent (mutating one does not touch another);
- ``train_range`` trains only the requested slice;
- ``evaluate`` returns tiered (high/low/overall) aggregated metrics.
"""

from __future__ import annotations

import copy

import torch

from maps.domains.agl.pool import AGLNetworkPool
from maps.domains.agl.trainer import SETTINGS_REGISTRY, AGLTrainer
from maps.utils.config import load_config
from maps.utils.seeding import set_all_seeds


def _built_trainer(setting_id="setting-3-second-order-only"):
    cfg = load_config(
        "domains/agl/training",
        overrides=[
            "train.n_epochs_pretrain=1",
            "train.batch_size_pretrain=6",
            "train.batch_size_training=6",
        ],
    )
    t = AGLTrainer(SETTINGS_REGISTRY[setting_id], seed=42, cfg=cfg)
    t.build()
    t.pre_train()
    return t


def _fo_state(cell):
    return copy.deepcopy(cell.first_order.state_dict())


def _equal(a, b) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


def test_pool_has_num_networks_cells():
    set_all_seeds(42)
    pool = AGLNetworkPool(_built_trainer(), num_networks=4)
    assert len(pool) == 4


def test_pool_cells_are_independent():
    set_all_seeds(42)
    pool = AGLNetworkPool(_built_trainer(), num_networks=4)
    with torch.no_grad():
        for p in pool.cells[0].first_order.parameters():
            p.add_(1.0)
    # cell 1 must be untouched
    assert not _equal(
        pool.cells[0].first_order.state_dict(), pool.cells[1].first_order.state_dict()
    )


def test_pool_cells_start_identical():
    """All cells start from the same (post-pretrain) weights."""
    set_all_seeds(42)
    pool = AGLNetworkPool(_built_trainer(), num_networks=4)
    assert _equal(pool.cells[0].first_order.state_dict(), pool.cells[2].first_order.state_dict())


def test_train_range_only_touches_slice():
    set_all_seeds(42)
    pool = AGLNetworkPool(_built_trainer(), num_networks=4)
    untouched_before = _fo_state(pool.cells[3])
    trained_before = _fo_state(pool.cells[0])
    pool.train_range(0, 2, n_epochs=1)
    assert _equal(pool.cells[3].first_order.state_dict(), untouched_before), (
        "out-of-range cell changed"
    )
    assert not _equal(pool.cells[0].first_order.state_dict(), trained_before), (
        "in-range cell unchanged"
    )


def test_evaluate_returns_tiered_metrics():
    set_all_seeds(42)
    pool = AGLNetworkPool(_built_trainer(), num_networks=4)
    pool.train_range(0, 2, n_epochs=2)  # High tier
    pool.train_range(2, 4, n_epochs=1)  # Low tier
    result = pool.evaluate()
    assert set(result) == {"high", "low", "overall"}
    for tier in result.values():
        for key in (
            "precision_1st",
            "wager_precision",
            "wager_recall",
            "wager_f1",
            "wager_accuracy",
        ):
            assert key in tier
            assert f"{key}_std" in tier
            assert 0.0 <= tier[key] <= 1.0
