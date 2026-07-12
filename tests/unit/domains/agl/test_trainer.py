"""Unit tests for :class:`maps.domains.agl.trainer.AGLTrainer` (Sprint 13.D).

Covers the load-bearing AGL mechanics:
- build() instantiates nets/optimizers/schedulers and caches the initial
  1st-order weights;
- **the reset** (D-agl-reset): after pre_train the 1st-order weights are
  bit-exactly restored to their initial values;
- the two-loss pattern trains the 2nd-order during pre-train (meta setting);
- baseline (no 2nd-order) leaves the 2nd-order untouched during pre-train;
- Grammar-A training() keeps the 2nd-order **frozen** (source L969 meta=False)
  while the 1st-order changes.
"""

from __future__ import annotations

import copy

import pytest
import torch

from maps.domains.agl.trainer import SETTINGS_REGISTRY, AGLTrainer
from maps.utils.config import load_config
from maps.utils.seeding import set_all_seeds


def _cfg():
    return load_config(
        "domains/agl/training",
        overrides=[
            "train.n_epochs_pretrain=2",
            "train.batch_size_pretrain=8",
            "train.batch_size_training=8",
        ],
    )


def _weights_equal(sd_a, sd_b) -> bool:
    return all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)


def _weights_changed(sd_a, sd_b) -> bool:
    return any(not torch.equal(sd_a[k], sd_b[k]) for k in sd_a)


def test_build_creates_components_and_caches_initial():
    set_all_seeds(42)
    t = AGLTrainer(SETTINGS_REGISTRY["setting-1-baseline"], seed=42, cfg=_cfg())
    t.build()
    assert t.first_order is not None
    assert t.second_order is not None
    assert t.optimizer_1 is not None and t.optimizer_2 is not None
    assert t.scheduler_1 is not None and t.scheduler_2 is not None
    assert t._initial_first_order_state is not None


def test_pretrain_resets_first_order_bit_exact():
    """D-agl-reset: 1st-order returns to initial weights after pre-train."""
    set_all_seeds(42)
    t = AGLTrainer(SETTINGS_REGISTRY["setting-3-second-order-only"], seed=42, cfg=_cfg())
    t.build()
    initial = copy.deepcopy(t.first_order.state_dict())
    t.pre_train()
    assert _weights_equal(t.first_order.state_dict(), initial), "1st-order not reset to initial"


def test_pretrain_trains_second_order_when_meta():
    """Two-loss pattern updates the 2nd-order during pre-train (meta on)."""
    set_all_seeds(42)
    t = AGLTrainer(SETTINGS_REGISTRY["setting-3-second-order-only"], seed=42, cfg=_cfg())
    t.build()
    before = copy.deepcopy(t.second_order.state_dict())
    t.pre_train()
    assert _weights_changed(t.second_order.state_dict(), before), "2nd-order did not train"


def test_pretrain_baseline_leaves_second_order_untouched():
    """No 2nd-order training when setting.second_order is False."""
    set_all_seeds(42)
    t = AGLTrainer(SETTINGS_REGISTRY["setting-1-baseline"], seed=42, cfg=_cfg())
    t.build()
    before = copy.deepcopy(t.second_order.state_dict())
    t.pre_train()
    assert _weights_equal(t.second_order.state_dict(), before), "2nd-order changed without meta"


def test_pretrain_returns_per_epoch_metrics():
    set_all_seeds(42)
    t = AGLTrainer(SETTINGS_REGISTRY["setting-3-second-order-only"], seed=42, cfg=_cfg())
    t.build()
    losses_1, losses_2, precision = t.pre_train()
    assert losses_1.shape == losses_2.shape == precision.shape == (2,)
    assert (losses_2 > 0).all()  # BCE-sum wager loss is recorded (meta on)


def test_training_freezes_second_order_updates_first():
    """Grammar-A training keeps the 2nd-order frozen (meta forced False, L969)."""
    set_all_seeds(42)
    t = AGLTrainer(SETTINGS_REGISTRY["setting-3-second-order-only"], seed=42, cfg=_cfg())
    t.build()
    t.pre_train()  # includes the reset
    so_before = copy.deepcopy(t.second_order.state_dict())
    fo_before = copy.deepcopy(t.first_order.state_dict())
    t.training(n_epochs=2)
    assert _weights_equal(t.second_order.state_dict(), so_before), (
        "2nd-order not frozen in training"
    )
    assert _weights_changed(t.first_order.state_dict(), fo_before), "1st-order did not train"


def test_cascade_setting_runs():
    """A cascade setting (50 iterations) runs end-to-end without error."""
    set_all_seeds(42)
    cfg = load_config(
        "domains/agl/training",
        overrides=["train.n_epochs_pretrain=1", "train.batch_size_pretrain=4"],
    )
    t = AGLTrainer(SETTINGS_REGISTRY["setting-6-full-maps"], seed=42, cfg=cfg)
    t.build()
    losses_1, _, _ = t.pre_train()
    assert losses_1.shape == (1,)


def test_ensure_built_guard():
    t = AGLTrainer(SETTINGS_REGISTRY["setting-1-baseline"], seed=42, cfg=_cfg())
    with pytest.raises(RuntimeError, match="build"):
        t.pre_train()
