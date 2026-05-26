"""Unit tests for :mod:`maps.domains.blindsight.trainer`."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from maps.domains.blindsight.trainer import (
    SETTINGS_REGISTRY,
    BlindsightTrainer,
    EvalMetrics,
    TrainingMetrics,
    _build_optimizer,
)
from maps.utils.config import load_config

# ---------------------------------------------------------------------------
# Fixtures — small config that runs fast
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg() -> DictConfig:
    """Composed + resolved Blindsight cfg with reduced epochs / batch for speed."""
    training_cfg = load_config("domains/blindsight/training")
    env_cfg = load_config("domains/blindsight/env", resolve=False)
    merged = OmegaConf.merge(training_cfg, env_cfg)
    OmegaConf.resolve(merged)
    # Shrink for unit-test speed
    merged.train.n_epochs = 2
    merged.train.batch_size = 20
    merged.eval.patterns_number = 20
    return merged  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Settings registry
# ---------------------------------------------------------------------------


def test_settings_registry_has_six_entries() -> None:
    assert len(SETTINGS_REGISTRY) == 6


def test_settings_registry_keys_canonical() -> None:
    expected = {
        "setting-1-baseline",
        "setting-2-cascade-1st",
        "setting-3-second-order-only",
        "setting-4-maps-1st",
        "setting-5-cascade-2nd",
        "setting-6-full-maps",
    }
    assert set(SETTINGS_REGISTRY) == expected


def test_setting_dataclass_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    s = SETTINGS_REGISTRY["setting-6-full-maps"]
    with pytest.raises(FrozenInstanceError):
        s.cascade_1st = False  # type: ignore[misc]


def test_setting_6_is_full_maps() -> None:
    s = SETTINGS_REGISTRY["setting-6-full-maps"]
    assert s.cascade_1st and s.cascade_2nd and s.second_order


def test_setting_1_baseline_has_no_features() -> None:
    s = SETTINGS_REGISTRY["setting-1-baseline"]
    assert not s.cascade_1st
    assert not s.cascade_2nd
    assert not s.second_order


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


def test_build_optimizer_dispatches_adamax(small_cfg: DictConfig) -> None:
    import torch

    params = [torch.nn.Parameter(torch.randn(3))]
    opt = _build_optimizer("ADAMAX", params, lr=0.5)
    assert opt.__class__.__name__ == "Adamax"


def test_build_optimizer_unknown_raises() -> None:
    import torch

    params = [torch.nn.Parameter(torch.randn(3))]
    with pytest.raises(ValueError, match="unknown optimizer"):
        _build_optimizer("MyCustomOpt", params, lr=0.1)


# ---------------------------------------------------------------------------
# BlindsightTrainer.build()
# ---------------------------------------------------------------------------


def test_build_constructs_networks_and_optimizers(small_cfg: DictConfig) -> None:
    trainer = BlindsightTrainer(
        setting=SETTINGS_REGISTRY["setting-6-full-maps"],
        seed=42,
        cfg=small_cfg,
    )
    trainer.build()
    assert trainer.first_order is not None
    assert trainer.second_order is not None
    assert trainer.optimizer_1 is not None
    assert trainer.optimizer_2 is not None
    assert trainer.scheduler_1 is not None
    assert trainer.scheduler_2 is not None


def test_train_without_build_raises(small_cfg: DictConfig) -> None:
    trainer = BlindsightTrainer(
        setting=SETTINGS_REGISTRY["setting-6-full-maps"],
        seed=42,
        cfg=small_cfg,
    )
    with pytest.raises(RuntimeError, match=r"call \.build"):
        trainer.train()


# ---------------------------------------------------------------------------
# Cascade dispatch per setting
# ---------------------------------------------------------------------------


def test_cascade_params_baseline(small_cfg: DictConfig) -> None:
    """Setting 1 (no cascade) → rate=1, iters=1 on both."""
    trainer = BlindsightTrainer(SETTINGS_REGISTRY["setting-1-baseline"], 42, small_cfg)
    r1, i1, r2, i2 = trainer._cascade_params()
    assert (r1, i1, r2, i2) == (1.0, 1, 1.0, 1)


def test_cascade_params_cascade_1st_only(small_cfg: DictConfig) -> None:
    """Setting 2 → cascade on 1st only."""
    trainer = BlindsightTrainer(SETTINGS_REGISTRY["setting-2-cascade-1st"], 42, small_cfg)
    r1, i1, r2, i2 = trainer._cascade_params()
    assert r1 == 0.02 and i1 == 50
    assert r2 == 1.0 and i2 == 1


def test_cascade_params_cascade_2nd_only(small_cfg: DictConfig) -> None:
    """Setting 5 → cascade on 2nd only."""
    trainer = BlindsightTrainer(SETTINGS_REGISTRY["setting-5-cascade-2nd"], 42, small_cfg)
    r1, i1, r2, i2 = trainer._cascade_params()
    assert r1 == 1.0 and i1 == 1
    assert r2 == 0.02 and i2 == 50


def test_cascade_params_full_maps(small_cfg: DictConfig) -> None:
    """Setting 6 → cascade on both."""
    trainer = BlindsightTrainer(SETTINGS_REGISTRY["setting-6-full-maps"], 42, small_cfg)
    r1, i1, r2, i2 = trainer._cascade_params()
    assert r1 == 0.02 and i1 == 50
    assert r2 == 0.02 and i2 == 50


# ---------------------------------------------------------------------------
# Train smoke
# ---------------------------------------------------------------------------


def test_train_smoke_setting_6(small_cfg: DictConfig) -> None:
    """Full MAPS, 2 epochs, batch=20 — should run without error and
    produce finite losses."""
    trainer = BlindsightTrainer(SETTINGS_REGISTRY["setting-6-full-maps"], 42, small_cfg)
    trainer.build()
    metrics = trainer.train(n_epochs=2)
    assert isinstance(metrics, TrainingMetrics)
    assert metrics.losses_1.shape == (2,)
    assert metrics.losses_2.shape == (2,)
    assert np.isfinite(metrics.losses_1).all()
    assert np.isfinite(metrics.losses_2).all()
    # Loss_2 should be > 0 for setting 6 (second_order trained)
    assert metrics.losses_2[0] > 0


def test_train_smoke_setting_1_no_loss_2(small_cfg: DictConfig) -> None:
    """Baseline (no second_order) → losses_2 stays zero."""
    trainer = BlindsightTrainer(SETTINGS_REGISTRY["setting-1-baseline"], 42, small_cfg)
    trainer.build()
    metrics = trainer.train(n_epochs=2)
    assert (metrics.losses_2 == 0).all()
    # But losses_1 must still be > 0
    assert (metrics.losses_1 > 0).all()


def test_train_simclr_dispatch_runs(small_cfg: DictConfig) -> None:
    """first_order_loss.kind='simclr' (D12.4 augmentation) trains
    without crash. Loss may differ in scale from CAE but must be finite."""
    small_cfg.first_order_loss.kind = "simclr"
    trainer = BlindsightTrainer(SETTINGS_REGISTRY["setting-6-full-maps"], 42, small_cfg)
    trainer.build()
    metrics = trainer.train(n_epochs=2)
    assert np.isfinite(metrics.losses_1).all()


def test_train_invalid_loss_kind_raises(small_cfg: DictConfig) -> None:
    small_cfg.first_order_loss.kind = "unknown_loss"
    trainer = BlindsightTrainer(SETTINGS_REGISTRY["setting-6-full-maps"], 42, small_cfg)
    trainer.build()
    with pytest.raises(ValueError, match=r"first_order_loss\.kind"):
        trainer.train(n_epochs=1)


# ---------------------------------------------------------------------------
# Evaluate smoke
# ---------------------------------------------------------------------------


def test_evaluate_smoke_returns_3_conditions(small_cfg: DictConfig) -> None:
    trainer = BlindsightTrainer(SETTINGS_REGISTRY["setting-6-full-maps"], 42, small_cfg)
    trainer.build()
    trainer.train(n_epochs=2)  # need some training for non-trivial metrics
    eval_metrics = trainer.evaluate()
    assert isinstance(eval_metrics, EvalMetrics)
    assert set(eval_metrics.discrimination_accuracy) == {
        "superthreshold",
        "subthreshold",
        "low_vision",
    }
    assert set(eval_metrics.wager_accuracy) == {"superthreshold", "subthreshold", "low_vision"}
    # All metrics in [0, 1]
    for v in eval_metrics.discrimination_accuracy.values():
        assert 0 <= v <= 1
    for v in eval_metrics.wager_accuracy.values():
        assert 0 <= v <= 1
