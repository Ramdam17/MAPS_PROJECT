"""Integration smoke test for Blindsight pre-training.

Runs each of the 4 factorial settings for a handful of epochs and asserts:

- no crash / no NaN
- returned loss arrays have the expected shape
- the first-order network's weights have moved (i.e. actual training happened)

This is *not* a correctness test — see tests/parity/ for that.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from maps.experiments.blindsight import BlindsightSetting, BlindsightTrainer
from maps.utils import load_config, set_all_seeds

SETTINGS = [
    BlindsightSetting(id="neither", label="vanilla", cascade=False, second_order=False),
    BlindsightSetting(id="cascade_only", label="c", cascade=True, second_order=False),
    BlindsightSetting(id="second_order_only", label="so", cascade=False, second_order=True),
    BlindsightSetting(id="both", label="full", cascade=True, second_order=True),
]


@pytest.fixture
def tiny_cfg():
    """Small config for fast smoke runs (10 epochs, batch 20)."""
    cfg = load_config(
        "training/blindsight",
        overrides=[
            "train.n_epochs=10",
            "train.batch_size=20",
            # Cap cascade iterations so `cascade=True` settings finish in seconds.
            "cascade.n_iterations=5",
            "cascade.alpha=0.2",
        ],
    )
    return cfg


@pytest.mark.parametrize("setting", SETTINGS, ids=lambda s: s.id)
def test_blindsight_trainer_runs_all_settings(tiny_cfg, setting):
    set_all_seeds(42)
    trainer = BlindsightTrainer(tiny_cfg, setting)
    trainer.build()

    initial_w = trainer.first_order.fc1.weight.detach().clone()

    losses_1, losses_2 = trainer.pre_train()

    # Shape.
    assert losses_1.shape == (tiny_cfg.train.n_epochs,)
    assert losses_2.shape == (tiny_cfg.train.n_epochs,)

    # No NaNs.
    assert np.all(np.isfinite(losses_1)), f"losses_1 has non-finite values: {losses_1}"
    assert np.all(np.isfinite(losses_2)), f"losses_2 has non-finite values: {losses_2}"

    # losses_2 should be zero iff second_order is off.
    if setting.second_order:
        assert losses_2.max() > 0, "second_order=True but losses_2 are all zero"
    else:
        assert np.allclose(losses_2, 0.0), "second_order=False but losses_2 are non-zero"

    # Weights moved — training actually happened.
    final_w = trainer.first_order.fc1.weight.detach()
    assert not torch.allclose(initial_w, final_w), "first_order weights unchanged after training"
