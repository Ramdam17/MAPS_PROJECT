"""Integration smoke test for AGL pre-training.

Runs each of the 4 factorial settings for a handful of epochs and asserts:

- no crash / no NaN
- returned loss arrays have the expected shape
- the second-order network's wagering weights have moved when meta=True
  (the reference resets first-order at end of pre_train, so we check
  the second-order side instead for evidence of training)

This is *not* a correctness test — see tests/parity/ for that.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from maps.experiments.agl import AGLSetting, AGLTrainer
from maps.utils import load_config, set_all_seeds

SETTINGS = [
    AGLSetting(id="neither", label="vanilla", cascade=False, second_order=False),
    AGLSetting(id="cascade_only", label="c", cascade=True, second_order=False),
    AGLSetting(id="second_order_only", label="so", cascade=False, second_order=True),
    AGLSetting(id="both", label="full", cascade=True, second_order=True),
]


@pytest.fixture
def tiny_cfg():
    """Small config for fast smoke runs (10 epochs, batch 20)."""
    return load_config(
        "training/agl",
        overrides=[
            "train.n_epochs_pretrain=10",
            "train.batch_size_pretrain=20",
            "cascade.n_iterations=5",
            "cascade.alpha=0.2",
        ],
    )


@pytest.mark.parametrize("setting", SETTINGS, ids=lambda s: s.id)
def test_agl_trainer_runs_all_settings(tiny_cfg, setting):
    set_all_seeds(42)
    trainer = AGLTrainer(tiny_cfg, setting)
    trainer.build()

    initial_wager_w = trainer.second_order.wagering_head.wager.weight.detach().clone()

    losses_1, losses_2 = trainer.pre_train()

    # Shape.
    assert losses_1.shape == (tiny_cfg.train.n_epochs_pretrain,)
    assert losses_2.shape == (tiny_cfg.train.n_epochs_pretrain,)

    # No NaNs.
    assert np.all(np.isfinite(losses_1)), f"losses_1 has non-finite values: {losses_1}"
    assert np.all(np.isfinite(losses_2)), f"losses_2 has non-finite values: {losses_2}"

    # losses_2 should be zero iff second_order is off.
    if setting.second_order:
        assert losses_2.max() > 0, "second_order=True but losses_2 are all zero"
    else:
        assert np.allclose(losses_2, 0.0), "second_order=False but losses_2 are non-zero"

    # If second_order is on, wager weights should have moved.
    # (First-order weights are reset to initial at end of pre_train — reference behavior —
    #  so they won't differ from the captured initial state.)
    final_wager_w = trainer.second_order.wagering_head.wager.weight.detach()
    if setting.second_order:
        assert not torch.allclose(initial_wager_w, final_wager_w), (
            "wager weights unchanged despite second_order=True"
        )

    # Explicit: first-order weights MUST be back to initial (reference L751).
    final_fo_w = trainer.first_order.fc1.weight.detach()
    initial_fo_w = trainer._initial_first_order_state["fc1.weight"]
    assert torch.allclose(final_fo_w, initial_fo_w), (
        "first-order weights not restored to initial state — reference behavior broken"
    )
