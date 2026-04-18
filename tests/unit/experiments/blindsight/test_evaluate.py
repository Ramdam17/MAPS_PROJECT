"""Unit tests for ``BlindsightTrainer.evaluate`` — the held-out eval pass.

Covers the contract (shape of returned dict, threshold plumbing from config,
behavior when ``second_order=False``) rather than numerical accuracy
(that needs a full 200-epoch run and is covered by the Sprint 06 grid).
"""

from __future__ import annotations

import math

import pytest

from maps.experiments.blindsight import BlindsightSetting, BlindsightTrainer
from maps.experiments.blindsight.data import StimulusCondition
from maps.utils import load_config, set_all_seeds


def _trainer(setting_id: str, *, n_epochs: int = 2) -> BlindsightTrainer:
    cfg = load_config("training/blindsight", overrides=[f"train.n_epochs={n_epochs}"])
    factorial = load_config("experiments/factorial_2x2")
    s = BlindsightSetting.from_dict(next(x for x in factorial.settings if x.id == setting_id))
    set_all_seeds(42)
    t = BlindsightTrainer(cfg, s)
    t.build()
    t.pre_train()
    return t


def test_evaluate_returns_all_three_conditions():
    t = _trainer("both")
    metrics = t.evaluate()
    assert set(metrics.keys()) == {"superthreshold", "subthreshold", "low_vision"}


def test_evaluate_full_maps_has_discrimination_and_wager():
    t = _trainer("both")
    m = t.evaluate()["superthreshold"]
    assert "discrimination_accuracy" in m
    assert "wager_accuracy" in m
    assert 0.0 <= m["discrimination_accuracy"] <= 1.0
    assert 0.0 <= m["wager_accuracy"] <= 1.0


def test_evaluate_without_second_order_has_no_wager_key():
    t = _trainer("neither")
    m = t.evaluate()["superthreshold"]
    assert "discrimination_accuracy" in m
    assert "wager_accuracy" not in m


def test_evaluate_before_build_raises():
    cfg = load_config("training/blindsight", overrides=["train.n_epochs=1"])
    factorial = load_config("experiments/factorial_2x2")
    s = BlindsightSetting.from_dict(next(x for x in factorial.settings if x.id == "both"))
    t = BlindsightTrainer(cfg, s)
    with pytest.raises(RuntimeError, match="build"):
        t.evaluate()


def test_evaluate_respects_patterns_number_override():
    t = _trainer("both")
    m_small = t.evaluate(eval_patterns_number=40)
    m_large = t.evaluate(eval_patterns_number=400)
    # Both return valid accuracies; override is accepted without error.
    for d in (m_small, m_large):
        for cond in d.values():
            assert not math.isnan(cond["discrimination_accuracy"])


def test_evaluate_custom_condition_subset():
    t = _trainer("both")
    m = t.evaluate(conditions=(StimulusCondition.SUPERTHRESHOLD,))
    assert list(m.keys()) == ["superthreshold"]
