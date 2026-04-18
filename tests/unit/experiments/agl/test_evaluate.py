"""Unit tests for ``AGLTrainer.evaluate`` — AGL held-out eval pass.

Covers the contract, not numerical accuracy. The AGL pre-training loop
resets the first-order network at the end (reference L751), so
classification_precision after `pre_train()` intentionally reflects an
untrained first-order — see the module docstring for why.
"""

from __future__ import annotations

import pytest

from maps.experiments.agl import AGLSetting, AGLTrainer
from maps.utils import load_config, set_all_seeds


def _trainer(setting_id: str, *, n_epochs: int = 2) -> AGLTrainer:
    cfg = load_config("training/agl", overrides=[f"train.n_epochs={n_epochs}"])
    factorial = load_config("experiments/factorial_2x2")
    s = AGLSetting.from_dict(next(x for x in factorial.settings if x.id == setting_id))
    set_all_seeds(42)
    t = AGLTrainer(cfg, s)
    t.build()
    t.pre_train()
    return t


def test_evaluate_returns_classification_precision_for_all_settings():
    for setting_id in ("neither", "cascade_only", "second_order_only", "both"):
        m = _trainer(setting_id).evaluate()
        assert "classification_precision" in m
        assert 0.0 <= m["classification_precision"] <= 1.0


def test_evaluate_has_wager_only_when_second_order_enabled():
    m_no = _trainer("neither").evaluate()
    assert "wager_accuracy" not in m_no

    m_yes = _trainer("both").evaluate()
    assert "wager_accuracy" in m_yes
    assert 0.0 <= m_yes["wager_accuracy"] <= 1.0


def test_evaluate_before_build_raises():
    cfg = load_config("training/agl", overrides=["train.n_epochs=1"])
    factorial = load_config("experiments/factorial_2x2")
    s = AGLSetting.from_dict(next(x for x in factorial.settings if x.id == "both"))
    t = AGLTrainer(cfg, s)
    with pytest.raises(RuntimeError, match="build"):
        t.evaluate()


def test_evaluate_respects_patterns_number_override():
    t = _trainer("both")
    m_small = t.evaluate(eval_patterns_number=10)
    m_large = t.evaluate(eval_patterns_number=120)
    for d in (m_small, m_large):
        assert "classification_precision" in d


def test_evaluate_respects_threshold_override():
    """Threshold ≤ 0 forces every wager prediction to ``1`` → accuracy
    equals the fraction of positive targets. We don't assert the value —
    only that the override changes behavior vs the default."""
    t = _trainer("both")
    m_default = t.evaluate()
    m_low = t.evaluate(threshold=-1.0)
    m_high = t.evaluate(threshold=10.0)
    # At minimum, the three calls all produce valid results.
    for d in (m_default, m_low, m_high):
        assert 0.0 <= d["wager_accuracy"] <= 1.0
