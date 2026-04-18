"""Unit tests for SARL+CL dynamic loss weighting helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from maps.experiments.sarl_cl.loss_weighting import (
    DynamicLossWeighter,
    individual_losses,
    min_max_norm,
    update_moving_average,
)

# ── update_moving_average ──────────────────────────────────────────────────


def test_update_moving_average_formula():
    """EMA: new = m·old + (1-m)·value."""
    assert update_moving_average(1.0, 3.0, momentum=0.9) == pytest.approx(1.2)
    assert update_moving_average(0.0, 10.0, momentum=0.5) == pytest.approx(5.0)


def test_update_moving_average_works_with_tensors():
    old = torch.tensor(2.0)
    new = torch.tensor(4.0)
    result = update_moving_average(old, new, momentum=0.5)
    assert torch.allclose(result, torch.tensor(3.0))


# ── min_max_norm ───────────────────────────────────────────────────────────


def test_min_max_norm_divides_mean_by_max():
    """Paper quirk: returns ``(mean / max(values), max(values))`` — not true min-max."""
    scaled, max_val = min_max_norm(2.0, [1.0, 4.0, 2.0])
    assert scaled == pytest.approx(0.5)
    assert max_val == pytest.approx(4.0)


def test_min_max_norm_with_numpy_array():
    scaled, max_val = min_max_norm(np.float64(3.0), np.array([1.0, 6.0, 3.0]))
    assert scaled == pytest.approx(0.5)
    assert max_val == pytest.approx(6.0)


# ── individual_losses ──────────────────────────────────────────────────────


def test_individual_losses_returns_95th_percentile():
    """Per-sample losses → scalar 95th percentile."""
    output = torch.arange(10, dtype=torch.float32).unsqueeze(1)  # values 0..9
    target = torch.zeros(10, 1)
    # MSE per sample = i² for sample i. 95th percentile over [0, 1, 4, ..., 81].
    p95 = individual_losses(output, target, loss_fn=F.mse_loss)
    expected = float(np.percentile(np.arange(10) ** 2, 95))
    assert p95 == pytest.approx(expected)


def test_individual_losses_handles_small_batch():
    output = torch.tensor([[1.0], [2.0]])
    target = torch.tensor([[0.0], [0.0]])
    p95 = individual_losses(output, target, loss_fn=F.mse_loss)
    expected = float(np.percentile([1.0, 4.0], 95))
    assert p95 == pytest.approx(expected)


# ── DynamicLossWeighter ────────────────────────────────────────────────────


def _fresh_weighter() -> DynamicLossWeighter:
    return DynamicLossWeighter(update_interval=10)


def test_weighter_initial_state():
    w = _fresh_weighter()
    assert set(w.keys) == {"task", "distillation", "feature"}
    assert all(w.historical_max[k] == float("-inf") for k in w.keys)
    assert w.steps == 0


def test_weighter_update_tracks_running_max():
    w = _fresh_weighter()
    w.update({"task": 1.0, "distillation": 0.5, "feature": 2.0})
    w.update({"task": 0.3, "distillation": 3.0, "feature": 1.0})
    w.update({"task": 2.5, "distillation": 1.0, "feature": 0.1})
    assert w.historical_max["task"] == pytest.approx(2.5)
    assert w.historical_max["distillation"] == pytest.approx(3.0)
    assert w.historical_max["feature"] == pytest.approx(2.0)


def test_weighter_update_accepts_tensors_and_detaches():
    """``update`` must consume tensors without tripping autograd."""
    w = _fresh_weighter()
    losses = {
        "task": torch.tensor(1.0, requires_grad=True),
        "distillation": torch.tensor(2.0, requires_grad=True),
        "feature": torch.tensor(3.0, requires_grad=True),
    }
    # Should not raise, should not keep graph references on the stored floats.
    w.update(losses)
    assert isinstance(w.moving_avgs["task"], float)
    assert w.historical_max["feature"] == pytest.approx(3.0)


def test_weighter_weight_losses_divides_by_running_max():
    w = _fresh_weighter()
    w.update({"task": 2.0, "distillation": 4.0, "feature": 8.0})
    out = w.weight_losses({"task": 1.0, "distillation": 2.0, "feature": 4.0})
    # 1/2, 2/4, 4/8 → all 0.5 (modulo the 1e-16 epsilon, negligible).
    for key in out:
        assert out[key] == pytest.approx(0.5, abs=1e-10)


def test_weighter_weight_losses_preserves_tensor_type():
    """Tensor inputs come back as tensors (so autograd can flow)."""
    w = _fresh_weighter()
    w.update({"task": 2.0, "distillation": 4.0, "feature": 8.0})
    losses = {
        "task": torch.tensor(1.0, requires_grad=True),
        "distillation": torch.tensor(2.0, requires_grad=True),
        "feature": torch.tensor(4.0, requires_grad=True),
    }
    weighted = w.weight_losses(losses)
    assert isinstance(weighted["task"], torch.Tensor)
    # Gradient should flow back to the input tensor.
    weighted["task"].backward()
    assert losses["task"].grad is not None


def test_weighter_epsilon_prevents_division_by_zero():
    w = _fresh_weighter()
    # historical_max starts at -inf; after a single zero-update it stays 0.
    w.update({"task": 0.0, "distillation": 0.0, "feature": 0.0})
    out = w.weight_losses({"task": 1.0, "distillation": 1.0, "feature": 1.0})
    # 1 / (0 + 1e-16) is finite and very large — confirms no ZeroDivisionError.
    assert all(np.isfinite(v) for v in out.values())


def test_weighter_mid_interval_snapshot_records_previous_max():
    """At step == interval // 2, the current value is copied into historical_max_prev."""
    w = DynamicLossWeighter(update_interval=4)  # mid = step 2
    w.update({"task": 1.0, "distillation": 1.0, "feature": 1.0})  # step 1
    w.update({"task": 5.0, "distillation": 5.0, "feature": 5.0})  # step 2 (mid)
    assert w.historical_max_prev["task"] == pytest.approx(5.0)
    assert w.historical_max_prev["distillation"] == pytest.approx(5.0)
    assert w.historical_max_prev["feature"] == pytest.approx(5.0)


def test_weighter_get_stats_snapshot_is_decoupled():
    """``get_stats`` returns copies, not live references."""
    w = _fresh_weighter()
    w.update({"task": 1.0, "distillation": 2.0, "feature": 3.0})
    stats = w.get_stats()
    stats["moving_averages"]["task"] = 999.0
    # Internal state should be untouched.
    assert w.moving_avgs["task"] == pytest.approx(1.0)


def test_weighter_steps_increment_monotonically():
    w = _fresh_weighter()
    for i in range(1, 5):
        w.update({"task": float(i), "distillation": float(i), "feature": float(i)})
        assert w.steps == i


def test_weighter_custom_keys():
    """Can be parametrized with different key names for ablation scripts."""
    w = DynamicLossWeighter(keys=("a", "b", "c"))
    w.update({"a": 1.0, "b": 2.0, "c": 3.0})
    weighted = w.weight_losses({"a": 0.5, "b": 1.0, "c": 1.5})
    for key in ("a", "b", "c"):
        assert weighted[key] == pytest.approx(0.5, abs=1e-10)
