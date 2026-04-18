"""Integration smoke test for ``maps.experiments.sarl.training_loop.run_training``.

Runs an abbreviated DQN training cell on MinAtar and checks the
pipeline end-to-end:

* Buffer fills past ``replay_start_size``.
* Policy-net weights diverge from their initial state.
* At least one gradient update happens.
* Metrics are populated with finite values.
* With ``meta=True`` the second-order network also trains.

This is **not** a correctness test — per-update parity is covered in
``tests/parity/sarl/test_tier3_update.py``. We deliberately use a tiny
``num_frames`` so the test finishes in a few seconds on a Mac CPU.

MinAtar is an optional dependency (``uv sync --extra sarl``). The tests
auto-skip when the import fails.
"""

from __future__ import annotations

import copy

import pytest
import torch

pytest.importorskip("minatar", reason="requires `uv sync --extra sarl`")

from minatar import Environment

from maps.experiments.sarl.training_loop import (
    SarlTrainingConfig,
    run_training,
    setting_to_config,
)
from maps.utils import set_all_seeds

# Frame counts chosen so the buffer just passes replay_start_size and we get
# a handful of gradient updates — runs in <10s on CPU.
NUM_FRAMES = 600
REPLAY_START = 200
BATCH_SIZE = 32
SEED = 2026


def _make_cfg(setting: int) -> SarlTrainingConfig:
    base = SarlTrainingConfig(
        game="space_invaders",
        seed=SEED,
        num_frames=NUM_FRAMES,
        batch_size=BATCH_SIZE,
        replay_buffer_size=2_000,
        replay_start_size=REPLAY_START,
        training_freq=1,
        target_update_freq=50,
        scheduler_period=50,
        validation_every_episodes=10_000,  # effectively disable validation
        validation_iterations=1,
        device="cpu",
    )
    return setting_to_config(setting, base)


def _weights_diverged(before: torch.nn.Module, after: torch.nn.Module) -> bool:
    """True when at least one named parameter changed numerically."""
    b = dict(before.named_parameters())
    for name, p_after in after.named_parameters():
        if not torch.allclose(b[name], p_after.detach().cpu(), atol=1e-8):
            return True
    return False


@pytest.mark.slow
def test_setting1_vanilla_dqn_trains():
    """Setting 1: no cascade, no meta. Verifies the base DQN loop trains."""
    set_all_seeds(SEED)
    env = Environment("space_invaders")
    cfg = _make_cfg(setting=1)
    assert cfg.meta is False
    assert cfg.cascade_iterations_1 == 1
    assert cfg.cascade_iterations_2 == 1

    # Snapshot policy-net init weights by building a twin net with same seed.
    from maps.experiments.sarl.model import SarlQNetwork

    torch.manual_seed(SEED)
    initial_policy = SarlQNetwork(env.state_shape()[-1], env.num_actions())

    policy_net, second_order_net, metrics = run_training(env, cfg)

    assert second_order_net is None
    assert metrics.total_frames == NUM_FRAMES
    assert metrics.total_updates > 0, "training loop should have stepped at least once"
    assert metrics.wall_time_seconds > 0

    # Loss values that were recorded are finite.
    for loss in metrics.episode_losses_first:
        if loss is not None:
            assert torch.isfinite(torch.tensor(loss))

    # Weights moved away from init.
    assert _weights_diverged(initial_policy, policy_net)


@pytest.mark.slow
def test_setting6_full_maps_trains_both_networks():
    """Setting 6 (full MAPS): meta + cascade_1=50 + cascade_2=50.

    Keeps cascade iterations high (matches the paper) so we actually exercise
    the cross-gradient path through ``loss_second.backward(retain_graph=True)``.
    """
    set_all_seeds(SEED)
    env = Environment("space_invaders")
    cfg = _make_cfg(setting=6)
    assert cfg.meta is True
    assert cfg.cascade_iterations_1 == 50
    assert cfg.cascade_iterations_2 == 50

    # Cascade=50 forward passes are expensive; keep this cell even tinier.
    cfg = SarlTrainingConfig(
        **{
            **cfg.__dict__,
            "num_frames": 300,
            "replay_start_size": 100,
            "batch_size": 16,
        }
    )

    from maps.experiments.sarl.model import SarlQNetwork, SarlSecondOrderNetwork

    torch.manual_seed(SEED)
    init_policy = SarlQNetwork(env.state_shape()[-1], env.num_actions())
    SarlQNetwork(env.state_shape()[-1], env.num_actions())  # target init slot
    init_second = SarlSecondOrderNetwork(env.state_shape()[-1])
    init_policy_clone = copy.deepcopy(init_policy)
    init_second_clone = copy.deepcopy(init_second)

    policy_net, second_order_net, metrics = run_training(env, cfg)

    assert second_order_net is not None
    assert metrics.total_updates > 0
    # Both networks must have moved.
    assert _weights_diverged(init_policy_clone, policy_net)
    assert _weights_diverged(init_second_clone, second_order_net)

    # Episode-level second-order loss is recorded whenever an update fired.
    any_loss2 = any(loss is not None for loss in metrics.episode_losses_second)
    assert any_loss2, "meta=True runs must record second-order loss"


@pytest.mark.slow
def test_output_dir_persists_weights_and_metrics(tmp_path):
    """When ``output_dir`` is set, policy_net.pt + metrics.json are written."""
    set_all_seeds(SEED)
    env = Environment("space_invaders")
    cfg = _make_cfg(setting=1)
    cfg = SarlTrainingConfig(**{**cfg.__dict__, "output_dir": tmp_path})

    run_training(env, cfg)

    assert (tmp_path / "policy_net.pt").exists()
    assert (tmp_path / "metrics.json").exists()
    # second_order_net.pt must NOT exist when meta=False
    assert not (tmp_path / "second_order_net.pt").exists()

    import json

    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["game"] == "space_invaders"
    assert payload["meta"] is False
    assert payload["total_frames"] == NUM_FRAMES
    assert isinstance(payload["episode_returns"], list)


def test_setting_to_config_table_completeness():
    """All 6 paper settings must resolve to distinct (meta, cascade_1, cascade_2)."""
    seen = set()
    for s in range(1, 7):
        cfg = setting_to_config(s)
        seen.add((cfg.meta, cfg.cascade_iterations_1, cfg.cascade_iterations_2))
    assert len(seen) == 6, f"settings must yield distinct triples, got {seen}"


def test_setting_to_config_rejects_out_of_range():
    with pytest.raises(ValueError):
        setting_to_config(0)
    with pytest.raises(ValueError):
        setting_to_config(7)


def test_setting_to_config_preserves_base_fields():
    base = SarlTrainingConfig(game="breakout", seed=99, num_frames=1234)
    cfg = setting_to_config(6, base)
    assert cfg.game == "breakout"
    assert cfg.seed == 99
    assert cfg.num_frames == 1234
    # And setting-6 knobs are applied
    assert cfg.meta is True
    assert cfg.cascade_iterations_1 == 50
    assert cfg.cascade_iterations_2 == 50
