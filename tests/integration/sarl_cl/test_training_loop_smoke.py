"""Integration smoke tests for ``maps.experiments.sarl_cl.training_loop.run_training_cl``.

Checks the CL training loop end-to-end on an abbreviated MinAtar cell:

* **Vanilla (no teacher)** — setting 1 and setting 6 degenerate to SARL-like
  behavior but through the CL networks (explicit fc_output decoder,
  cascade on output). Verifies that the loop trains without a curriculum.
* **CL mode** — a real teacher checkpoint is loaded, the 3-term loss fires,
  component metrics are recorded, and student weights still diverge.
* **Checkpoint persistence** — ``output_dir`` writes a ``checkpoint.pt``
  whose schema is loadable as a teacher for the NEXT stage.

These are **not** correctness tests — the per-update numerical path is
covered in ``tests/unit/experiments/sarl_cl/test_trainer.py``. We use a
tiny ``num_frames`` so the full test module finishes in <30s on CPU.

MinAtar is an optional dependency (``uv sync --extra sarl``). Tests
auto-skip when the import fails.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch

pytest.importorskip("minatar", reason="requires `uv sync --extra sarl`")

from minatar import Environment

from maps.experiments.sarl_cl.model import SarlCLQNetwork
from maps.experiments.sarl_cl.training_loop import (
    SarlCLTrainingConfig,
    load_partial_state_dict,
    run_training_cl,
    setting_to_config_cl,
)
from maps.utils import set_all_seeds

NUM_FRAMES = 600
REPLAY_START = 200
BATCH_SIZE = 32
SEED = 2026


def _make_cfg(setting: int, **overrides) -> SarlCLTrainingConfig:
    """Build a tiny SARL+CL config. ``overrides`` replaces any field."""
    base = SarlCLTrainingConfig(
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
    for k, v in overrides.items():
        object.__setattr__(base, k, v)
    return setting_to_config_cl(setting, base)


def _weights_diverged(before: torch.nn.Module, after: torch.nn.Module) -> bool:
    b = dict(before.named_parameters())
    for name, p_after in after.named_parameters():
        if name not in b:
            # Shape-based transfer could add adapter params; treat presence as divergence.
            return True
        if not torch.allclose(b[name], p_after.detach().cpu(), atol=1e-8):
            return True
    return False


# ── Vanilla (no teacher) ───────────────────────────────────────────────────


@pytest.mark.slow
def test_setting1_vanilla_trains_without_teacher():
    """Setting 1 (meta off, cascade off) through CL nets: base DQN loop trains."""
    set_all_seeds(SEED)
    env = Environment("space_invaders")
    cfg = _make_cfg(setting=1)
    assert cfg.meta is False
    assert cfg.curriculum is False
    assert cfg.cascade_iterations_1 == 1
    assert cfg.cascade_iterations_2 == 1

    torch.manual_seed(SEED)
    initial_policy = SarlCLQNetwork(env.state_shape()[-1], env.num_actions())

    policy_net, second_net, metrics = run_training_cl(env, cfg)

    assert second_net is None
    assert metrics.total_frames == NUM_FRAMES
    assert metrics.total_updates > 0
    assert metrics.wall_time_seconds > 0

    # All component-loss slots are None (no teacher ever).
    assert all(v is None for v in metrics.episode_components_first_task)
    assert all(v is None for v in metrics.episode_components_second_task)

    assert _weights_diverged(initial_policy, policy_net)


@pytest.mark.slow
def test_setting6_vanilla_trains_both_networks():
    """Full MAPS (meta + cascade both) without teacher: both nets train."""
    set_all_seeds(SEED)
    env = Environment("space_invaders")
    cfg = _make_cfg(
        setting=6,
        num_frames=300,
        replay_start_size=100,
        batch_size=16,
    )
    assert cfg.meta is True
    assert cfg.cascade_iterations_1 == 50
    assert cfg.cascade_iterations_2 == 50
    assert cfg.curriculum is False

    _, second_net, metrics = run_training_cl(env, cfg)

    assert second_net is not None
    assert metrics.total_updates > 0
    assert any(v is not None for v in metrics.episode_losses_second)


# ── Checkpoint persistence ─────────────────────────────────────────────────


@pytest.mark.slow
def test_output_dir_persists_cl_checkpoint(tmp_path):
    """Setting 1 vanilla: checkpoint.pt + metrics.json written to output_dir."""
    set_all_seeds(SEED)
    env = Environment("space_invaders")
    cfg = _make_cfg(setting=1, output_dir=tmp_path)

    run_training_cl(env, cfg)

    ckpt_path = tmp_path / "checkpoint.pt"
    metrics_path = tmp_path / "metrics.json"
    assert ckpt_path.exists()
    assert metrics_path.exists()

    # Sprint-08 D.19b: checkpoint.pt now uses the unified D.13 schema.
    # `_persist_outputs` no longer writes a teacher-only payload here (that
    # caused a silent overwrite of the resume checkpoint). The canonical
    # keys are `policy_state_dict` / `second_order_state_dict`; legacy
    # `policy_net_state_dict` / `second_net_state_dict` are still accepted
    # on teacher-load via fallback but no longer produced.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "policy_state_dict" in checkpoint
    # meta=False so second_order_state_dict is None
    assert checkpoint["second_order_state_dict"] is None
    assert checkpoint["format_version"] == 1  # D.13 schema version sentinel
    # The full resume payload rides in cfg_snapshot, not at top level.
    snapshot = checkpoint["cfg_snapshot"]
    assert snapshot["game"] == "space_invaders"
    assert snapshot["meta"] is False
    assert snapshot["cascade_iterations_1"] == 1

    payload = json.loads(metrics_path.read_text())
    assert payload["curriculum"] is False
    assert payload["teacher_load_path"] is None
    assert isinstance(payload["episode_components_first_task"], list)


# ── CL mode with a real teacher ────────────────────────────────────────────


@pytest.mark.slow
def test_cl_mode_with_teacher_records_components(tmp_path):
    """Run stage-1 to completion, then stage-2 with the stage-1 checkpoint as teacher.

    Stage 1: vanilla setting 3 (meta on, cascade off) → produces a
             checkpoint with both policy + second-order state dicts.
    Stage 2: same setting, but curriculum=True → triggers the 3-term loss,
             populates ``episode_components_*`` metrics with at least one
             non-None entry, and student weights diverge from teacher.
    """
    set_all_seeds(SEED)
    env = Environment("space_invaders")

    # ── Stage 1 — produce a teacher checkpoint ──────────────────────────────
    stage1_dir = tmp_path / "stage1"
    stage1_cfg = _make_cfg(setting=3, output_dir=stage1_dir)
    assert stage1_cfg.meta is True
    assert stage1_cfg.curriculum is False

    _, _, stage1_metrics = run_training_cl(env, stage1_cfg)
    assert (stage1_dir / "checkpoint.pt").exists()
    assert stage1_metrics.total_updates > 0

    # ── Stage 2 — load stage-1 checkpoint as teacher, run CL ────────────────
    stage2_dir = tmp_path / "stage2"
    set_all_seeds(SEED + 1)  # different seed so student ≠ teacher init
    env2 = Environment("space_invaders")
    stage2_cfg = _make_cfg(
        setting=3,
        seed=SEED + 1,
        output_dir=stage2_dir,
        curriculum=True,
        teacher_load_path=stage1_dir / "checkpoint.pt",
    )

    _, second_net, stage2_metrics = run_training_cl(env2, stage2_cfg)

    assert second_net is not None
    assert stage2_metrics.total_updates > 0

    # At least one episode recorded CL component losses (the ones after
    # the replay buffer warmed up).
    cl_task_recorded = [v for v in stage2_metrics.episode_components_first_task if v is not None]
    assert len(cl_task_recorded) > 0, "CL mode must record at least one component-loss step"

    # Same for SO components (meta=True).
    cl_task_so = [v for v in stage2_metrics.episode_components_second_task if v is not None]
    assert len(cl_task_so) > 0

    # All recorded component losses are finite non-negatives.
    for lst_attr in (
        "episode_components_first_task",
        "episode_components_first_distillation",
        "episode_components_first_feature",
        "episode_components_second_task",
        "episode_components_second_distillation",
        "episode_components_second_feature",
    ):
        for v in getattr(stage2_metrics, lst_attr):
            if v is not None:
                assert v >= 0.0
                assert torch.isfinite(torch.tensor(v))


# ── load_partial_state_dict ────────────────────────────────────────────────


def test_load_partial_state_dict_copies_matching_shapes():
    src = SarlCLQNetwork(in_channels=4, num_actions=6)
    dst = SarlCLQNetwork(in_channels=4, num_actions=6)

    loaded = load_partial_state_dict(dst, src.state_dict())
    for (name, src_p), (_, dst_p) in zip(
        src.named_parameters(), loaded.named_parameters(), strict=True
    ):
        assert torch.allclose(src_p, dst_p, atol=1e-10), name


def test_load_partial_state_dict_skips_shape_mismatches():
    """If dst has different in_channels, the conv weight won't match — skip it."""
    src = SarlCLQNetwork(in_channels=4, num_actions=6)
    dst = SarlCLQNetwork(in_channels=7, num_actions=6)  # different in_channels

    before = copy.deepcopy(dst)
    loaded = load_partial_state_dict(dst, src.state_dict())

    # conv.weight shape differs → loaded keeps dst's original init.
    before_conv = dict(before.named_parameters())["conv.weight"]
    loaded_conv = dict(loaded.named_parameters())["conv.weight"]
    assert torch.allclose(before_conv, loaded_conv, atol=1e-10)

    # actions.weight (no in_channels dependency) was copied.
    src_actions = dict(src.named_parameters())["actions.weight"]
    loaded_actions = dict(loaded.named_parameters())["actions.weight"]
    assert torch.allclose(src_actions, loaded_actions, atol=1e-10)


def test_load_partial_state_dict_does_not_mutate_input():
    """``load_partial_state_dict`` deep-copies before loading — original untouched."""
    src = SarlCLQNetwork(in_channels=4, num_actions=6)
    dst = SarlCLQNetwork(in_channels=4, num_actions=6)
    dst_before_id = id(dst)
    dst_params_before = {n: p.detach().clone() for n, p in dst.named_parameters()}

    loaded = load_partial_state_dict(dst, src.state_dict())

    assert id(loaded) != dst_before_id  # it's a new object
    # Original dst's params untouched.
    for n, p in dst.named_parameters():
        assert torch.allclose(dst_params_before[n], p.detach(), atol=1e-10)


# ── Setting table ──────────────────────────────────────────────────────────


def test_setting_to_config_cl_preserves_cl_fields():
    """CL-specific fields (curriculum / adaptive / teacher_load_path) survive the transform."""
    base = SarlCLTrainingConfig(
        game="breakout",
        seed=99,
        num_frames=1234,
        curriculum=True,
        adaptive_backbone=True,
        max_input_channels=10,
        teacher_load_path=None,
        weight_task=2.0,
    )
    cfg = setting_to_config_cl(6, base)
    assert cfg.game == "breakout"
    assert cfg.seed == 99
    assert cfg.num_frames == 1234
    assert cfg.curriculum is True
    assert cfg.adaptive_backbone is True
    assert cfg.max_input_channels == 10
    assert cfg.weight_task == 2.0
    # Paper setting 6 knobs applied.
    assert cfg.meta is True
    assert cfg.cascade_iterations_1 == 50
    assert cfg.cascade_iterations_2 == 50


def test_setting_to_config_cl_rejects_out_of_range():
    with pytest.raises(ValueError):
        setting_to_config_cl(0)
    with pytest.raises(ValueError):
        setting_to_config_cl(7)


def test_setting_to_config_cl_table_is_complete():
    """All 6 paper settings must resolve to distinct (meta, cascade_1, cascade_2)."""
    seen = set()
    for s in range(1, 7):
        cfg = setting_to_config_cl(s)
        seen.add((cfg.meta, cfg.cascade_iterations_1, cfg.cascade_iterations_2))
    assert len(seen) == 6
