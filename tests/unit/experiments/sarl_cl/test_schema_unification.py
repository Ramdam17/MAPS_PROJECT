"""Sprint-08 D.19b — schema-unification tests for sarl_cl checkpoint files.

Ensures that:

1. ``_persist_checkpoint_cl`` writes a file that the teacher-loading path
   inside ``_build_networks`` can read — i.e. the SAME file works both for
   resume (full state) and as a ``--teacher-load-path`` target.
2. ``_persist_outputs`` no longer writes ``checkpoint.pt`` (that path is now
   owned exclusively by ``_persist_checkpoint_cl``). ``metrics.json`` is
   still produced.
3. Legacy checkpoints written by the pre-D.19b ``_persist_outputs`` schema
   (``policy_net_state_dict`` / ``second_net_state_dict``) still load via
   the backward-compatible fallback in ``_build_networks``.

CPU-only; no MinAtar dependency. Builds minimal networks + cfg and calls the
sarl_cl training_loop primitives directly.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from maps.experiments.sarl_cl.loss_weighting import DynamicLossWeighter
from maps.experiments.sarl_cl.training_loop import (
    CLTrainingMetrics,
    SarlCLTrainingConfig,
    SarlReplayBuffer,
    _build_networks,
    _build_optimizers,
    _persist_checkpoint_cl,
    _persist_outputs,
)


IN_CHANNELS = 4
NUM_ACTIONS = 6


def _base_cfg(tmp_path: Path, *, curriculum: bool = False, meta: bool = True) -> SarlCLTrainingConfig:
    return SarlCLTrainingConfig(
        game="space_invaders",
        seed=42,
        meta=meta,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        curriculum=curriculum,
        adaptive_backbone=False,
        max_input_channels=10,
        teacher_load_path=None,
        weight_task=1.0,
        weight_distillation=1.0,
        weight_feature=1.0,
        num_frames=1_000,
        batch_size=32,
        replay_buffer_size=100,
        replay_start_size=10,
        training_freq=1,
        target_update_freq=10,
        step_size_1=0.0003,
        step_size_2=0.0002,
        scheduler_period=1_000,  # skip D.9 step=1 warning in tests
        scheduler_gamma=0.999,
        alpha=45.0,
        gamma=0.999,
        checkpoint_every_updates=10_000,
        resume_from=None,
        device="cpu",
        output_dir=tmp_path,
    )


def test_persist_outputs_no_longer_writes_checkpoint(tmp_path: Path) -> None:
    """D.19b: `_persist_outputs` must NOT create/overwrite `checkpoint.pt`."""
    cfg = _base_cfg(tmp_path, meta=True, curriculum=False)
    policy_net, _, second_net, _, _ = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    metrics = CLTrainingMetrics()

    # Plant a pre-existing checkpoint.pt with a known sentinel payload so we
    # can detect whether _persist_outputs touches it.
    ckpt_path = tmp_path / "checkpoint.pt"
    sentinel = {"sentinel": "do_not_overwrite", "format_version": 99}
    torch.save(sentinel, ckpt_path)

    _persist_outputs(policy_net, second_net, metrics, cfg)

    # Sentinel must survive.
    reloaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert reloaded == sentinel, "_persist_outputs overwrote checkpoint.pt (D.19b regression)"

    # metrics.json should still be produced.
    assert (tmp_path / "metrics.json").is_file()
    meta_json = json.loads((tmp_path / "metrics.json").read_text())
    assert meta_json["game"] == cfg.game


def test_teacher_loading_from_d13_checkpoint(tmp_path: Path) -> None:
    """D.19b: a D.13 checkpoint (canonical keys) serves as a teacher source."""
    # Task-1 run (no teacher) → produce a D.13 checkpoint.
    cfg_task1 = _base_cfg(tmp_path, curriculum=False, meta=True)
    policy_net, target_net, second_net, _, _ = _build_networks(
        IN_CHANNELS, NUM_ACTIONS, cfg_task1
    )
    opt1, opt2, sch1, sch2 = _build_optimizers(policy_net, second_net, cfg_task1)

    ckpt = tmp_path / "task1_checkpoint.pt"
    _persist_checkpoint_cl(
        ckpt,
        t=0, episode_idx=0, policy_update_counter=0,
        policy_net=policy_net, target_net=target_net, second_order_net=second_net,
        teacher_first_net=None, teacher_second_net=None,
        optimizer=opt1, optimizer2=opt2, scheduler1=sch1, scheduler2=sch2,
        loss_weighter=None, loss_weighter_second=None,
        buffer=SarlReplayBuffer(cfg_task1.replay_buffer_size),
        metrics=CLTrainingMetrics(),
        cfg=cfg_task1,
    )

    # Task-2 run: point teacher at that file, same cfg except curriculum=True.
    cfg_task2 = replace(cfg_task1, curriculum=True, teacher_load_path=ckpt)
    (_, _, _, teacher_first, teacher_second) = _build_networks(
        IN_CHANNELS, NUM_ACTIONS, cfg_task2
    )

    assert teacher_first is not None, "FO teacher failed to load from D.13 checkpoint"
    assert teacher_second is not None, "SO teacher failed to load from D.13 checkpoint"

    # Teachers must be frozen.
    assert all(not p.requires_grad for p in teacher_first.parameters())
    assert all(not p.requires_grad for p in teacher_second.parameters())

    # Teacher FO weights must match the original policy_net (conv layer check).
    for k, v in policy_net.state_dict().items():
        if k in teacher_first.state_dict():
            assert torch.equal(v, teacher_first.state_dict()[k]), (
                f"teacher FO param {k} drifted from policy source"
            )


def test_teacher_loading_from_legacy_checkpoint(tmp_path: Path) -> None:
    """D.19b backward-compat: legacy {policy_net_state_dict, second_net_state_dict}
    schema (pre-D.19b `_persist_outputs`) must still be loadable via fallback."""
    # Build a reference policy + second to populate the legacy-schema ckpt.
    cfg_ref = _base_cfg(tmp_path, curriculum=False, meta=True)
    ref_policy, _, ref_second, _, _ = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg_ref)

    legacy_ckpt = tmp_path / "legacy_checkpoint.pt"
    torch.save(
        {
            "policy_net_state_dict": ref_policy.state_dict(),
            "second_net_state_dict": ref_second.state_dict(),
            "game": "space_invaders",
            "seed": 42,
        },
        legacy_ckpt,
    )

    # Task-2 cfg pointing at the legacy file.
    cfg_task2 = replace(cfg_ref, curriculum=True, teacher_load_path=legacy_ckpt)
    (_, _, _, teacher_first, teacher_second) = _build_networks(
        IN_CHANNELS, NUM_ACTIONS, cfg_task2
    )

    assert teacher_first is not None
    assert teacher_second is not None
    # Weights must match the legacy reference.
    for k, v in ref_policy.state_dict().items():
        if k in teacher_first.state_dict():
            assert torch.equal(v, teacher_first.state_dict()[k])


def test_teacher_loading_rejects_unknown_schema(tmp_path: Path) -> None:
    """A checkpoint with neither canonical nor legacy keys should raise clearly."""
    junk_ckpt = tmp_path / "junk.pt"
    torch.save({"unrelated_key": torch.zeros(1)}, junk_ckpt)

    cfg = _base_cfg(tmp_path, curriculum=True, meta=False)
    cfg = replace(cfg, teacher_load_path=junk_ckpt)
    with pytest.raises(ValueError, match="policy_state_dict"):
        _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
