"""Sprint-08 D.21 — SARL+CL checkpoint/resume regression tests.

Parallel coverage to ``tests/unit/experiments/sarl/test_checkpoint.py`` (D.13)
but exercising the CL variant: 5 networks (policy, target, second,
teacher_first, teacher_second) + 2 DynamicLossWeighter instances that MUST
round-trip so the running-max anchor for the distillation term survives a
paused run.

CPU-only; no MinAtar env dependency. Builds minimal nets + cfg and calls the
sarl_cl training_loop checkpoint primitives directly.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from maps.experiments.sarl_cl.loss_weighting import DynamicLossWeighter
from maps.experiments.sarl_cl.training_loop import (
    CLTrainingMetrics,
    SarlCLTrainingConfig,
    SarlReplayBuffer,
    Transition,
    _build_networks,
    _build_optimizers,
    _persist_checkpoint_cl,
    _restore_from_checkpoint_cl,
)


IN_CHANNELS = 4
NUM_ACTIONS = 6


def _base_cfg(
    tmp_path: Path, *, meta: bool = True, curriculum: bool = False
) -> SarlCLTrainingConfig:
    """Realistic-but-minimal CL cfg for checkpoint tests."""
    return SarlCLTrainingConfig(
        game="space_invaders",
        seed=42,
        meta=meta,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        curriculum=curriculum,
        adaptive_backbone=False,
        max_input_channels=10,  # post-D.20 paper value
        teacher_load_path=None,
        weight_task=0.3,  # post-D.20 paper Table 11
        weight_distillation=0.6,
        weight_feature=0.1,
        num_frames=1_000,
        batch_size=32,
        replay_buffer_size=100,
        replay_start_size=10,
        training_freq=1,
        target_update_freq=10,
        step_size_1=0.0003,
        step_size_2=0.0002,
        scheduler_period=1_000,  # avoid the D.9 step=1 log.warning noise
        scheduler_gamma=0.999,
        adam_betas=(0.95, 0.95),
        alpha=45.0,
        gamma=0.999,
        checkpoint_every_updates=10_000,
        resume_from=None,
        device="cpu",
        output_dir=tmp_path,
    )


def _fake_transition() -> Transition:
    """Tiny Transition stand-in for buffer serialisation tests."""
    return Transition(
        state=torch.zeros(1, IN_CHANNELS, 10, 10),
        next_state=torch.zeros(1, IN_CHANNELS, 10, 10),
        action=torch.tensor([[0]], dtype=torch.int64),
        reward=torch.tensor([[0.0]]),
        is_terminal=torch.tensor([[0]], dtype=torch.int64),
    )


def _make_teacher_checkpoint(tmp_path: Path, cfg: SarlCLTrainingConfig) -> Path:
    """Write a throwaway D.13 checkpoint that a follow-up run can load as teacher."""
    policy, target, second, _, _ = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    opt1, opt2, sch1, sch2 = _build_optimizers(policy, second, cfg)

    ckpt = tmp_path / "teacher.pt"
    _persist_checkpoint_cl(
        ckpt,
        t=0, episode_idx=0, policy_update_counter=0,
        policy_net=policy, target_net=target, second_order_net=second,
        teacher_first_net=None, teacher_second_net=None,
        optimizer=opt1, optimizer2=opt2, scheduler1=sch1, scheduler2=sch2,
        loss_weighter=None, loss_weighter_second=None,
        buffer=SarlReplayBuffer(cfg.replay_buffer_size),
        metrics=CLTrainingMetrics(),
        cfg=cfg,
    )
    return ckpt


def _seed_training_rngs(seed: int) -> None:
    """Seed the same 3 streams as maps.utils.seeding.set_all_seeds."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


# ─── Tests ─────────────────────────────────────────────────────────────────


def test_cl_checkpoint_roundtrip_preserves_weights(tmp_path: Path) -> None:
    """All 5 nets (policy, target, second, teacher_first, teacher_second)
    must match bit-for-bit after a save → restore cycle."""
    # Stage 1: build a teacher-source checkpoint so curriculum=True wires up
    # teacher nets in the later run.
    teacher_src_cfg = _base_cfg(tmp_path, meta=True, curriculum=False)
    teacher_src_ckpt = _make_teacher_checkpoint(tmp_path, teacher_src_cfg)

    cfg = _base_cfg(tmp_path, meta=True, curriculum=True)
    cfg = replace(cfg, teacher_load_path=teacher_src_ckpt)

    (
        policy_net, target_net, second_net, teacher_first, teacher_second
    ) = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    opt1, opt2, sch1, sch2 = _build_optimizers(policy_net, second_net, cfg)

    # Mutate policy weights so defaults can't accidentally match post-restore.
    with torch.no_grad():
        for p in policy_net.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    buffer = SarlReplayBuffer(cfg.replay_buffer_size)
    buffer.add(*_fake_transition())
    metrics = CLTrainingMetrics()
    metrics.total_updates = 7
    metrics.total_frames = 84

    loss_weighter = DynamicLossWeighter()
    loss_weighter_second = DynamicLossWeighter()

    ckpt = tmp_path / "cl_checkpoint.pt"
    _persist_checkpoint_cl(
        ckpt,
        t=84, episode_idx=3, policy_update_counter=7,
        policy_net=policy_net, target_net=target_net, second_order_net=second_net,
        teacher_first_net=teacher_first, teacher_second_net=teacher_second,
        optimizer=opt1, optimizer2=opt2, scheduler1=sch1, scheduler2=sch2,
        loss_weighter=loss_weighter, loss_weighter_second=loss_weighter_second,
        buffer=buffer, metrics=metrics, cfg=cfg,
    )

    # Fresh nets with DIFFERENT default inits.
    (
        policy2, target2, second2, teacher_first2, teacher_second2
    ) = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    opt1_b, opt2_b, sch1_b, sch2_b = _build_optimizers(policy2, second2, cfg)

    t, ep, upd, buf, mets, w1, w2 = _restore_from_checkpoint_cl(
        ckpt, cfg,
        policy2, target2, second2, teacher_first2, teacher_second2,
        opt1_b, opt2_b, sch1_b, sch2_b,
    )

    assert t == 84
    assert ep == 3
    assert upd == 7
    assert mets.total_frames == 84
    assert len(buf) == 1

    # All 5 nets must round-trip.
    for src, dst, name in [
        (policy_net, policy2, "policy"),
        (target_net, target2, "target"),
        (second_net, second2, "second"),
        (teacher_first, teacher_first2, "teacher_first"),
        (teacher_second, teacher_second2, "teacher_second"),
    ]:
        assert src is not None and dst is not None
        for k, v in src.state_dict().items():
            assert torch.equal(v, dst.state_dict()[k]), (
                f"weight mismatch on {name}.{k}"
            )


def test_cl_checkpoint_rng_preserves_determinism(tmp_path: Path) -> None:
    """After restore the next random draws on all 3 streams match the
    uninterrupted continuation."""
    cfg = _base_cfg(tmp_path, meta=False, curriculum=False)

    _seed_training_rngs(42)
    _ = torch.randn(10)
    _ = random.random()
    _ = np.random.rand(5)

    policy_net, target_net, _, _, _ = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    opt1, _, sch1, _ = _build_optimizers(policy_net, None, cfg)

    ckpt = tmp_path / "rng_ckpt.pt"
    _persist_checkpoint_cl(
        ckpt,
        t=0, episode_idx=0, policy_update_counter=0,
        policy_net=policy_net, target_net=target_net, second_order_net=None,
        teacher_first_net=None, teacher_second_net=None,
        optimizer=opt1, optimizer2=None, scheduler1=sch1, scheduler2=None,
        loss_weighter=None, loss_weighter_second=None,
        buffer=SarlReplayBuffer(cfg.replay_buffer_size),
        metrics=CLTrainingMetrics(),
        cfg=cfg,
    )

    # Reference continuation without a pause.
    ref_torch = torch.randn(3)
    ref_py = random.random()
    ref_np = np.random.rand(2)

    # Reset to a different seed, then restore: draws must match the reference.
    _seed_training_rngs(999)
    policy2, target2, _, _, _ = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    opt1_b, _, sch1_b, _ = _build_optimizers(policy2, None, cfg)
    _restore_from_checkpoint_cl(
        ckpt, cfg, policy2, target2, None, None, None, opt1_b, None, sch1_b, None
    )
    assert torch.equal(torch.randn(3), ref_torch)
    assert random.random() == ref_py
    assert np.array_equal(np.random.rand(2), ref_np)


def test_cl_checkpoint_loss_weighter_state_roundtrip(tmp_path: Path) -> None:
    """DynamicLossWeighter running-max MUST survive the pause — otherwise the
    distillation anchor silently resets and training diverges on resume."""
    teacher_src_cfg = _base_cfg(tmp_path, meta=True, curriculum=False)
    teacher_src_ckpt = _make_teacher_checkpoint(tmp_path, teacher_src_cfg)

    cfg = _base_cfg(tmp_path, meta=True, curriculum=True)
    cfg = replace(cfg, teacher_load_path=teacher_src_ckpt)

    policy_net, target_net, second_net, teacher_first, teacher_second = _build_networks(
        IN_CHANNELS, NUM_ACTIONS, cfg
    )
    opt1, opt2, sch1, sch2 = _build_optimizers(policy_net, second_net, cfg)

    # Feed the weighters some non-trivial state so defaults wouldn't accidentally match.
    weighter_first = DynamicLossWeighter()
    weighter_second = DynamicLossWeighter()
    for step, loss_tuple in enumerate([(2.5, 7.1, 0.9), (1.8, 10.2, 1.3), (3.0, 9.7, 0.5)]):
        weighter_first.update(
            {"task": loss_tuple[0], "distillation": loss_tuple[1], "feature": loss_tuple[2]}
        )
        weighter_second.update(
            {"task": loss_tuple[0] * 0.5, "distillation": loss_tuple[1] * 0.5, "feature": loss_tuple[2] * 0.5}
        )

    # Snapshot pre-persist for later comparison.
    pre_first_max = dict(weighter_first.historical_max)
    pre_second_max = dict(weighter_second.historical_max)
    pre_first_steps = weighter_first.steps

    ckpt = tmp_path / "weighter_ckpt.pt"
    _persist_checkpoint_cl(
        ckpt,
        t=0, episode_idx=0, policy_update_counter=0,
        policy_net=policy_net, target_net=target_net, second_order_net=second_net,
        teacher_first_net=teacher_first, teacher_second_net=teacher_second,
        optimizer=opt1, optimizer2=opt2, scheduler1=sch1, scheduler2=sch2,
        loss_weighter=weighter_first, loss_weighter_second=weighter_second,
        buffer=SarlReplayBuffer(cfg.replay_buffer_size),
        metrics=CLTrainingMetrics(),
        cfg=cfg,
    )

    # Rebuild fresh and restore.
    (
        policy2, target2, second2, teacher_first2, teacher_second2
    ) = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    opt1_b, opt2_b, sch1_b, sch2_b = _build_optimizers(policy2, second2, cfg)
    _, _, _, _, _, restored_w1, restored_w2 = _restore_from_checkpoint_cl(
        ckpt, cfg,
        policy2, target2, second2, teacher_first2, teacher_second2,
        opt1_b, opt2_b, sch1_b, sch2_b,
    )

    assert restored_w1 is not None and restored_w2 is not None
    # Running-max must be exactly preserved.
    assert restored_w1.historical_max == pre_first_max
    assert restored_w2.historical_max == pre_second_max
    # Step counter too — otherwise the mid-interval snapshot logic desyncs.
    assert restored_w1.steps == pre_first_steps


def test_cl_checkpoint_cfg_guardrail_rejects_curriculum_mismatch(tmp_path: Path) -> None:
    """Resuming with curriculum=False on a curriculum=True checkpoint should
    raise — silently downgrading would discard the teacher anchor."""
    # Need a real teacher source to set curriculum=True.
    teacher_src_cfg = _base_cfg(tmp_path, meta=True, curriculum=False)
    teacher_src_ckpt = _make_teacher_checkpoint(tmp_path, teacher_src_cfg)

    cfg_curr = _base_cfg(tmp_path, meta=True, curriculum=True)
    cfg_curr = replace(cfg_curr, teacher_load_path=teacher_src_ckpt)

    policy, target, second, tf, ts = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg_curr)
    opt1, opt2, sch1, sch2 = _build_optimizers(policy, second, cfg_curr)

    ckpt = tmp_path / "ckpt.pt"
    _persist_checkpoint_cl(
        ckpt,
        t=0, episode_idx=0, policy_update_counter=0,
        policy_net=policy, target_net=target, second_order_net=second,
        teacher_first_net=tf, teacher_second_net=ts,
        optimizer=opt1, optimizer2=opt2, scheduler1=sch1, scheduler2=sch2,
        loss_weighter=DynamicLossWeighter(), loss_weighter_second=DynamicLossWeighter(),
        buffer=SarlReplayBuffer(cfg_curr.replay_buffer_size),
        metrics=CLTrainingMetrics(),
        cfg=cfg_curr,
    )

    cfg_no_curr = replace(cfg_curr, curriculum=False, teacher_load_path=None)
    p2, t2, s2, _, _ = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg_no_curr)
    opt1b, opt2b, sch1b, sch2b = _build_optimizers(p2, s2, cfg_no_curr)
    with pytest.raises(ValueError, match="curriculum"):
        _restore_from_checkpoint_cl(
            ckpt, cfg_no_curr,
            p2, t2, s2, None, None, opt1b, opt2b, sch1b, sch2b,
        )


def test_cl_checkpoint_cfg_guardrail_rejects_max_channels_mismatch(tmp_path: Path) -> None:
    """max_input_channels is a guarded CL field — resume with different value
    should raise."""
    cfg = _base_cfg(tmp_path, meta=False, curriculum=False)
    p, t, _, _, _ = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    opt1, _, sch1, _ = _build_optimizers(p, None, cfg)

    ckpt = tmp_path / "ckpt.pt"
    _persist_checkpoint_cl(
        ckpt,
        t=0, episode_idx=0, policy_update_counter=0,
        policy_net=p, target_net=t, second_order_net=None,
        teacher_first_net=None, teacher_second_net=None,
        optimizer=opt1, optimizer2=None, scheduler1=sch1, scheduler2=None,
        loss_weighter=None, loss_weighter_second=None,
        buffer=SarlReplayBuffer(cfg.replay_buffer_size),
        metrics=CLTrainingMetrics(),
        cfg=cfg,
    )

    cfg_different = replace(cfg, max_input_channels=7)
    p2, t2, _, _, _ = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg_different)
    opt1b, _, sch1b, _ = _build_optimizers(p2, None, cfg_different)
    with pytest.raises(ValueError, match="max_input_channels"):
        _restore_from_checkpoint_cl(
            ckpt, cfg_different,
            p2, t2, None, None, None, opt1b, None, sch1b, None,
        )


def test_cl_checkpoint_atomic_on_interrupted_write(tmp_path: Path) -> None:
    """Interrupted torch.save mid-write must leave the target path unchanged."""
    cfg = _base_cfg(tmp_path, meta=False, curriculum=False)
    p, t, _, _, _ = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    opt1, _, sch1, _ = _build_optimizers(p, None, cfg)
    buffer = SarlReplayBuffer(cfg.replay_buffer_size)
    metrics = CLTrainingMetrics()
    ckpt = tmp_path / "ckpt.pt"

    # First write succeeds.
    _persist_checkpoint_cl(
        ckpt,
        t=1, episode_idx=0, policy_update_counter=0,
        policy_net=p, target_net=t, second_order_net=None,
        teacher_first_net=None, teacher_second_net=None,
        optimizer=opt1, optimizer2=None, scheduler1=sch1, scheduler2=None,
        loss_weighter=None, loss_weighter_second=None,
        buffer=buffer, metrics=metrics, cfg=cfg,
    )
    original_t = torch.load(ckpt, map_location="cpu", weights_only=False)["t"]

    # Second write explodes mid-save — the os.replace never fires.
    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated disk-full mid-write")

    real_save = torch.save
    torch.save = _boom  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="simulated disk-full"):
            _persist_checkpoint_cl(
                ckpt,
                t=999, episode_idx=99, policy_update_counter=99,
                policy_net=p, target_net=t, second_order_net=None,
                teacher_first_net=None, teacher_second_net=None,
                optimizer=opt1, optimizer2=None, scheduler1=sch1, scheduler2=None,
                loss_weighter=None, loss_weighter_second=None,
                buffer=buffer, metrics=metrics, cfg=cfg,
            )
    finally:
        torch.save = real_save  # type: ignore[assignment]

    # Target file holds the original payload.
    assert ckpt.is_file()
    restored = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert restored["t"] == original_t == 1


def test_cl_checkpoint_without_teacher_degenerate_path(tmp_path: Path) -> None:
    """curriculum=False + no teacher: ensure the Null-branch survives a
    full save → restore round-trip (teachers & weighters stay None)."""
    cfg = _base_cfg(tmp_path, meta=True, curriculum=False)
    policy, target, second, tf, ts = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    opt1, opt2, sch1, sch2 = _build_optimizers(policy, second, cfg)

    assert tf is None and ts is None, "degenerate path should not build teachers"

    ckpt = tmp_path / "ckpt.pt"
    _persist_checkpoint_cl(
        ckpt,
        t=5, episode_idx=0, policy_update_counter=3,
        policy_net=policy, target_net=target, second_order_net=second,
        teacher_first_net=None, teacher_second_net=None,
        optimizer=opt1, optimizer2=opt2, scheduler1=sch1, scheduler2=sch2,
        loss_weighter=None, loss_weighter_second=None,
        buffer=SarlReplayBuffer(cfg.replay_buffer_size),
        metrics=CLTrainingMetrics(),
        cfg=cfg,
    )

    p2, t2, s2, _, _ = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    opt1b, opt2b, sch1b, sch2b = _build_optimizers(p2, s2, cfg)
    t, ep, upd, buf, mets, w1, w2 = _restore_from_checkpoint_cl(
        ckpt, cfg, p2, t2, s2, None, None, opt1b, opt2b, sch1b, sch2b,
    )

    assert t == 5
    assert upd == 3
    assert w1 is None and w2 is None, "weighters must stay None on degenerate resume"
