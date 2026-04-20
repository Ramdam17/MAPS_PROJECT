"""Unit tests for the Sprint-08 D.13 SARL checkpoint/resume primitives.

Covers:
* Round-trip save → load preserves all model / optimizer / scheduler weights.
* Round-trip preserves the three RNG streams (torch, Python random, numpy
  legacy) so the next draw after resume matches a run that never paused.
* Atomic write: an interrupted ``torch.save`` does not leave a half-written
  checkpoint at the target path.
* Cfg guardrail: resuming with a mismatched guarded field raises ValueError
  rather than silently mixing protocols.

Tests run CPU-only and avoid the MinAtar dependency (no env interaction).
The checkpoint functions are exercised directly against small tensor state.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from maps.experiments.sarl.model import SarlQNetwork, SarlSecondOrderNetwork
from maps.experiments.sarl.training_loop import (
    SarlReplayBuffer,
    SarlTrainingConfig,
    Transition,
    TrainingMetrics,
    _build_networks,
    _build_optimizers,
    _persist_checkpoint,
    _restore_from_checkpoint,
)


IN_CHANNELS = 4
NUM_ACTIONS = 6


def _base_cfg(tmp_path: Path, *, meta: bool = True) -> SarlTrainingConfig:
    """Minimal-but-realistic cfg for checkpoint tests."""
    return SarlTrainingConfig(
        game="space_invaders",
        seed=42,
        meta=meta,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        num_frames=1_000,
        batch_size=32,
        replay_buffer_size=100,
        replay_start_size=10,
        training_freq=1,
        target_update_freq=10,
        step_size_1=0.0003,
        step_size_2=0.0002,
        scheduler_period=1_000,  # avoid the step=1 log.warning noise in tests
        scheduler_gamma=0.999,
        adam_betas=(0.95, 0.95),
        gamma=0.999,
        alpha=45.0,
        checkpoint_every_updates=10_000,
        resume_from=None,
        device="cpu",
        output_dir=tmp_path,
    )


def _fake_transition() -> Transition:
    """Small Transition stand-in for buffer serialisation tests."""
    return Transition(
        state=torch.zeros(1, IN_CHANNELS, 10, 10),
        next_state=torch.zeros(1, IN_CHANNELS, 10, 10),
        action=torch.tensor([[0]], dtype=torch.int64),
        reward=torch.tensor([[0.0]]),
        is_terminal=torch.tensor([[0]], dtype=torch.int64),
    )


def _seed_training_rngs(seed: int) -> None:
    """Seed torch / Python random / numpy legacy (same as set_all_seeds)."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def test_checkpoint_roundtrip_preserves_weights(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path, meta=True)
    policy_net, target_net, second_net = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    opt1, opt2, sch1, sch2 = _build_optimizers(policy_net, second_net, cfg)

    # Mutate weights so the "default init" doesn't accidentally mask a bug.
    with torch.no_grad():
        for p in policy_net.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    buffer = SarlReplayBuffer(cfg.replay_buffer_size)
    buffer.add(*_fake_transition())
    metrics = TrainingMetrics()
    metrics.total_updates = 5
    metrics.total_frames = 42

    ckpt = tmp_path / "checkpoint.pt"
    _persist_checkpoint(
        ckpt,
        t=42,
        episode_idx=3,
        policy_update_counter=5,
        policy_net=policy_net,
        target_net=target_net,
        second_order_net=second_net,
        optimizer=opt1,
        optimizer2=opt2,
        scheduler1=sch1,
        scheduler2=sch2,
        buffer=buffer,
        metrics=metrics,
        cfg=cfg,
    )

    # Fresh networks (default-init, guaranteed different weights).
    policy2, target2, second2 = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    opt1_b, opt2_b, sch1_b, sch2_b = _build_optimizers(policy2, second2, cfg)

    t, ep, upd, buf, mets = _restore_from_checkpoint(
        ckpt, cfg, policy2, target2, second2, opt1_b, opt2_b, sch1_b, sch2_b
    )

    assert t == 42
    assert ep == 3
    assert upd == 5
    assert mets.total_frames == 42
    assert len(buf) == 1

    # Every policy weight must match post-restore.
    for k, v in policy_net.state_dict().items():
        assert torch.equal(v, policy2.state_dict()[k]), f"policy weight mismatch on {k}"
    for k, v in target_net.state_dict().items():
        assert torch.equal(v, target2.state_dict()[k]), f"target weight mismatch on {k}"
    for k, v in second_net.state_dict().items():
        assert torch.equal(v, second2.state_dict()[k]), f"second weight mismatch on {k}"


def test_checkpoint_roundtrip_preserves_rng_state(tmp_path: Path) -> None:
    """After restore, the next random draws match the un-interrupted continuation."""
    cfg = _base_cfg(tmp_path, meta=False)

    # Reference: seed, consume a few draws, snapshot the next expected values
    # from *that same RNG stream* — these are what the restored checkpoint
    # must reproduce.
    _seed_training_rngs(42)
    _ = torch.randn(10)  # some pre-checkpoint consumption
    _ = random.random()
    _ = np.random.rand(5)

    policy_net, target_net, _ = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    opt1, _, sch1, _ = _build_optimizers(policy_net, None, cfg)
    buffer = SarlReplayBuffer(cfg.replay_buffer_size)
    metrics = TrainingMetrics()

    ckpt = tmp_path / "checkpoint.pt"
    _persist_checkpoint(
        ckpt,
        t=0, episode_idx=0, policy_update_counter=0,
        policy_net=policy_net, target_net=target_net, second_order_net=None,
        optimizer=opt1, optimizer2=None, scheduler1=sch1, scheduler2=None,
        buffer=buffer, metrics=metrics, cfg=cfg,
    )

    # Reference continuation (no pause).
    ref_torch = torch.randn(3)
    ref_py = random.random()
    ref_np = np.random.rand(2)

    # Now simulate a "restart": reseed different, then restore.
    _seed_training_rngs(999)
    policy2, target2, _ = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    opt1_b, _, sch1_b, _ = _build_optimizers(policy2, None, cfg)
    _restore_from_checkpoint(
        ckpt, cfg, policy2, target2, None, opt1_b, None, sch1_b, None
    )

    # These draws should match the reference continuation bit-for-bit.
    assert torch.equal(torch.randn(3), ref_torch)
    assert random.random() == ref_py
    assert np.array_equal(np.random.rand(2), ref_np)


def test_checkpoint_atomic_on_interrupted_write(tmp_path: Path) -> None:
    """If torch.save fails mid-write, the target path keeps its prior content."""
    cfg = _base_cfg(tmp_path, meta=False)
    policy_net, target_net, _ = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    opt1, _, sch1, _ = _build_optimizers(policy_net, None, cfg)
    buffer = SarlReplayBuffer(cfg.replay_buffer_size)
    metrics = TrainingMetrics()
    ckpt = tmp_path / "checkpoint.pt"

    # 1st successful write so we have a non-empty target.
    _persist_checkpoint(
        ckpt,
        t=1, episode_idx=0, policy_update_counter=0,
        policy_net=policy_net, target_net=target_net, second_order_net=None,
        optimizer=opt1, optimizer2=None, scheduler1=sch1, scheduler2=None,
        buffer=buffer, metrics=metrics, cfg=cfg,
    )
    original_size = ckpt.stat().st_size
    original_payload = torch.load(ckpt, map_location="cpu", weights_only=False)

    # 2nd write with a monkey-patched torch.save that explodes mid-serialisation.
    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated disk error mid-write")

    real_save = torch.save
    torch.save = _boom  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="simulated disk error"):
            _persist_checkpoint(
                ckpt,
                t=999, episode_idx=99, policy_update_counter=99,
                policy_net=policy_net, target_net=target_net, second_order_net=None,
                optimizer=opt1, optimizer2=None, scheduler1=sch1, scheduler2=None,
                buffer=buffer, metrics=metrics, cfg=cfg,
            )
    finally:
        torch.save = real_save  # type: ignore[assignment]

    # Target file must still hold the original payload — atomic replace never fired.
    assert ckpt.is_file()
    assert ckpt.stat().st_size == original_size
    restored = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert restored["t"] == original_payload["t"] == 1

    # The .tmp file may or may not linger (depends on which stage of save failed);
    # cleanliness is nice-to-have, not a correctness requirement here.


def test_checkpoint_cfg_guardrail_rejects_mismatched_seed(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path, meta=False)
    policy_net, target_net, _ = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg)
    opt1, _, sch1, _ = _build_optimizers(policy_net, None, cfg)
    buffer = SarlReplayBuffer(cfg.replay_buffer_size)
    metrics = TrainingMetrics()
    ckpt = tmp_path / "checkpoint.pt"

    _persist_checkpoint(
        ckpt,
        t=0, episode_idx=0, policy_update_counter=0,
        policy_net=policy_net, target_net=target_net, second_order_net=None,
        optimizer=opt1, optimizer2=None, scheduler1=sch1, scheduler2=None,
        buffer=buffer, metrics=metrics, cfg=cfg,
    )

    # Same cfg except seed — that's a guarded field, so resume MUST reject.
    bad_cfg = replace(cfg, seed=1337)
    policy2, target2, _ = _build_networks(IN_CHANNELS, NUM_ACTIONS, bad_cfg)
    opt1_b, _, sch1_b, _ = _build_optimizers(policy2, None, bad_cfg)
    with pytest.raises(ValueError, match="seed"):
        _restore_from_checkpoint(
            ckpt, bad_cfg, policy2, target2, None, opt1_b, None, sch1_b, None
        )


def test_checkpoint_meta_mismatch_raises(tmp_path: Path) -> None:
    """Checkpoint saved with meta=True cannot be loaded by a meta=False caller (guarded)."""
    cfg_meta = _base_cfg(tmp_path, meta=True)
    policy_net, target_net, second_net = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg_meta)
    opt1, opt2, sch1, sch2 = _build_optimizers(policy_net, second_net, cfg_meta)
    buffer = SarlReplayBuffer(cfg_meta.replay_buffer_size)
    metrics = TrainingMetrics()
    ckpt = tmp_path / "checkpoint.pt"

    _persist_checkpoint(
        ckpt,
        t=0, episode_idx=0, policy_update_counter=0,
        policy_net=policy_net, target_net=target_net, second_order_net=second_net,
        optimizer=opt1, optimizer2=opt2, scheduler1=sch1, scheduler2=sch2,
        buffer=buffer, metrics=metrics, cfg=cfg_meta,
    )

    cfg_nometa = replace(cfg_meta, meta=False)
    policy2, target2, _ = _build_networks(IN_CHANNELS, NUM_ACTIONS, cfg_nometa)
    opt1_b, _, sch1_b, _ = _build_optimizers(policy2, None, cfg_nometa)
    # meta is a guarded field → caller sees a cfg mismatch error first, not an
    # optimizer2/scheduler2 None-check.
    with pytest.raises(ValueError, match="meta"):
        _restore_from_checkpoint(
            ckpt, cfg_nometa, policy2, target2, None, opt1_b, None, sch1_b, None
        )
