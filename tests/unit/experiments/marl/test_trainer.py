"""Unit tests for :class:`MAPPOTrainer` (E.9a).

Uses a minimal fake rollout buffer to exercise ``ppo_update`` and ``train``
without needing the full E.9b runner/buffer stack.

Scope (E.9a) :
- cal_value_loss with/without ValueNorm + Huber + clip
- prep_training/prep_rollout toggle modes
- ppo_update with meta=False (baseline) : returns valid floats
- ppo_update with meta=True : includes wager loss, requires wager_objective
- train() outer loop averages across mini-batches
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from gymnasium import spaces

from maps.experiments.marl import MAPPOPolicy, MAPPOTrainer
from maps.utils import load_config


HIDDEN = 32
OBS_SHAPE = (11, 11, 3)
BATCH = 8
N_AGENTS = 4
EPISODE_LEN = 4


@pytest.fixture
def cfg():
    """Smaller config for fast tests."""
    return load_config(
        "training/marl",
        overrides=[
            f"model.hidden_size={HIDDEN}",
            "model.second_order_dropout=0.0",
            "ppo.ppo_epoch=2",
            "ppo.num_mini_batch=1",
            "ppo.data_chunk_length=2",
        ],
    )


@pytest.fixture
def policy(cfg):
    action_space = spaces.Discrete(8)
    return MAPPOPolicy(
        cfg,
        obs_shape=OBS_SHAPE,
        cent_obs_shape=OBS_SHAPE,
        action_space=action_space,
        meta=True,
        cascade_iterations1=1,
        cascade_iterations2=1,
    )


@dataclass
class _FakeBuffer:
    """Minimal buffer interface used by MAPPOTrainer.train.

    Provides :
    - ``returns`` : (T+1, N, 1) value-target array
    - ``value_preds`` : (T+1, N, 1) predicted-value array
    - ``active_masks`` : (T+1, N, 1) active-agent mask
    - ``recurrent_generator(advantages, num_mini_batch, chunk_length)`` :
      yields 12-tuples of sample tensors matching ppo_update's expected shape.
    """

    returns: np.ndarray
    value_preds: np.ndarray
    active_masks: np.ndarray
    obs: np.ndarray
    actions: np.ndarray
    rnn_states: np.ndarray
    rnn_states_critic: np.ndarray
    masks: np.ndarray
    old_action_log_probs: np.ndarray

    def recurrent_generator(self, advantages, num_mini_batch, chunk_length):
        T, N = advantages.shape[:2]
        # Flatten (T, N, ...) → ((T*N), ...). All input arrays are (T+1, N, ...)
        # so slice [:T] first (drops the bootstrap last step).
        share_obs = self.obs[:T].reshape((T * N,) + self.obs.shape[2:])
        obs = self.obs[:T].reshape((T * N,) + self.obs.shape[2:])
        actions = self.actions.reshape((T * N,) + self.actions.shape[2:])  # actions shape is (T, N, 1)
        value_preds = self.value_preds[:-1].reshape((T * N, -1))
        returns = self.returns[:-1].reshape((T * N, -1))
        masks = self.masks[:-1].reshape((T * N, -1))
        active_masks = self.active_masks[:-1].reshape((T * N, -1))
        old_logprobs = self.old_action_log_probs.reshape((T * N, -1))
        adv = advantages.reshape((T * N, -1))

        rnn_states = self.rnn_states[:-1].reshape((T * N,) + self.rnn_states.shape[2:])
        rnn_states_c = self.rnn_states_critic[:-1].reshape((T * N,) + self.rnn_states_critic.shape[2:])

        yield (
            torch.from_numpy(share_obs).float(),
            torch.from_numpy(obs).float(),
            torch.from_numpy(rnn_states).float(),
            torch.from_numpy(rnn_states_c).float(),
            torch.from_numpy(actions).long(),
            value_preds,
            returns,
            torch.from_numpy(masks).float(),
            active_masks,
            old_logprobs,
            adv,
            None,  # available_actions
        )


def _make_fake_buffer(n_agents: int = N_AGENTS, episode_len: int = EPISODE_LEN):
    T, N = episode_len, n_agents
    obs_shape = OBS_SHAPE
    np.random.seed(42)
    returns = np.random.randn(T + 1, N, 1).astype(np.float32)
    value_preds = np.random.randn(T + 1, N, 1).astype(np.float32)
    active_masks = np.ones((T + 1, N, 1), dtype=np.float32)
    obs = np.random.randint(0, 256, (T + 1, N, *obs_shape), dtype=np.uint8).astype(np.float32)
    actions = np.random.randint(0, 8, (T, N, 1), dtype=np.int64)
    rnn_states = np.zeros((T + 1, N, 1, HIDDEN), dtype=np.float32)
    rnn_states_critic = np.zeros((T + 1, N, 1, HIDDEN), dtype=np.float32)
    masks = np.ones((T + 1, N, 1), dtype=np.float32)
    old_logprobs = np.random.randn(T, N, 1).astype(np.float32) - 2.0

    return _FakeBuffer(
        returns=returns,
        value_preds=value_preds,
        active_masks=active_masks,
        obs=obs,
        actions=actions,
        rnn_states=rnn_states,
        rnn_states_critic=rnn_states_critic,
        masks=masks,
        old_action_log_probs=old_logprobs,
    )


def test_trainer_builds_with_valuenorm(cfg, policy):
    trainer = MAPPOTrainer(cfg, policy, device="cpu")
    assert trainer.value_normalizer is not None


def test_trainer_rejects_popart(cfg, policy):
    cfg.ppo.use_popart = True
    with pytest.raises(NotImplementedError, match="use_popart"):
        MAPPOTrainer(cfg, policy, device="cpu")


def test_trainer_prep_toggle(cfg, policy):
    trainer = MAPPOTrainer(cfg, policy, device="cpu")
    trainer.prep_training()
    assert policy.actor.training
    assert policy.actor_meta.training
    trainer.prep_rollout()
    assert not policy.actor.training
    assert not policy.actor_meta.training


def test_cal_value_loss_returns_scalar(cfg, policy):
    trainer = MAPPOTrainer(cfg, policy, device="cpu")
    B = 16
    values = torch.randn(B, 1)
    value_preds = torch.randn(B, 1)
    returns = torch.randn(B, 1)
    masks = torch.ones(B, 1)
    loss = trainer.cal_value_loss(values, value_preds, returns, masks)
    assert loss.dim() == 0  # scalar
    assert torch.isfinite(loss)


def test_train_baseline_no_meta(cfg, policy):
    """Baseline train run — no wager signal, no meta update."""
    trainer = MAPPOTrainer(cfg, policy, device="cpu")
    trainer.prep_training()

    buffer = _make_fake_buffer()
    info = trainer.train(buffer, wager_objective=None, meta=False)

    assert "value_loss" in info
    assert "policy_loss" in info
    assert "dist_entropy" in info
    assert "actor_grad_norm" in info
    assert "critic_grad_norm" in info
    assert "ratio" in info
    assert info["wager_loss_actor"] == 0.0
    assert info["wager_loss_critic"] == 0.0
    assert all(np.isfinite(v) for v in info.values())


def test_train_meta_requires_wager_objective(cfg, policy):
    trainer = MAPPOTrainer(cfg, policy, device="cpu")
    trainer.prep_training()
    buffer = _make_fake_buffer()
    with pytest.raises(ValueError, match="wager_objective"):
        trainer.train(buffer, wager_objective=None, meta=True)


def test_train_meta_path_produces_wager_loss(cfg, policy):
    """meta=True path logs wager_loss > 0 (BCE on random targets)."""
    trainer = MAPPOTrainer(cfg, policy, device="cpu")
    trainer.prep_training()
    buffer = _make_fake_buffer()

    # wager_objective shape : (B, 2) one-hot ; student uses binary {0,1} broadcast.
    # We use proper (B, 2) one-hot labels.
    B = EPISODE_LEN * N_AGENTS
    wager_obj = np.zeros((B, 2), dtype=np.float32)
    wager_obj[np.arange(B) % 2, 0] = 1.0  # alternating high/low
    wager_obj[np.arange(B) % 2 == 0, 1] = 1.0

    info = trainer.train(buffer, wager_objective=wager_obj, meta=True)

    assert info["wager_loss_actor"] > 0
    assert info["wager_loss_critic"] > 0


def test_train_with_meta_false_skips_wager_optimizers(cfg, policy):
    """meta=False : actor_meta / critic_meta optimizers are NOT stepped."""
    trainer = MAPPOTrainer(cfg, policy, device="cpu")
    trainer.prep_training()
    buffer = _make_fake_buffer()

    # Snapshot actor_meta parameters.
    snapshot = {k: v.detach().clone() for k, v in policy.actor_meta.named_parameters()}

    trainer.train(buffer, wager_objective=None, meta=False)

    # actor_meta parameters should not have moved.
    for k, v in policy.actor_meta.named_parameters():
        assert torch.allclose(snapshot[k], v), f"actor_meta param {k} changed under meta=False"


# ──────────────────────────────────────────────────────────────
# E.17c regression : meta=True with asymmetric obs vs share_obs shapes
# ──────────────────────────────────────────────────────────────


def test_train_meta_path_handles_asymmetric_obs_share_obs(cfg):
    """Regression : actor_meta is built with ``obs_shape`` (per-agent RGB,
    e.g. 11×11×3). Previously the critic-side wager forward passed
    ``share_obs_batch`` (centralised WORLD.RGB, e.g. 24×18×3) which blew up
    the CNN→Linear matmul on any substrate with obs ≠ share_obs. Caught at
    E.17c launch on commons_harvest_closed. This test mirrors the shape
    asymmetry with a minimal setup."""
    obs_shape = (11, 11, 3)
    share_obs_shape = (24, 18, 3)

    action_space = spaces.Discrete(8)
    policy = MAPPOPolicy(
        cfg,
        obs_shape=obs_shape,
        cent_obs_shape=share_obs_shape,  # asymmetric with obs_shape
        action_space=action_space,
        meta=True,
        cascade_iterations1=1,
        cascade_iterations2=1,
    )
    trainer = MAPPOTrainer(cfg, policy, device="cpu")
    trainer.prep_training()

    # Build a fake buffer whose obs is (11, 11, 3) and share_obs is (24, 18, 3).
    T, N = EPISODE_LEN, N_AGENTS

    class _AsymBuffer:
        def __init__(self):
            np.random.seed(42)
            self.returns = np.random.randn(T + 1, N, 1).astype(np.float32)
            self.value_preds = np.random.randn(T + 1, N, 1).astype(np.float32)
            self.active_masks = np.ones((T + 1, N, 1), dtype=np.float32)

        def recurrent_generator(self, advantages, num_mini_batch, chunk_length):
            obs = np.random.randn(T * N, *obs_shape).astype(np.float32)
            share_obs = np.random.randn(T * N, *share_obs_shape).astype(np.float32)
            actions = np.zeros((T * N, 1), dtype=np.int64)
            rnn = np.zeros((T * N, 1, HIDDEN), dtype=np.float32)
            vp = np.zeros((T * N, 1), dtype=np.float32)
            ret = np.zeros((T * N, 1), dtype=np.float32)
            masks = np.ones((T * N, 1), dtype=np.float32)
            alp = np.zeros((T * N, 1), dtype=np.float32)
            adv = np.zeros((T * N, 1), dtype=np.float32)
            yield (
                torch.from_numpy(share_obs).float(),
                torch.from_numpy(obs).float(),
                torch.from_numpy(rnn).float(),
                torch.from_numpy(rnn).float(),
                torch.from_numpy(actions).long(),
                vp,
                ret,
                torch.from_numpy(masks).float(),
                masks,
                alp,
                adv,
                None,
            )

    B = T * N
    wager_obj = np.zeros((B, 2), dtype=np.float32)
    wager_obj[:, 0] = 1.0

    info = trainer.train(_AsymBuffer(), wager_objective=wager_obj, meta=True)
    assert info["wager_loss_actor"] > 0
    assert info["wager_loss_critic"] > 0
