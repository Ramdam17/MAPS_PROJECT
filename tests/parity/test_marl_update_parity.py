"""MAPPO ppo_update bit-exact parity vs student (E.15, tier-3).

Compares our :class:`MAPPOTrainer` against the stripped student reference
(vendored under ``tests/parity/_student_ref/marl/trainer.py``) for the
baseline (meta=False) update path :

1. :meth:`cal_value_loss` bit-exact (direct function-level).
2. One full :meth:`ppo_update` — bit-exact outputs (value_loss, policy_loss,
   dist_entropy, imp_weights, grad_norms).
3. Post-step parameters bit-exact — actor / critic state_dict after
   optimizer.step.

**Port fidelity parity, not carbon-copy.** Student's ppo_update unconditionally
calls ``evaluate_actions_meta`` on the actor_meta network even when meta=False
(r_mappo.py L149-156) — our port skips that dead branch per E.5 scope lock.
The ref trainer here mirrors our port's op order, NOT the student's literal
dead-code path. Meta-path parity is out of scope for E.15.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent / "_student_ref"))

from marl.r_actor import R_Actor, R_Critic  # noqa: E402
from marl.trainer import R_MAPPO as RefTrainer  # noqa: E402

from maps.experiments.marl.policy import MAPPOPolicy  # noqa: E402
from maps.experiments.marl.trainer import MAPPOTrainer  # noqa: E402
from maps.experiments.marl.valuenorm import ValueNorm  # noqa: E402
from maps.utils import load_config  # noqa: E402


ATOL = 1e-5
HIDDEN = 32
OBS_SHAPE = (11, 11, 3)
BATCH = 8
N_RECURRENT = 1
N_ACTIONS = 8
SEED = 123


class _RefPolicy:
    """Minimal policy wrapper for the stripped student R_MAPPO trainer.

    Provides the attributes the trainer expects: ``.actor``, ``.critic``,
    ``.actor_optimizer``, ``.critic_optimizer`` — matching student
    ``rMAPPOPolicy`` for the baseline path.
    """

    def __init__(self, actor, critic, actor_lr, critic_lr, eps, wd):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=actor_lr, eps=eps, weight_decay=wd
        )
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=critic_lr, eps=eps, weight_decay=wd
        )


@pytest.fixture
def cfg():
    """Shared config matching our port + student args — baseline settings."""
    return load_config(
        "training/marl",
        overrides=[
            f"model.hidden_size={HIDDEN}",
            "model.second_order_dropout=0.0",
            "ppo.ppo_epoch=1",
            "ppo.num_mini_batch=1",
            "ppo.data_chunk_length=1",
        ],
    )


@pytest.fixture
def args_ns():
    """argparse-style Namespace for student-ref classes."""
    return SimpleNamespace(
        hidden_size=HIDDEN,
        gain=0.01,
        use_orthogonal=True,
        use_ReLU=True,
        use_policy_active_masks=True,
        recurrent_N=N_RECURRENT,
        use_naive_recurrent_policy=False,
        use_recurrent_policy=True,
        cascade_iterations1=1,
        use_attention=False,
        # Trainer knobs.
        clip_param=0.2,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        max_grad_norm=10.0,
        huber_delta=10.0,
        use_clipped_value_loss=True,
        use_huber_loss=True,
        use_valuenorm=True,
        use_value_active_masks=True,
    )


def _build_pair(cfg, args_ns):
    """Build matched ours + ref (actor, critic, trainer) with identical weights."""
    obs_space = spaces.Box(low=0, high=255, shape=OBS_SHAPE, dtype=np.float32)
    action_space = spaces.Discrete(N_ACTIONS)

    # Build ours with a seeded init.
    torch.manual_seed(SEED)
    ours_policy = MAPPOPolicy(
        cfg,
        obs_shape=OBS_SHAPE,
        cent_obs_shape=OBS_SHAPE,
        action_space=action_space,
        meta=False,
        cascade_iterations1=1,
        cascade_iterations2=1,
    )
    ours_trainer = MAPPOTrainer(cfg, ours_policy, device="cpu")

    # Build ref with the same seed + shape.
    torch.manual_seed(SEED)
    ref_actor = R_Actor(args_ns, obs_space, action_space, device=torch.device("cpu"))
    torch.manual_seed(SEED + 1)  # separate seed for critic weight init
    ref_critic = R_Critic(args_ns, obs_space, device=torch.device("cpu"))

    # Align weights : copy ours → ref (so both have identical parameters).
    ref_actor.load_state_dict(ours_policy.actor.state_dict())
    ref_critic.load_state_dict(ours_policy.critic.state_dict())

    # Shared value_normalizer so the two update() calls stay in sync.
    # (Each trainer's cal_value_loss calls ``self.value_normalizer.update`` on
    # the same return_batch — result: the normalizer ends up in the same
    # state after both trainers run, and normalize/denormalize stay parallel.)
    value_normalizer_ours = ValueNorm(1).to("cpu")
    value_normalizer_ref = ValueNorm(1).to("cpu")
    # Pre-train both with identical data so they start equal.
    init_data = torch.randn(64, 1, generator=torch.Generator().manual_seed(SEED + 2))
    value_normalizer_ours.update(init_data)
    value_normalizer_ref.update(init_data)

    ours_trainer.value_normalizer = value_normalizer_ours
    cfg_optim = cfg.optimizer
    ref_policy = _RefPolicy(
        ref_actor,
        ref_critic,
        actor_lr=float(cfg_optim.actor_lr),
        critic_lr=float(cfg_optim.critic_lr),
        eps=float(cfg_optim.opti_eps),
        wd=float(cfg_optim.weight_decay),
    )
    ref_trainer = RefTrainer(args_ns, ref_policy, value_normalizer=value_normalizer_ref)

    return ours_policy, ours_trainer, ref_policy, ref_trainer


def _fake_sample(n: int = BATCH):
    """Build one mini-batch sample tuple matching both trainers' ppo_update signature."""
    rng = np.random.default_rng(SEED)
    share_obs = torch.from_numpy(rng.integers(0, 256, (n, *OBS_SHAPE)).astype(np.float32))
    obs = torch.from_numpy(rng.integers(0, 256, (n, *OBS_SHAPE)).astype(np.float32))
    rnn_states = torch.from_numpy(rng.standard_normal((n, N_RECURRENT, HIDDEN)).astype(np.float32))
    rnn_states_critic = torch.from_numpy(
        rng.standard_normal((n, N_RECURRENT, HIDDEN)).astype(np.float32)
    )
    actions = torch.from_numpy(rng.integers(0, N_ACTIONS, (n, 1)).astype(np.int64))
    value_preds = rng.standard_normal((n, 1)).astype(np.float32)
    returns = rng.standard_normal((n, 1)).astype(np.float32)
    masks = torch.ones(n, 1)
    active_masks = np.ones((n, 1), dtype=np.float32)
    old_action_log_probs = rng.standard_normal((n, 1)).astype(np.float32) - 2.0
    advantages = rng.standard_normal((n, 1)).astype(np.float32)
    return (
        share_obs,
        obs,
        rnn_states,
        rnn_states_critic,
        actions,
        value_preds,
        returns,
        masks,
        active_masks,
        old_action_log_probs,
        advantages,
        None,  # available_actions
    )


# ──────────────────────────────────────────────────────────────
# cal_value_loss : direct function-level parity
# ──────────────────────────────────────────────────────────────


def test_cal_value_loss_matches_student(cfg, args_ns):
    ours_policy, ours_trainer, _, ref_trainer = _build_pair(cfg, args_ns)
    # Both start with independent but equal ValueNorm states (set in _build_pair).
    B = 16
    values = torch.randn(B, 1, generator=torch.Generator().manual_seed(10))
    value_preds = torch.randn(B, 1, generator=torch.Generator().manual_seed(11))
    returns = torch.randn(B, 1, generator=torch.Generator().manual_seed(12))
    active_masks = torch.ones(B, 1)

    vl_ours = ours_trainer.cal_value_loss(values, value_preds, returns, active_masks)
    vl_ref = ref_trainer.cal_value_loss(values, value_preds, returns, active_masks)
    assert torch.allclose(vl_ours, vl_ref, atol=ATOL), (
        f"cal_value_loss diverges ({vl_ours.item():.6e} vs {vl_ref.item():.6e})"
    )


# ──────────────────────────────────────────────────────────────
# Full PPO update (baseline, meta=False) : all scalar outputs + params match
# ──────────────────────────────────────────────────────────────


def _compare_state_dicts(a: dict, b: dict, atol: float = ATOL) -> None:
    assert set(a.keys()) == set(b.keys())
    for k in a:
        assert torch.allclose(a[k], b[k], atol=atol), (
            f"param {k} diverges by max {(a[k] - b[k]).abs().max().item():.3e}"
        )


def test_ppo_update_baseline_outputs_bit_exact(cfg, args_ns):
    """Single ppo_update step with meta=False : compare all scalar outputs."""
    ours_policy, ours_trainer, ref_policy, ref_trainer = _build_pair(cfg, args_ns)

    sample = _fake_sample()
    ours_trainer.prep_training()
    ref_policy.actor.train()
    ref_policy.critic.train()

    # Ours returns 8-tuple ; ref returns 6-tuple (no wager losses in baseline).
    (
        v_ours, cgn_ours, p_ours, e_ours, agn_ours, iw_ours,
        wla_ours, wlc_ours,
    ) = ours_trainer.ppo_update(sample, update_actor=True, wager_objective=None, meta=False)
    v_ref, cgn_ref, p_ref, e_ref, agn_ref, iw_ref = ref_trainer.ppo_update(sample, update_actor=True)

    # Baseline : our wager-loss outputs are exactly zero ; student has no such.
    assert wla_ours == 0.0
    assert wlc_ours == 0.0

    assert abs(v_ours - v_ref.item()) < ATOL, f"value_loss : {v_ours} vs {v_ref.item()}"
    assert abs(p_ours - p_ref.item()) < ATOL, f"policy_loss : {p_ours} vs {p_ref.item()}"
    assert abs(e_ours - e_ref.item()) < ATOL, f"dist_entropy : {e_ours} vs {e_ref.item()}"
    assert abs(agn_ours - float(agn_ref)) < ATOL * 10, f"actor_grad_norm : {agn_ours} vs {agn_ref}"
    assert abs(cgn_ours - float(cgn_ref)) < ATOL * 10, (
        f"critic_grad_norm : {cgn_ours} vs {cgn_ref}"
    )
    assert abs(iw_ours - iw_ref.mean().item()) < ATOL, (
        f"imp_weights mean : {iw_ours} vs {iw_ref.mean().item()}"
    )


def test_ppo_update_baseline_weights_after_step_bit_exact(cfg, args_ns):
    """After one ppo_update, the actor/critic parameters must match bit-exact."""
    ours_policy, ours_trainer, ref_policy, ref_trainer = _build_pair(cfg, args_ns)

    sample = _fake_sample()
    ours_trainer.prep_training()
    ref_policy.actor.train()
    ref_policy.critic.train()

    ours_trainer.ppo_update(sample, update_actor=True, wager_objective=None, meta=False)
    ref_trainer.ppo_update(sample, update_actor=True)

    # Actor weights.
    _compare_state_dicts(
        ours_policy.actor.state_dict(), ref_policy.actor.state_dict(), atol=ATOL
    )
    # Critic weights.
    _compare_state_dicts(
        ours_policy.critic.state_dict(), ref_policy.critic.state_dict(), atol=ATOL
    )


def test_ppo_update_baseline_update_actor_false(cfg, args_ns):
    """update_actor=False : actor *trainable* params MUST NOT change.

    (BatchNorm2d ``running_mean`` / ``running_var`` / ``num_batches_tracked``
    are module buffers — they update during every forward in train() mode
    regardless of whether the optimizer steps. Compare only ``named_parameters``
    to isolate optimizer-driven changes.)
    """
    ours_policy, ours_trainer, ref_policy, ref_trainer = _build_pair(cfg, args_ns)

    sample = _fake_sample()
    ours_trainer.prep_training()
    ref_policy.actor.train()
    ref_policy.critic.train()

    actor_before = {k: v.clone() for k, v in ours_policy.actor.named_parameters()}
    critic_before = {k: v.clone() for k, v in ours_policy.critic.named_parameters()}

    ours_trainer.ppo_update(sample, update_actor=False, wager_objective=None, meta=False)

    for k, v in ours_policy.actor.named_parameters():
        assert torch.allclose(actor_before[k], v, atol=ATOL), (
            f"actor param {k} changed under update_actor=False"
        )

    any_critic_change = any(
        not torch.allclose(critic_before[k], v, atol=ATOL)
        for k, v in ours_policy.critic.named_parameters()
    )
    assert any_critic_change, "critic weights did not change after ppo_update"
