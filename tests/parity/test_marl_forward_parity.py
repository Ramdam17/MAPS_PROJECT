"""MAPPO forward-pass bit-exact parity vs student (E.13, tier-1).

Verifies that our port's ``MAPPOActor`` / ``MAPPOCritic`` produce numerically
identical outputs to the student reference (stripped, vendored under
``tests/parity/_student_ref/marl/``) given :

1. identical initialisation (copy state_dict ref → ours),
2. identical inputs (same seeded obs / rnn_states / masks).

atol = 1e-6 on the forward outputs (actions, log_probs, values, rnn_states).

**Known port deviation — rnn_states layout.** Student ``R_Actor.forward`` exits
with ``rnn_states = rnn_states.permute(1, 0, 2)`` (L117-118 of r_actor_critic.py),
yielding shape ``(N_recurrent, batch, hidden)``. Our ``MAPPOActor`` drops this
permute so the buffer / runner carry the more natural ``(batch, N_recurrent,
hidden)`` layout end-to-end. For the parity test we apply the permute on
the ref side before comparison — documented below as ``_RNN_PERMUTE_NOTE``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

# ensure the vendored student ref imports cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "_student_ref"))

from marl.act import ACTLayer as RefACTLayer  # noqa: E402
from marl.cnn import CNNBase as RefCNNBase  # noqa: E402
from marl.r_actor import R_Actor, R_Critic  # noqa: E402
from marl.rnn import RNNLayer as RefRNNLayer  # noqa: E402

from maps.experiments.marl.act import ACTLayer as OursACTLayer  # noqa: E402
from maps.experiments.marl.encoder import CNNBase as OursCNNBase  # noqa: E402
from maps.experiments.marl.policy import MAPPOActor, MAPPOCritic  # noqa: E402
from maps.experiments.marl.rnn import RNNLayer as OursRNNLayer  # noqa: E402
from maps.utils import load_config  # noqa: E402


_RNN_PERMUTE_NOTE = (
    "Student R_Actor.forward permutes rnn_states (batch, N, H) → (N, batch, H) "
    "on exit — our port drops this. Apply .permute(1, 0, 2) on ref before compare."
)

ATOL = 1e-6
HIDDEN = 32
OBS_SHAPE = (11, 11, 3)
BATCH = 4
N_RECURRENT = 1
N_ACTIONS = 8


# ──────────────────────────────────────────────────────────────
# CNN parity
# ──────────────────────────────────────────────────────────────


def test_cnn_forward_bit_exact_vs_student():
    torch.manual_seed(0)
    ref = RefCNNBase(
        obs_shape=OBS_SHAPE, hidden_size=HIDDEN, use_orthogonal=True, use_ReLU=True
    )
    torch.manual_seed(0)
    ours = OursCNNBase(
        obs_shape=OBS_SHAPE, hidden_size=HIDDEN, use_orthogonal=True, use_ReLU=True
    )
    ours.load_state_dict(ref.state_dict())
    ref.eval()
    ours.eval()

    obs = torch.randint(0, 256, (BATCH, *OBS_SHAPE), dtype=torch.uint8).float()
    with torch.no_grad():
        out_ref = ref(obs)
        out_ours = ours(obs)
    assert out_ref.shape == (BATCH, HIDDEN)
    assert torch.allclose(out_ref, out_ours, atol=ATOL), (
        f"CNN forward diverges ; max abs diff = {(out_ref - out_ours).abs().max()}"
    )


# ──────────────────────────────────────────────────────────────
# RNN parity
# ──────────────────────────────────────────────────────────────


def test_rnn_forward_bit_exact_vs_student_rollout_mode():
    """Rollout path : ``x.size(0) == hxs.size(0)`` (single-step forward)."""
    torch.manual_seed(1)
    ref = RefRNNLayer(HIDDEN, HIDDEN, recurrent_N=N_RECURRENT, use_orthogonal=True)
    torch.manual_seed(1)
    ours = OursRNNLayer(HIDDEN, HIDDEN, recurrent_n=N_RECURRENT, use_orthogonal=True)
    ours.load_state_dict(ref.state_dict())
    ref.eval()
    ours.eval()

    x = torch.randn(BATCH, HIDDEN)
    hxs = torch.randn(BATCH, N_RECURRENT, HIDDEN)
    masks = torch.ones(BATCH, 1)

    with torch.no_grad():
        out_ref, hxs_ref, casc_ref = ref(x, hxs, masks, None, 1.0)
        out_ours, hxs_ours, casc_ours = ours(x, hxs, masks, None, 1.0)

    assert torch.allclose(out_ref, out_ours, atol=ATOL)
    assert torch.allclose(hxs_ref, hxs_ours, atol=ATOL)
    assert torch.allclose(casc_ref, casc_ours, atol=ATOL)


def test_rnn_cascade_update_bit_exact_vs_student():
    """Cascade loop (cascade_rate=0.5) → ensures eq.6 update matches exactly."""
    torch.manual_seed(2)
    ref = RefRNNLayer(HIDDEN, HIDDEN, recurrent_N=N_RECURRENT, use_orthogonal=True)
    torch.manual_seed(2)
    ours = OursRNNLayer(HIDDEN, HIDDEN, recurrent_n=N_RECURRENT, use_orthogonal=True)
    ours.load_state_dict(ref.state_dict())
    ref.eval()
    ours.eval()

    x = torch.randn(BATCH, HIDDEN)
    hxs = torch.randn(BATCH, N_RECURRENT, HIDDEN)
    masks = torch.ones(BATCH, 1)
    prev = torch.randn(BATCH, HIDDEN)

    with torch.no_grad():
        _, _, casc_ref = ref(x, hxs, masks, prev, 0.5)
        _, _, casc_ours = ours(x, hxs, masks, prev, 0.5)
    assert torch.allclose(casc_ref, casc_ours, atol=ATOL)


# ──────────────────────────────────────────────────────────────
# ACT parity (deterministic mode → mode() argmax — no RNG dependency)
# ──────────────────────────────────────────────────────────────


def test_act_forward_bit_exact_vs_student_deterministic():
    torch.manual_seed(3)
    action_space = spaces.Discrete(N_ACTIONS)
    ref = RefACTLayer(action_space, HIDDEN, use_orthogonal=True, gain=0.01)
    torch.manual_seed(3)
    ours = OursACTLayer(action_space, HIDDEN, use_orthogonal=True, gain=0.01)
    ours.load_state_dict(ref.state_dict())

    x = torch.randn(BATCH, HIDDEN)
    with torch.no_grad():
        a_ref, lp_ref = ref(x, available_actions=None, deterministic=True)
        a_ours, lp_ours = ours(x, available_actions=None, deterministic=True)
    assert torch.equal(a_ref, a_ours), "argmax actions diverge"
    assert torch.allclose(lp_ref, lp_ours, atol=ATOL)


# ──────────────────────────────────────────────────────────────
# Full R_Actor / R_Critic parity
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def args_ns():
    """Student R_Actor/R_Critic consume an argparse.Namespace — stub the
    non-attention, recurrent, no-PopArt path that our port targets."""
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
    )


@pytest.fixture
def cfg():
    return load_config(
        "training/marl",
        overrides=[
            f"model.hidden_size={HIDDEN}",
            "model.second_order_dropout=0.0",
        ],
    )


def _weight_map_ref_to_ours_actor(ref_state_dict: dict) -> dict:
    """Remap student R_Actor state_dict keys to MAPPOActor keys.

    Student ``R_Actor`` submodules : ``base``, ``rnn``, ``act``.
    Our ``MAPPOActor`` submodules (same names). The difference is
    ``base.cnn.cnn`` vs ``base.cnn.cnn`` — both wrap a CNNLayer named ``cnn``.
    Keys match 1-to-1.
    """
    # Our MAPPOActor has: base.cnn (CNNLayer), rnn, act.
    # Student's R_Actor has the SAME layout (CNNBase.cnn (CNNLayer), rnn, act).
    # Direct copy.
    return dict(ref_state_dict)


def test_actor_forward_bit_exact_vs_student(args_ns, cfg):
    obs_space = spaces.Box(low=0, high=255, shape=OBS_SHAPE, dtype=np.float32)
    action_space = spaces.Discrete(N_ACTIONS)

    torch.manual_seed(10)
    ref = R_Actor(args_ns, obs_space, action_space, device=torch.device("cpu"))
    torch.manual_seed(10)
    ours = MAPPOActor(cfg, obs_shape=OBS_SHAPE, action_space=action_space, cascade_iterations=1)

    ours.load_state_dict(_weight_map_ref_to_ours_actor(ref.state_dict()))
    ref.eval()
    ours.eval()

    obs = torch.randint(0, 256, (BATCH, *OBS_SHAPE), dtype=torch.uint8).float()
    rnn_states = torch.randn(BATCH, N_RECURRENT, HIDDEN)
    masks = torch.ones(BATCH, 1)

    with torch.no_grad():
        a_ref, lp_ref, h_ref = ref(obs, rnn_states, masks, deterministic=True)
        a_ours, lp_ours, h_ours = ours(obs, rnn_states, masks, deterministic=True)

    # Actions (argmax in deterministic mode) must match bit-for-bit.
    assert torch.equal(a_ref, a_ours), "actor actions diverge"
    assert torch.allclose(lp_ref, lp_ours, atol=ATOL)

    # rnn_states : ref.permute(1, 0, 2) → ours (see _RNN_PERMUTE_NOTE).
    assert torch.allclose(h_ref.permute(1, 0, 2), h_ours, atol=ATOL), _RNN_PERMUTE_NOTE


def test_critic_forward_bit_exact_vs_student(args_ns, cfg):
    cent_obs_space = spaces.Box(low=0, high=255, shape=OBS_SHAPE, dtype=np.float32)

    torch.manual_seed(11)
    ref = R_Critic(args_ns, cent_obs_space, device=torch.device("cpu"))
    torch.manual_seed(11)
    ours = MAPPOCritic(cfg, cent_obs_shape=OBS_SHAPE, cascade_iterations=1)

    ours.load_state_dict(dict(ref.state_dict()))
    ref.eval()
    ours.eval()

    cent_obs = torch.randint(0, 256, (BATCH, *OBS_SHAPE), dtype=torch.uint8).float()
    rnn_states = torch.randn(BATCH, N_RECURRENT, HIDDEN)
    masks = torch.ones(BATCH, 1)

    with torch.no_grad():
        v_ref, h_ref = ref(cent_obs, rnn_states, masks)
        v_ours, h_ours = ours(cent_obs, rnn_states, masks)
    assert torch.allclose(v_ref, v_ours, atol=ATOL)
    assert torch.allclose(h_ref.permute(1, 0, 2), h_ours, atol=ATOL), _RNN_PERMUTE_NOTE


# ──────────────────────────────────────────────────────────────
# Cascade > 1 iteration parity
# ──────────────────────────────────────────────────────────────


def test_actor_forward_bit_exact_cascade_4(args_ns, cfg):
    """Cascade_iterations > 1 : make sure the cascade loop in our MAPPOActor
    matches student's loop."""
    args_ns.cascade_iterations1 = 4

    obs_space = spaces.Box(low=0, high=255, shape=OBS_SHAPE, dtype=np.float32)
    action_space = spaces.Discrete(N_ACTIONS)

    torch.manual_seed(20)
    ref = R_Actor(args_ns, obs_space, action_space, device=torch.device("cpu"))
    torch.manual_seed(20)
    ours = MAPPOActor(cfg, obs_shape=OBS_SHAPE, action_space=action_space, cascade_iterations=4)

    ours.load_state_dict(dict(ref.state_dict()))
    ref.eval()
    ours.eval()

    obs = torch.randint(0, 256, (BATCH, *OBS_SHAPE), dtype=torch.uint8).float()
    rnn_states = torch.randn(BATCH, N_RECURRENT, HIDDEN)
    masks = torch.ones(BATCH, 1)

    with torch.no_grad():
        a_ref, lp_ref, h_ref = ref(obs, rnn_states, masks, deterministic=True)
        a_ours, lp_ours, h_ours = ours(obs, rnn_states, masks, deterministic=True)
    assert torch.equal(a_ref, a_ours)
    assert torch.allclose(lp_ref, lp_ours, atol=ATOL)
    assert torch.allclose(h_ref.permute(1, 0, 2), h_ours, atol=ATOL)
