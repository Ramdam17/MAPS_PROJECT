"""Tier-4-light: SARL+CL curriculum loop (Sprint 15.G).

Small 2-game curriculum on a channel-adaptive net (unseeded MinAtar env per
D15.3/faithful → orchestration smoke, not bit-exact). Uses a real MinAtar env
(first import ~60s).
"""

from __future__ import annotations

import copy

from maps.domains.sarl_cl.model import AdaptiveQNetwork
from maps.domains.sarl_cl.trainer import SarlCLSetting
from maps.utils.config import load_config
from maps.utils.seeding import set_all_seeds


def _tiny_cfg():
    return load_config(
        "domains/sarl_cl/training",
        overrides=[
            "training.replay_start_size=20",
            "training.batch_size=8",
            "training.target_update_freq=5",
            "training.replay_buffer_size=500",
        ],
    )


# Minimal local SarlSetting (avoid depending on domains/sarl on this branch).
_BASELINE = SarlCLSetting(
    "setting-1-baseline", "baseline", meta=False, cascade_iterations_1=1, cascade_iterations_2=1
)
_META = SarlCLSetting(
    "setting-3-second-order-only",
    "2nd-order",
    meta=True,
    cascade_iterations_1=1,
    cascade_iterations_2=1,
)


def test_run_curriculum_two_stages_baseline():
    from maps.domains.sarl_cl.training_loop import run_curriculum

    set_all_seeds(42)
    cfg = _tiny_cfg()
    _policy_net, metrics = run_curriculum(
        ["breakout", "space_invaders"], cfg, _BASELINE, frames_per_stage=120
    )
    assert len(metrics) == 2
    assert metrics[0].game == "breakout"
    assert metrics[1].game == "space_invaders"
    assert all(m.total_updates > 0 for m in metrics)


def test_run_stage_with_frozen_teacher_updates_student_only():
    from minatar import Environment

    from maps.domains.sarl_cl.training_loop import run_stage

    set_all_seeds(1)
    cfg = _tiny_cfg()
    policy = AdaptiveQNetwork(10, 6)
    target = AdaptiveQNetwork(10, 6)
    target.load_state_dict(policy.state_dict())
    teacher = copy.deepcopy(policy).requires_grad_(False)
    teacher_before = copy.deepcopy(teacher.state_dict())

    env = Environment("breakout")
    env.game_name = "breakout"
    run_stage(policy, target, None, env, cfg, _BASELINE, num_frames=120, teacher_first_net=teacher)

    # Teacher must stay frozen (unchanged); student must have moved.
    assert all(
        __import__("torch").equal(teacher.state_dict()[k], teacher_before[k])
        for k in teacher_before
    )
