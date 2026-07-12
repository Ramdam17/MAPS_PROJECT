"""Tier-3: SARL+CL update step — the 3-term CL loss + two-loss (Sprint 15.F)."""

from __future__ import annotations

import copy

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from maps.domains.sarl_cl.data import Transition
from maps.domains.sarl_cl.loss_weighting import DynamicLossWeighter, LossMixingWeights
from maps.domains.sarl_cl.model import SarlCLQNetwork, SarlCLSecondOrderNetwork
from maps.domains.sarl_cl.trainer import sarl_cl_update_step
from maps.utils.seeding import set_all_seeds

IN_CHANNELS = 4
NUM_ACTIONS = 6


def _sample(n=8):
    return [
        Transition(
            torch.rand(1, IN_CHANNELS, 10, 10),
            torch.rand(1, IN_CHANNELS, 10, 10),
            torch.tensor([[i % NUM_ACTIONS]]),
            torch.tensor([[float(i % 3)]]),
            torch.tensor([[1 if i % 4 == 0 else 0]]),
        )
        for i in range(n)
    ]


def _nets_and_opt(meta=False):
    p = SarlCLQNetwork(IN_CHANNELS, NUM_ACTIONS)
    t = SarlCLQNetwork(IN_CHANNELS, NUM_ACTIONS)
    opt = SGD(p.parameters(), lr=0.01)
    sch = StepLR(opt, step_size=1, gamma=0.999)
    so = opt2 = sch2 = None
    if meta:
        so = SarlCLSecondOrderNetwork()
        opt2 = SGD(so.parameters(), lr=0.01)
        sch2 = StepLR(opt2, step_size=1, gamma=0.999)
    return p, t, so, opt, opt2, sch, sch2


def test_no_teacher_degenerates_to_plain_update():
    set_all_seeds(42)
    p, t, _, opt, _, sch, _ = _nets_and_opt()
    before = copy.deepcopy(p.state_dict())
    loss, _ls, comp, _cs = sarl_cl_update_step(
        _sample(),
        p,
        t,
        None,
        opt,
        None,
        sch,
        None,
        meta=False,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        alpha=25,
    )
    assert torch.isfinite(loss)
    assert comp.distillation == 0.0 and comp.feature == 0.0  # no teacher → task only
    assert any(not torch.equal(p.state_dict()[k], before[k]) for k in before)


def test_identical_teacher_gives_zero_distillation():
    set_all_seeds(1)
    p, t, _, opt, _, sch, _ = _nets_and_opt()
    teacher = copy.deepcopy(p)
    teacher.requires_grad_(False)
    lw = DynamicLossWeighter()
    _loss, _ls, comp, _cs = sarl_cl_update_step(
        _sample(),
        p,
        t,
        None,
        opt,
        None,
        sch,
        None,
        teacher_first_net=teacher,
        loss_weighter=lw,
        mixing=LossMixingWeights(),
        meta=False,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        alpha=25,
        train=False,
    )
    # student == teacher at this step → L2 anchor and feature MSE are 0
    assert comp.distillation == 0.0
    assert comp.feature == 0.0


def test_different_teacher_gives_positive_distillation():
    set_all_seeds(2)
    p, t, _, opt, _, sch, _ = _nets_and_opt()
    teacher = copy.deepcopy(p)
    with torch.no_grad():
        for param in teacher.parameters():
            param.add_(0.5)  # perturb teacher
    teacher.requires_grad_(False)
    lw = DynamicLossWeighter()
    _loss, _ls, comp, _cs = sarl_cl_update_step(
        _sample(),
        p,
        t,
        None,
        opt,
        None,
        sch,
        None,
        teacher_first_net=teacher,
        loss_weighter=lw,
        meta=False,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        alpha=25,
        train=False,
    )
    assert comp.distillation > 0.0  # L2 drift vs perturbed teacher


def test_meta_with_teachers_updates_both_nets():
    set_all_seeds(3)
    p, t, so, opt, opt2, sch, sch2 = _nets_and_opt(meta=True)
    teacher_f = copy.deepcopy(p)
    teacher_f.requires_grad_(False)
    teacher_s = copy.deepcopy(so)
    teacher_s.requires_grad_(False)
    lw, lw2 = DynamicLossWeighter(), DynamicLossWeighter()
    p_before = copy.deepcopy(p.state_dict())
    so_before = copy.deepcopy(so.state_dict())
    loss, loss_second, _cf, _cs = sarl_cl_update_step(
        _sample(),
        p,
        t,
        so,
        opt,
        opt2,
        sch,
        sch2,
        teacher_first_net=teacher_f,
        teacher_second_net=teacher_s,
        loss_weighter=lw,
        loss_weighter_second=lw2,
        meta=True,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        alpha=25,
    )
    assert torch.isfinite(loss) and torch.isfinite(loss_second)
    assert any(not torch.equal(p.state_dict()[k], p_before[k]) for k in p_before)
    assert any(not torch.equal(so.state_dict()[k], so_before[k]) for k in so_before)
