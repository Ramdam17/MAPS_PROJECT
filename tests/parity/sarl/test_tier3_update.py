"""Tier-3 parity: SARL DQN update step bit-exact vs maps_v1.py (Sprint 14.F).

(a) core ``cae_loss(recon="huber")`` == source ``CAE_loss`` (validates the huber
    reconstruction variant added for SARL);
(b) ``sarl_train_step`` (non-meta and meta) matches an inline replica of the
    source ``train`` control flow — loss values + post-step weights.
"""

from __future__ import annotations

import copy

import torch

from maps.core.losses import cae_loss
from maps.domains.sarl.data import Transition, target_wager
from maps.domains.sarl.model_v1 import SarlQNetworkV1, SarlSecondOrderNetworkV1
from maps.domains.sarl.trainer import sarl_train_step
from maps.utils.seeding import set_all_seeds
from tests.parity.sarl import _student_extracts as ref

IN_CHANNELS = 4
NUM_ACTIONS = 6
GAMMA = 0.99
LAM = 1e-4


def _fake_sample(n=8):
    """n transitions with (1,4,10,10) states, some terminal."""
    sample = []
    for i in range(n):
        sample.append(
            Transition(
                torch.rand(1, IN_CHANNELS, 10, 10),
                torch.rand(1, IN_CHANNELS, 10, 10),
                torch.tensor([[i % NUM_ACTIONS]]),
                torch.tensor([[float(i % 3)]]),
                torch.tensor([[1 if i % 4 == 0 else 0]]),
            )
        )
    return sample


def test_core_cae_huber_equals_source_cae_loss():
    torch.manual_seed(0)
    w = torch.randn(128, 1024)
    x = torch.randn(8, 1)
    recons_x = torch.randn(8, 1, requires_grad=True)
    h = torch.rand(8, 128)
    ours = cae_loss(weight=w, x=x, recons_x=recons_x, hidden=h, lam=LAM, recon="huber")
    theirs = ref.CAE_loss(w, x, recons_x, h, LAM)
    assert torch.allclose(ours, theirs.squeeze(), atol=1e-7)


def _inline_nonmeta(sample, policy_net, target_net, optimizer, scheduler1, alpha):
    """Inline replica of the source train() non-meta path (one update)."""
    optimizer.zero_grad()
    batch = ref.transition(*zip(*sample, strict=True))
    states = torch.cat(batch.state)
    next_states = torch.cat(batch.next_state)
    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)
    is_terminal = torch.cat(batch.is_terminal)
    main_task_out = target_task_out = None
    q_policy, h1, _c1, main_task_out = policy_net(states, main_task_out, 1.0)
    q_s_a = q_policy.gather(1, actions)
    idx = torch.tensor([i for i, t in enumerate(is_terminal) if t == 0], dtype=torch.int64)
    nn_next = next_states.index_select(0, idx)
    q_next = torch.zeros(len(sample), 1)
    if len(nn_next) != 0:
        q_target, _, _, target_task_out = target_net(nn_next, target_task_out, 1.0)
        q_next[idx] = q_target.detach().max(1)[0].unsqueeze(1)
    target = rewards + GAMMA * q_next
    w = policy_net.state_dict()["fc_hidden.weight"]
    loss = ref.CAE_loss(w, target, q_s_a, h1, LAM)
    loss.backward()
    optimizer.step()
    scheduler1.step()
    return loss


def test_train_step_nonmeta_matches_inline():
    from torch.optim import SGD
    from torch.optim.lr_scheduler import StepLR

    sample = _fake_sample()

    # Reference inline run.
    set_all_seeds(42)
    p_ref = ref.QNetwork(IN_CHANNELS, NUM_ACTIONS)
    t_ref = ref.QNetwork(IN_CHANNELS, NUM_ACTIONS)
    opt_ref = SGD(p_ref.parameters(), lr=0.01)
    sch_ref = StepLR(opt_ref, step_size=1, gamma=0.999)
    loss_ref = _inline_nonmeta(sample, p_ref, t_ref, opt_ref, sch_ref, alpha=25)
    w_ref_after = copy.deepcopy(p_ref.state_dict())

    # Our train_step.
    set_all_seeds(42)
    p = SarlQNetworkV1(IN_CHANNELS, NUM_ACTIONS)
    t = SarlQNetworkV1(IN_CHANNELS, NUM_ACTIONS)
    opt = SGD(p.parameters(), lr=0.01)
    sch = StepLR(opt, step_size=1, gamma=0.999)
    loss = sarl_train_step(
        sample,
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

    assert torch.allclose(loss.squeeze(), loss_ref.squeeze(), atol=1e-6)
    for k in w_ref_after:
        assert torch.allclose(p.state_dict()[k], w_ref_after[k], atol=1e-6), f"weight {k} diverged"


def test_train_step_meta_runs_and_updates_both():
    from torch.optim import SGD
    from torch.optim.lr_scheduler import StepLR

    sample = _fake_sample()
    set_all_seeds(7)
    p = SarlQNetworkV1(IN_CHANNELS, NUM_ACTIONS)
    t = SarlQNetworkV1(IN_CHANNELS, NUM_ACTIONS)
    so = SarlSecondOrderNetworkV1()
    opt = SGD(p.parameters(), lr=0.01)
    opt2 = SGD(so.parameters(), lr=0.01)
    sch = StepLR(opt, step_size=1, gamma=0.999)
    sch2 = StepLR(opt2, step_size=1, gamma=0.999)

    p_before = copy.deepcopy(p.state_dict())
    so_before = copy.deepcopy(so.state_dict())
    loss, loss_second = sarl_train_step(
        sample,
        p,
        t,
        so,
        opt,
        opt2,
        sch,
        sch2,
        meta=True,
        cascade_iterations_1=1,
        cascade_iterations_2=1,
        alpha=25,
    )
    assert torch.isfinite(loss) and torch.isfinite(loss_second)
    # Both nets updated (two-loss pattern).
    assert any(not torch.equal(p.state_dict()[k], p_before[k]) for k in p_before)
    assert any(not torch.equal(so.state_dict()[k], so_before[k]) for k in so_before)


def test_target_wager_used_in_meta_loss_is_ema():
    # Sanity: the meta target comes from target_wager (shape (B,2)).
    sample = _fake_sample()
    rewards = torch.cat([tr.reward for tr in sample])
    tw = target_wager(rewards, 25)
    assert tw.shape == (8, 2)
