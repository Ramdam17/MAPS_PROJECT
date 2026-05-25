"""Parity test : :class:`ACBTrainer._update` ≡ reference ``acb_train_step``.

Strategy
--------
1. Build both networks with identical seeds → identical initial parameters.
2. Construct a small batch of fake ``Transition`` samples (no real MinAtar
   env is needed for the math — only tensor shapes matter).
3. Drive ``N`` update steps on both, with identical inputs and identical
   hyperparameter values.
4. After ``N`` steps, assert :

   - ``network.parameters()`` bit-identical (atol = 0, rtol = 0).
   - Eligibility traces bit-identical.
   - RMSprop MSG buffers bit-identical.

The reference's ``world_dynamics`` involves a stochastic ``multinomial``
sample over the policy distribution. We bypass it here by hand-crafting
transitions, so the parity test does not depend on RNG sync — only on the
arithmetic of the update rule itself.
"""

from __future__ import annotations

import pytest
import torch

from maps.experiments.sarl.actor_critic import (
    ACBConfig,
    ACBTrainer,
    ACNetwork as PortedACNetwork,
    Transition,
)
from tests.parity.sarl._reference_acb import (
    ACNetwork as ReferenceACNetwork,
    acb_train_step,
    transition as RefTransition,
)


# Shapes mirror MinAtar Breakout : (1, 4, 10, 10).
IN_CHANNELS = 4
NUM_ACTIONS = 4
NUM_STEPS = 30  # enough to exercise the time-step debiasing term meaningfully
ALPHA = 0.00048828125
LAMBDA = 0.8
GAMMA = 0.99
BETA = 0.01
GAMMA_RMS = 0.999
EPS_RMS = 0.0001
MIN_DENOM = 0.0001


def _seeded_pair_of_networks(seed: int) -> tuple[PortedACNetwork, ReferenceACNetwork]:
    """Build a port + reference network with identical initial weights.

    Because the two implementations have the same module shapes in the same
    order, we can `state_dict()` from one and `load_state_dict()` into the
    other — guaranteeing identical init regardless of how PyTorch ordered
    the underlying RNG draws during ``__init__``.
    """
    torch.manual_seed(seed)
    ported = PortedACNetwork(IN_CHANNELS, NUM_ACTIONS)
    ref = ReferenceACNetwork(IN_CHANNELS, NUM_ACTIONS)
    ref.load_state_dict(ported.state_dict())
    return ported, ref


def _make_fake_transitions(num_steps: int, seed: int) -> list[Transition]:
    """Deterministic synthetic transitions — same on every call.

    Each step produces a random observation tensor and an action sampled
    from a fixed policy. ``last_state`` is None for step 0 (mirrors the
    reference's initial-step branch), then the previous step's state.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    transitions = []
    last_state = None
    last_reward = None
    last_terminal = None
    for k in range(num_steps):
        state = torch.rand((1, IN_CHANNELS, 10, 10), generator=g)
        # Deterministic action choice (cycle 0..NUM_ACTIONS-1).
        action = torch.tensor([k % NUM_ACTIONS], dtype=torch.long)
        # Reward is a small float, terminal every 7 steps.
        reward = torch.tensor([[(k % 3) * 0.5 - 0.5]])
        terminal = torch.tensor([[(k + 1) % 7 == 0]])
        transitions.append(
            Transition(state, last_state, action, last_reward, last_terminal)
        )
        last_state = state
        last_reward = reward
        last_terminal = terminal
    return transitions


@pytest.mark.parametrize("seed", [0, 42, 137])
def test_acb_update_matches_reference(seed: int) -> None:
    """N steps of ACBTrainer._update produce bit-identical weights vs ref."""
    ported_net, ref_net = _seeded_pair_of_networks(seed)

    # Build the trainer-side buffers using ported_net's parameters.
    cfg = ACBConfig(
        game="breakout",
        seed=seed,
        num_frames=NUM_STEPS,
        output_dir=__file__,  # unused — train() is never called
        alpha=ALPHA, lambda_=LAMBDA, gamma=GAMMA, beta=BETA,
        gamma_rms=GAMMA_RMS, eps_rms=EPS_RMS, min_denom=MIN_DENOM,
    )
    # ACBTrainer.__init__ wants an env — pass None and assign network/buffers
    # manually to bypass env-touching code.
    trainer = ACBTrainer.__new__(ACBTrainer)
    trainer.cfg = cfg
    trainer.env = None
    trainer.device = torch.device("cpu")
    trainer.network = ported_net
    trainer.traces = [torch.zeros_like(p) for p in ported_net.parameters()]
    trainer.grads = [torch.zeros_like(p) for p in ported_net.parameters()]
    trainer.msg = [torch.zeros_like(p) for p in ported_net.parameters()]

    # Build the reference buffers from ref_net's parameters (same shapes).
    ref_traces = [torch.zeros_like(p) for p in ref_net.parameters()]
    ref_grads = [torch.zeros_like(p) for p in ref_net.parameters()]
    ref_msg = [torch.zeros_like(p) for p in ref_net.parameters()]

    transitions = _make_fake_transitions(NUM_STEPS, seed=seed)

    for t, port_sample in enumerate(transitions):
        # Port update.
        trainer._update(port_sample, t)
        # Reference update — same sample, namedtuple shape conversion.
        ref_sample = RefTransition(
            state=port_sample.state,
            last_state=port_sample.last_state,
            action=port_sample.action,
            reward=port_sample.reward,
            is_terminal=port_sample.is_terminal,
        )
        acb_train_step(
            ref_sample, ref_traces, ref_grads, ref_msg, ref_net, ALPHA, t,
            LAMBDA=LAMBDA, GAMMA=GAMMA, BETA=BETA,
            GAMMA_RMS=GAMMA_RMS, EPS_RMS=EPS_RMS, MIN_DENOM=MIN_DENOM,
        )

    # Parameter parity.
    for name_p, p_port, p_ref in zip(
        [n for n, _ in ported_net.named_parameters()],
        ported_net.parameters(),
        ref_net.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            p_port, p_ref, atol=0, rtol=0,
            msg=f"parameter {name_p!r} diverged after {NUM_STEPS} steps (seed={seed})",
        )

    # Trace parity.
    for k, (t_port, t_ref) in enumerate(zip(trainer.traces, ref_traces, strict=True)):
        torch.testing.assert_close(
            t_port, t_ref, atol=0, rtol=0,
            msg=f"trace #{k} diverged after {NUM_STEPS} steps (seed={seed})",
        )

    # RMSprop MSG buffer parity.
    for k, (m_port, m_ref) in enumerate(zip(trainer.msg, ref_msg, strict=True)):
        torch.testing.assert_close(
            m_port, m_ref, atol=0, rtol=0,
            msg=f"MSG #{k} diverged after {NUM_STEPS} steps (seed={seed})",
        )


def test_ac_network_forward_shapes() -> None:
    """ACNetwork output shapes match the reference for MinAtar inputs."""
    ported, ref = _seeded_pair_of_networks(seed=0)
    x = torch.rand(1, IN_CHANNELS, 10, 10)

    pi_port, v_port = ported(x)
    pi_ref, v_ref = ref(x)

    assert pi_port.shape == (1, NUM_ACTIONS) == pi_ref.shape
    assert v_port.shape == (1, 1) == v_ref.shape
    torch.testing.assert_close(pi_port, pi_ref, atol=0, rtol=0)
    torch.testing.assert_close(v_port, v_ref, atol=0, rtol=0)
    # Policy is a probability distribution.
    torch.testing.assert_close(
        pi_port.sum(dim=1),
        torch.ones(1),
        atol=1e-6, rtol=0,
    )
