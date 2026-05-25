"""Paper-faithful reference for SARL Setting 7 (ACB) parity tests.

Origin
------
Stripped from ``external/paper_reference/sarl_ac_lambda.py`` (itself a
verbatim copy of ``external/MinAtar/examples/AC_lambda.py``, Young & Tian
2019). The original file globalises constants and instantiates a
``NvidiaEnergyTracker`` at import time ; for tests we need a callable that
takes its constants as arguments and has no import side-effects.

Stripped vs original source
---------------------------
- No module-level ``NvidiaEnergyTracker`` / ``MLModelEnergyEfficiency`` instantiation.
- No module-level ``print("Cuda available?")``.
- Constants accepted as kwargs of :func:`acb_train_step` rather than read
  from module globals — lets the parity test override them when needed.
- ``device`` pinned to CPU for the test (forward math is device-independent).
- ``Transition``, ``ACNetwork``, ``world_dynamics``, ``get_state`` kept
  byte-equivalent to the reference (identifiers, comment layout, casing
  preserved for clean diffs against the source).

The ``ruff: noqa`` block tolerates the paper naming style ; do not reformat.
"""
# ruff: noqa: N801, N802, N803, N806, E741, B007, SIM108, UP008, UP032, RUF001, E712, E711

# flake8: noqa

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as f

device = torch.device("cpu")


dSiLU = lambda x: torch.sigmoid(x) * (1 + x * (1 - torch.sigmoid(x)))
SiLU = lambda x: x * torch.sigmoid(x)


class ACNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):

        super(ACNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        self.policy = nn.Linear(in_features= 128, out_features=num_actions)
        self.value = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = SiLU(self.conv(x))
        x = dSiLU(self.fc_hidden(x.view(x.size(0), -1)))
        return f.softmax(self.policy(x), dim=1), self.value(x)


transition = namedtuple('transition', 'state, last_state, action, reward, is_terminal')


def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


def world_dynamics(s, env, network):
    with torch.no_grad():
        action = torch.multinomial(network(s)[0],1)[0]

    reward, terminated = env.act(action)

    s_prime = get_state(env.state())

    return s_prime, action, torch.tensor([[reward]], device=device).float(), torch.tensor([[terminated]], device=device)


def acb_train_step(
    sample,
    traces,
    grads,
    MSGs,
    network,
    alpha,
    time_step,
    *,
    LAMBDA: float = 0.8,
    GAMMA: float = 0.99,
    BETA: float = 0.01,
    GAMMA_RMS: float = 0.999,
    EPS_RMS: float = 0.0001,
    MIN_DENOM: float = 0.0001,
):
    """Byte-equivalent port of the reference ``train()`` function.

    Only difference vs the reference : constants are accepted as kwargs so a
    parity test can drive identical values into both reference and port.
    """
    last_state = sample.last_state
    state = sample.state
    action = sample.action
    reward = sample.reward
    is_terminal = sample.is_terminal

    pi, V_curr = network(state)

    trace_potential = V_curr+0.5*torch.log(pi[0,action]+MIN_DENOM)
    entropy = -torch.sum(torch.log(pi+MIN_DENOM)*pi)

    network.zero_grad()
    trace_potential.backward(retain_graph=True)

    with torch.no_grad():
        for param, grad in zip(network.parameters(), grads):
            grad.data.copy_(param.grad)

    if(last_state is not None):
        network.zero_grad()
        entropy.backward()
        with torch.no_grad():
            V_last = network(last_state)[1]
            delta = GAMMA*(0 if is_terminal else V_curr)+reward-V_last

            for param, trace, MSG in zip(network.parameters(), traces, MSGs):
                param_grad = 0 if param.grad is None else param.grad
                grad = trace*delta[0]+BETA*param_grad
                MSG.copy_(GAMMA_RMS*MSG+(1-GAMMA_RMS)*grad*grad)
                param.copy_(param+alpha*grad/(torch.sqrt(MSG/(1-GAMMA_RMS**(time_step+1))+EPS_RMS)))

    with torch.no_grad():
        for grad, trace in zip(grads, traces):
            trace.copy_(LAMBDA*GAMMA*trace+grad)
