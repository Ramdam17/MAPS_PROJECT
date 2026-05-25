"""Paper-faithful reference for SARL parity tests.

Origin
------
Originally extracted verbatim from ``external/MinAtar/examples/maps.py``
(commit ``ec5bcb7``; Vargas et al., MAPS TMLR submission 2025) for Sprint-04b
parity tests. **Sprint-08 D.15 (2026-04-20)** updated this module to track
**paper Table 11** instead of the student extract — policy locked 2026-04-19:
"paper = source of truth".

What changed from the student extract (and why)
-----------------------------------------------
- ``GAMMA = 0.999`` (was 0.99). Paper Table 11 row 14. D-sarl-gamma.
- ``step_size1 = 0.0003`` (unchanged; student = paper on this row). D8 retraction.
- ``step_size2 = 0.0002`` (was 0.00005). Paper Table 11 row 10. D-sarl-lr-2nd.
- ``QNetwork.forward`` reconstruction uses a learnable ``self.b_recon`` bias
  (paper eq.12: ``Ŷ^(1) = ReLU(W^T·Hidden + b_recon)``). Student omitted it;
  we add it zero-initialised so numerical parity at init is preserved while
  training can learn the bias. D-sarl-recon-bias.

Unchanged from the student extract (identifiers, comment layout, capitalisation,
commented-out lines are preserved exactly for readable diffs against the paper
source). The ``ruff: noqa`` block stays to tolerate the paper naming style.

Stripped vs original source
---------------------------
- No module-level ``NvidiaEnergyTracker`` instantiation (import side effect).
- No module-level ``print("using ", device)``.
- ``device`` resolved to CPU (parity tests run on CPU; forward-pass math is
  device-independent at atol=1e-6 for MinAtar sizes).
"""
# ruff: noqa: N801, N802, N803, N806, E741, B007, SIM108, UP008, UP032, RUF001, E712, E711

# flake8: noqa

from collections import namedtuple

import random

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init
from torch.autograd import Variable  # deprecated but present in paper source

# Parity tests pin CPU for reproducibility. The paper uses CUDA when available,
# but forward-pass math is device-independent at atol=1e-6 for the sizes here.
device = torch.device("cpu")

# Paper Table 11 constants (D.15 alignment). Module-level globals are
# preserved for faithful transcription of `train()`.
GAMMA = 0.999  # paper Table 11 row 14 (student used 0.99 — D-sarl-gamma)
MIN_SQUARED_GRAD = 0.01
step_size1 = 0.0003  # paper Table 11 row 9 (student=paper on this row)
step_size2 = 0.0002  # paper Table 11 row 10 (student used 0.00005 — D-sarl-lr-2nd)
scheduler_step = 0.999


# ─── QNetwork support ────────────────────────────────────────────────────────
# Source: external/MinAtar/examples/maps.py:129-133

def size_linear_unit(size, kernel_size=3, stride=1):
    return (size - (kernel_size - 1) - 1) // stride + 1

num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16


# ─── QNetwork ────────────────────────────────────────────────────────────────
# Source: external/MinAtar/examples/maps.py:134-182

#default version
class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):

        super(QNetwork, self).__init__()

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units

        self.sigmoid = nn.Sigmoid()

        #autoencoder
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        # Output layer:
        self.actions = nn.Linear(in_features=128, out_features=num_actions)

        #self.fc_comparison = nn.Linear(in_features=128, out_features=num_linear_units)

        # Paper eq.12: reconstruction decoder is ReLU(fc_hidden.weight.T · Hidden + b_recon).
        # Student extract omitted b_recon; D.15 adds it paper-faithful, zero-init so
        # forward parity at init is preserved. torch.zeros does not consume the RNG →
        # conv/fc_hidden/actions init draws stay aligned with the pre-D.15 stream.
        self.b_recon = nn.Parameter(torch.zeros(num_linear_units))


    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x , prev_h2, cascade_rate): #torch.Size([32, 4, 10, 10])

        x = f.relu(self.conv(x)) #torch.Size([32, 16, 8, 8])
        Input= x.view(x.size(0),-1)  # torch.Size([32, 1024])
        Hidden = f.relu(self.fc_hidden(Input))   # torch.Size([32, 128])

        if prev_h2 is not None:
            Hidden= cascade_rate*Hidden +  (1-cascade_rate)*prev_h2

        # Returns the output from the fully-connected linear layer
        x=self.actions(Hidden) # torch.Size([32, 6])

        # Paper eq.12 reconstruction with learnable bias (D.15).
        Output_comparison = f.relu(f.linear(Hidden, self.fc_hidden.weight.t(), self.b_recon))

        #Output_comparison = f.relu(self.fc_comparison(Hidden))   # torch.Size([32, 1024])

        Comparisson= Input - Output_comparison

        return x , Hidden , Comparisson, Hidden


# ─── SecondOrderNetwork ──────────────────────────────────────────────────────
# Source: external/MinAtar/examples/maps.py:245-278

class SecondOrderNetwork(nn.Module):
    def __init__(self, in_channels):
        super(SecondOrderNetwork, self).__init__()

        # Define a linear layer for comparing the difference between input and output of the first-order network
        #self.comparison_layer = nn.Linear(in_features=num_linear_units, out_features=128)

        # Linear layer for determining wagers
        self.wager = nn.Linear(num_linear_units, 2)
        self.dropout = nn.Dropout(p=0.1)  # 10% dropout

        self.sigmoid = torch.sigmoid

        # Initialize the weights of the network
        self._init_weights()

    def _init_weights(self):
        # Kaiming initialization for stability
        #init.uniform_(self.comparison_layer.weight, -1.0, 1.0)
        init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(self, comparison_matrix, prev_comparison, cascade_rate):

        # Pass the input through the comparison layer and apply dropout and activation
        #comparison_out = self.dropout(f.relu(self.comparison_layer(comparison_matrix)))
        comparison_out= self.dropout(comparison_matrix)

        if prev_comparison is not None:
          comparison_out = cascade_rate * comparison_out + (1 - cascade_rate) * prev_comparison

        # Pass through wager layer
        wager = self.wager(comparison_out)

        return wager ,  comparison_out


# ─── replay_buffer ───────────────────────────────────────────────────────────
# Source: external/MinAtar/examples/maps.py:289-307

transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


# ─── get_state ───────────────────────────────────────────────────────────────
# Source: external/MinAtar/examples/maps.py:322-323

def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


# ─── target_wager ────────────────────────────────────────────────────────────
# Source: external/MinAtar/examples/maps.py:514-532

def target_wager(rewards, alpha):
    flattened_rewards = rewards.view(-1)  # Flatten rewards to 1D for easy indexing
    alpha = float(alpha/100)  # EMA hyperparameter
    EMA = 0.0

    batch_size = rewards.size(0)  # Get the batch size (first dimension of rewards)
    new_tensor = torch.zeros(batch_size, 2, device=rewards.device)  # Create [batch_size, 2] tensor on the same device

    for i in range(batch_size):
        G = flattened_rewards[i]  # Current reward
        EMA = alpha * G + (1 - alpha) * EMA  # Update EMA

        # Set values based on comparison with EMA
        if G > EMA:
            new_tensor[i] = torch.tensor([1, 0], device=rewards.device)
        else:
            new_tensor[i] = torch.tensor([0, 1], device=rewards.device)

    return new_tensor


# ─── CAE_loss ────────────────────────────────────────────────────────────────
# Source: external/MinAtar/examples/maps.py:330-379

def CAE_loss(W, x, recons_x, h, lam):
    """Compute the Contractive AutoEncoder Loss

    Evalutes the CAE loss, which is composed as the summation of a Mean
    Squared Error and the weighted l2-norm of the Jacobian of the hidden
    units with respect to the inputs.

    Contrastive loss plays a crucial role in maintaining the similarity
    and correlation of latent representations across different modalities.
    This is because it helps to ensure that similar instances are represented
    by similar vectors and dissimilar instances are represented by dissimilar vectors.


    See reference below for an in-depth discussion:
      #1: http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder

    Args:
        `W` (FloatTensor): (N_hidden x N), where N_hidden and N are the
          dimensions of the hidden units and input respectively.
        `x` (Variable): the input to the network, with dims (N_batch x N)
        recons_x (Variable): the reconstruction of the input, with dims
          N_batch x N.
        `h` (Variable): the hidden units of the network, with dims
          batch_size x N_hidden
        `lam` (float): the weight given to the jacobian regulariser term

    Returns:
        Variable: the (scalar) CAE loss
    """

    #input, target

    #mse = mse_loss(recons_x, x)

    #mse=f.smooth_l1_loss( recons_x , x)        # loss 1 is converging, eval reward not go up so fast
    #mse=f.mse_loss( recons_x , x)               #loss 1 seems to explode at first but then go down fast, eval rewards go up fast
    #mse=f.l1_loss( recons_x , x)                 # loss ok, but reward not so high
    mse=f.huber_loss( recons_x , x)
    #mse=f.cross_entropy(recons_x, x)


    # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
    # opposed to #1
    dh = h * (1 - h) # Hadamard product produces size N_batch x N_hidden
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(Variable(W)**2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)


# ─── reference_dqn_update_step ───────────────────────────────────────────────
# Source: external/MinAtar/examples/maps.py:663-887 (train function)
#
# De-globalized adaptation for parity testing:
# - All globals (policy_net, target_net, second_order_net, optimizer,
#   optimizer2, scheduler1, scheduler2) are passed as explicit arguments.
# - The continuous-learning branch (`previous_loss != None`) is EXCISED — parity
#   for CL lives in `sarl_cl` (Sprint 04b §4.6), and would require teacher nets
#   and a loss_weighter. The standard DQN path is unchanged.
# - `train_or_test` hardcoded to True (we always update during parity tests).
# - `counter_list_losses` increment and anomaly detection removed (no side
#   effect needed for single-step parity check).
# - RETURN SIGNATURE: always (loss, loss_second_or_None, accuracy_second_or_None)
#   so callers get a uniform shape regardless of `meta`.

def reference_dqn_update_step(
    sample,
    policy_net,
    target_net,
    second_order_net,
    optimizer,
    optimizer2,
    scheduler1,
    scheduler2,
    meta,
    alpha,
    cascade_iterations_1,
    cascade_iterations_2,
):
    """Verbatim-structure DQN+MAPS update step, globals lifted to kwargs.

    Preserves EXACT statement order from paper's ``train()``. In particular,
    the meta branch executes ``loss_second.backward(retain_graph=True)`` →
    ``optimizer2.step()`` BEFORE ``loss.backward()`` → ``optimizer.step()``.
    Policy_net receives gradient contributions from BOTH losses.
    """

    # Calculate cascade rates for iterative information flow
    cascade_rate_1 = float(1.0/cascade_iterations_1)
    cascade_rate_2 = float(1.0/cascade_iterations_2)

    # Initialize outputs for cascade model connections
    comparison_out = None
    main_task_out = None
    target_task_out = None

    # Reset gradients
    optimizer.zero_grad()
    if meta:
        optimizer2.zero_grad()

    # Unpack transitions from replay buffer
    batch_samples = transition(*zip(*sample))

    # Extract batch elements
    states = torch.cat(batch_samples.state)
    next_states = torch.cat(batch_samples.next_state)
    actions = torch.cat(batch_samples.action)
    rewards = torch.cat(batch_samples.reward)

    # Calculate wagering targets based on rewards and alpha
    targets_wagering = target_wager(rewards, alpha)

    is_terminal = torch.cat(batch_samples.is_terminal)

    # Process through cascade model for main DQN (iterative information flow)
    for j in range(cascade_iterations_1):
        output_DQN_policy, h1, comparison_1, main_task_out = policy_net(states, main_task_out, cascade_rate_1)

    # Gather Q-values for the actions taken in each state
    Q_s_a = output_DQN_policy.gather(1, actions)

    # Handle target calculation for Q-learning (max Q-value in next state)
    none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0], dtype=torch.int64, device=device)
    none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)

    # Initialize target Q-values
    Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=device)
    if len(none_terminal_next_states) != 0:
        for k in range(cascade_iterations_1):
            output_DQN_target, _, _, target_task_out = target_net(none_terminal_next_states, target_task_out, cascade_rate_1)
        Q_s_prime_a_prime[none_terminal_next_state_index] = output_DQN_target.detach().max(1)[0].unsqueeze(1)

    # Compute the TD target (reward + discounted future reward)
    target = rewards + GAMMA * Q_s_prime_a_prime

    # Setup for contractive autoencoder loss
    W = policy_net.state_dict()['fc_hidden.weight']
    lam = 1e-4

    # Standard DQN loss when not using continuous learning
    loss = CAE_loss(W, target, Q_s_a, h1, lam)

    # Metacognitive (2nd order) network branch
    if meta:
        with torch.set_grad_enabled(True):
            for i in range(cascade_iterations_2):
                output_second, comparison_out = second_order_net(comparison_1, comparison_out, cascade_rate_2)

        loss_second = f.binary_cross_entropy_with_logits(output_second, targets_wagering)

        # wagering accuracy (stubbed — paper calls calculate_wagering_accuracy,
        # which is a readout metric and does not affect gradients). Parity
        # ignores this output.
        accuracy_second = None

        # Backward pass for metacognitive network first
        loss_second.backward(retain_graph=True)
        optimizer2.step()

        # Then backward pass for main network
        loss.backward()
        optimizer.step()

        # Step the learning rate schedulers
        scheduler1.step()
        scheduler2.step()

        return loss, loss_second, accuracy_second

    else:
        loss.backward()
        optimizer.step()
        scheduler1.step()
        return loss, None, None
