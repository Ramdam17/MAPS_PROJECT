################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian(ttian@ualberta.ca)                                                                                 #
#                                                                                                              #
# python3 dqn.py -g <game>                                                                                     #
#   -o, --output <directory/file name prefix>                                                                  #
#   -v, --verbose: outputs the average returns every 1000 episodes                                             #
#   -l, --loadfile <directory/file name of the saved model>                                                    #
#   -a, --alpha <number>: step-size parameter                                                                  #
#   -s, --save: save model data every 1000 episodes                                                            #
#   -r, --replayoff: disable the replay buffer and train on each state transition                              #
#   -t, --targetoff: disable the target network                                                                #
#                                                                                                              #
# References used for this implementation:                                                                     #
#   https://pytorch.org/docs/stable/nn.html#                                                                   #
#   https://pytorch.org/docs/stable/torch.html                                                                 #
#   https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html                                   #
################################################################################################################



################     COMMAND EXAMPLE: python examples/dqn_2nd_order.py -ema 25 -cascade 50 -g breakout -seed 1 -setting 2 



import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time
import math
from collections import OrderedDict


import random, numpy, argparse, logging, os

from collections import namedtuple
from minatar import Environment

#########NEW
import torch.nn.init as init
from torch.autograd import Variable
import random
import copy
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import pickle
import io
import sys
from pprint import pprint

from scipy.stats import norm

import torch.serialization
#only for digital alliance canada
'''
torch.serialization.add_safe_globals([
    'numpy._core.multiarray.scalar',
    'numpy.core.multiarray.scalar',
    'numpy.ndarray',
    'collections.OrderedDict'
])
'''

################################################################################################################
# Constants
#
################################################################################################################
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 100000
TRAINING_FREQ = 1
FIRST_N_FRAMES = 100000
REPLAY_START_SIZE = 5000
END_EPSILON = 0.1
step_size1 = 0.0003
step_size2 = 0.00005

GRAD_MOMENTUM = 0.95
SQUARED_GRAD_MOMENTUM = 0.95
MIN_SQUARED_GRAD = 0.01
GAMMA = 0.99
EPSILON = 1.0
LAMBDA=0.8
scheduler_step=0.999
MAX_INPUT_CHANNELS=10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using ", device)

from distutils.util import strtobool


def str2bool(v):
    return bool(strtobool(v))


################################################################################################################
# class QNetwork
#
# One hidden 2D conv with variable number of input channels.  We use 16 filters, a quarter of the original DQN
# paper of 64.  One hidden fully connected linear layer with a quarter of the original DQN paper of 512
# rectified units.  Finally, the output layer is a fully connected linear layer with a single output for each
# valid action.
#
################################################################################################################
def size_linear_unit(size, kernel_size=3, stride=1):
    return (size - (kernel_size - 1) - 1) // stride + 1

num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16

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
        self.fc_output = nn.Linear(in_features=128, out_features=num_linear_units)

        # Output layer:
        self.actions = nn.Linear(in_features=num_linear_units, out_features=num_actions)

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x , prev_h2, cascade_rate): #torch.Size([32, 4, 10, 10])
        
        x = f.relu(self.conv(x)) #torch.Size([32, 16, 8, 8])
        Input= x.view(x.size(0),-1)  # torch.Size([32, 1024])
        Hidden = f.relu(self.fc_hidden(Input))   # torch.Size([32, 128])
        Output = f.relu(self.fc_output(Hidden))   # torch.Size([32, 1024])
        
        if prev_h2 is not None:
            Output= cascade_rate*Output +  (1-cascade_rate)*prev_h2
            
        # Returns the output from the fully-connected linear layer
        x=self.actions(Output) # torch.Size([32, 6])
        Comparisson= Input - Output
        
        return x , Hidden , Comparisson, Output

class AdaptiveQNetwork(nn.Module):
    def __init__(self, max_input_channels, num_actions):
        super(AdaptiveQNetwork, self).__init__()
        
        # Using max_input_channels for initialization
        self.max_input_channels = max_input_channels
        
        # Input adaptation layer
        self.input_adapter = nn.Sequential(
            nn.Conv2d(max_input_channels, max_input_channels, kernel_size=1, stride=1),
            nn.ReLU()
        )
        
        # Main convolution layer
        self.conv = nn.Conv2d(max_input_channels, 16, kernel_size=3, stride=1)
        
        # Calculate the output size of conv layer for a 10x10 input
        conv_output_size = self._get_conv_output_size((max_input_channels, 10, 10))
        
        # Autoencoder and output layers
        self.fc_hidden = nn.Linear(in_features=conv_output_size, out_features=128)
        self.fc_output = nn.Linear(in_features=128, out_features=conv_output_size)
        self.actions = nn.Linear(in_features=conv_output_size, out_features=num_actions)
        
    def _get_conv_output_size(self, shape):
        # Helper function to calculate conv output size
        bs = 1
        input = torch.rand(bs, *shape)
        output = self.conv(input)
        return int(numpy.prod(output.size()[1:]))

    def adapt_input(self, x):
        # Handle different input channel counts
        if x.size(1) < self.max_input_channels:
            # Pad with zeros to match max_input_channels
            padding = torch.zeros(x.size(0), self.max_input_channels - x.size(1), 
                                x.size(2), x.size(3), device=x.device)
            x = torch.cat([x, padding], dim=1)
        return x

    def forward(self, x, prev_h2, cascade_rate):
        # Adapt input to match expected channel count
        x = self.adapt_input(x)
        
        # Apply input adaptation
        x = self.input_adapter(x)
        
        # Rest of the forward pass
        x = f.relu(self.conv(x))
        Input = x.view(x.size(0), -1)
        Hidden = f.relu(self.fc_hidden(Input))
        Output = f.relu(self.fc_output(Hidden))
        
        if prev_h2 is not None:
            Output = cascade_rate * Output + (1 - cascade_rate) * prev_h2
            
        x = self.actions(Output)
        Comparison = Input - Output
        
        return x, Hidden, Comparison, Output

class SecondOrderNetwork(nn.Module):
    def __init__(self, in_channels):
        super(SecondOrderNetwork, self).__init__()
        
        # Define a linear layer for comparing the difference between input and output of the first-order network
        self.comparison_layer = nn.Linear(in_features=num_linear_units, out_features=num_linear_units)
        
        # Linear layer for determining wagers
        self.wager = nn.Linear(num_linear_units, 2)
        self.dropout = nn.Dropout(p=0.1)  # 10% dropout
        self.softmax = nn.Softmax()  # Specify dimension for Softmax
        self.sigmoid = nn.Sigmoid()

        # Initialize the weights of the network
        self._init_weights()

    def _init_weights(self):
        # Kaiming initialization for stability
        init.uniform_(self.comparison_layer.weight, -1.0, 1.0)
        init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(self, comparison_matrix, prev_comparison, cascade_rate):
        
        # Pass the input through the comparison layer and apply dropout and activation
        comparison_out = self.dropout(f.relu(self.comparison_layer(comparison_matrix))) 
        
        if prev_comparison is not None:
          comparison_out = cascade_rate * comparison_out + (1 - cascade_rate) * prev_comparison

        # Pass through wager layer 
        wager = self.wager(comparison_out)
        
        return wager ,  comparison_out


###########################################################################################################
# class replay_buffer
#
# A cyclic buffer of a fixed size containing the last N number of recent transitions.  A transition is a
# tuple of state, next_state, action, reward, is_terminal.  The boolean is_terminal is used to indicate
# whether if the next state is a terminal state or not.
#
###########################################################################################################
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


################################################################################################################
# get_state
#
# Converts the state given by the environment to a tensor of size (in_channel, 10, 10), and then
# unsqueeze to expand along the 0th dimension so the function returns a tensor of size (1, in_channel, 10, 10).
#
# Input:
#   s: current state as numpy array
#
# Output: current state as tensor, permuted to match expected dimensions
#
################################################################################################################
def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()




#contractive loss

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


    # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
    # opposed to #1
    dh = h * (1 - h) # Hadamard product produces size N_batch x N_hidden
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(Variable(W)**2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)


#Loss not working for continuous learning
'''
class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0):
        """
        Initialize the distillation loss function.
        
        Args:
            temperature (float): Temperature parameter for softening probability distributions.
                               Higher values produce softer distributions.
        """
        super(DistillationLoss, self).__init__()
        self.T = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, student_outputs, teacher_outputs, hard_labels=None, alpha=0.5):
        """
        Compute the distillation loss between student and teacher outputs.
        
        Args:
            student_outputs (torch.Tensor): Output logits from the student model
            teacher_outputs (torch.Tensor): Output logits from the teacher model
            hard_labels (torch.Tensor, optional): Ground truth labels (for classification)
            alpha (float, optional): Weight for the distillation loss component
            
        Returns:
            torch.Tensor: Combined distillation loss
        """
        
        #loss= f.mse_loss(student_outputs, teacher_outputs)
        
        # Compute soft targets for teacher with temperature scaling
        soft_targets = f.softmax(teacher_outputs / self.T, dim=-1)
        
        # Compute log probabilities for student with temperature scaling
        log_probs = f.log_softmax(student_outputs / self.T, dim=-1)
        
        # Calculate KL divergence loss (scaled by T²)
        soft_loss = torch.sum(soft_targets * (soft_targets.log() - log_probs), dim=-1).mean()

        # If hard labels are provided, compute the standard cross-entropy loss
        hard_loss = 0
        if hard_labels is not None:
            hard_loss = self.criterion(student_outputs, hard_labels)


        # Combine both soft and hard losses
        loss = alpha * soft_loss + (1 - alpha) * hard_loss
        return loss
    
distillation_loss = DistillationLoss(temperature=2.0)
'''

def compute_weight_regularization(model, teacher_model):
    reg_loss = 0
    # Loop through parameters and compare
    for param, param_old in zip(model.parameters(), teacher_model.parameters()):
        reg_loss += torch.sum((param - param_old) ** 2)
    
    return reg_loss  


################################################################################################################
# world_dynamics
#
# It generates the next state and reward after taking an action according to the behavior policy.  The behavior
# policy is epsilon greedy: epsilon probability of selecting a random action and 1 - epsilon probability of
# selecting the action with max Q-value.
#
# Inputs:
#   t : frame
#   replay_start_size: number of frames before learning starts
#   num_actions: number of actions
#   s: current state
#   env: environment of the game
#   policy_net: policy network, an instance of QNetwork
#
# Output: next state, action, reward, is_terminated
#
################################################################################################################
def world_dynamics(t, replay_start_size, num_actions, s, env, cascade_iterations_1, cascade_iterations_2):

    cascade_rate_1= float(1.0/cascade_iterations_1)
    main_task_out=None
    
    # A uniform random policy is run before the learning starts
    if t < replay_start_size:
        action = torch.tensor([[random.randrange(num_actions)]], device=device)
    else:
        # Epsilon-greedy behavior policy for action selection
        # Epsilon is annealed linearly from 1.0 to END_EPSILON over the FIRST_N_FRAMES and stays 0.1 for the
        # remaining frames
        epsilon = END_EPSILON if t - replay_start_size >= FIRST_N_FRAMES \
            else ((END_EPSILON - EPSILON) / FIRST_N_FRAMES) * (t - replay_start_size) + EPSILON

        if numpy.random.binomial(1, epsilon) == 1:
            action = torch.tensor([[random.randrange(num_actions)]], device=device)
        else:
            # State is 10x10xchannel, max(1)[1] gives the max action value (i.e., max_{a} Q(s, a)).
            # view(1,1) shapes the tensor to be the right form (e.g. tensor([[0]])) without copying the
            # underlying tensor.  torch._no_grad() avoids tracking history in autograd.
            with torch.no_grad():
                for j in range(cascade_iterations_1):
                    output_network_policy , hn, comparison_n, main_task_out= policy_net(s, main_task_out, cascade_rate_1)
                    
                action = output_network_policy.max(1)[1].view(1, 1)

    # Act according to the action and observe the transition and reward
    reward, terminated = env.act(action)

    # Obtain s_prime
    s_prime = get_state(env.state())

    return s_prime, action, torch.tensor([[reward]], device=device).float(), torch.tensor([[terminated]], device=device)


################################################################################################################
# train
#
# This is where learning happens. More specifically, this function learns the weights of the policy network
# using huber loss.
#
# Inputs:
#   sample: a batch of size 1 or 32 transitions
#   policy_net: an instance of QNetwork
#   target_net: an instance of QNetwork
#   optimizer: centered RMSProp
#
################################################################################################################



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
   
            
def min_max_norm(mean, values):
    values = numpy.array(values)
    #min_val = numpy.min(values)
    max_val = numpy.max(values)
    return mean/max_val , max_val


def individual_losses(output, target, loss_fn):
    # Compute the loss for each sample in the batch individually
    batch_size = output.size(0)
    losses = []
    
    for i in range(batch_size):
        loss = loss_fn(output[i:i+1], target[i:i+1])
        losses.append(loss.item())  # Store the individual loss value for each sample
    
    losses_numpy=numpy.array(losses)
    # Return the maximum loss wrapped in a list
    return numpy.percentile(losses_numpy, 95) 
     

def update_moving_average(current_avg, new_value, momentum=0.9):
    # Detach the tensors before computing moving average
    return momentum * current_avg + (1 - momentum) * new_value

class DynamicLossWeighter:
    def __init__(self):
        self.moving_avgs = {
            'task': 1.0,
            'distillation': 1.0, 
            'feature': 1.0
        }
        
        self.historical_max = {
            'task': float('-inf'),
            'distillation': float('-inf'),
            'feature': float('-inf')
        }
        
        self.historical_max_prev = {
            'task': float('-inf'),
            'distillation': float('-inf'),
            'feature': float('-inf')
        }
        
        self.steps = 0
        self.update_interval = 10000  # Update historical max every 100 steps
        self.scale_factors = {k: 1.0 for k in self.moving_avgs}
    
    def update(self, losses):
        self.steps = self.steps + 1

        # Detach losses before updating
        detached_losses = {k: v.detach() if isinstance(v, torch.Tensor) else v 
                         for k, v in losses.items()}
        
        # Update moving averages
        for key, value in detached_losses.items():
            value_float = float(value.item() if isinstance(value, torch.Tensor) else value)
            self.moving_avgs[key] = value_float
            
            if self.steps % self.update_interval == (self.update_interval // 2):
                self.historical_max_prev[key] = value_float
                
            #if self.steps % self.update_interval == 0:
             #   self.historical_max[key] = max( self.historical_max_prev[key],value_float)
            #else:
            self.historical_max[key] = max(self.historical_max[key], value_float)
         
        '''
        # Calculate normalization factors using historical maximums
        for key in self.moving_avgs:
            if self.historical_max[key] > 0:
                # Normalize current loss by its historical maximum
                normalized_value = self.moving_avgs[key] / self.historical_max[key]
                self.scale_factors[key] = 1.0 / normalized_value if normalized_value > 0 else 1.0
        '''
    
    def weight_losses(self, losses):
        epsilon=1e-16
        weighted_losses = {}
        for key, value in losses.items():
            
            #weighted_losses[key] = value * self.scale_factors[key]
            weighted_losses[key] = value / (self.historical_max[key] + epsilon)
            #weighted_losses[key] = value
            

            #weighted_losses[key] = (update_moving_average(self.moving_avgs[key], value)) /  self.historical_max[key] 
        return weighted_losses

    def get_stats(self):
        """Returns current statistics for monitoring"""
        return {
            'moving_averages': self.moving_avgs.copy(),
            'historical_max': self.historical_max.copy(),
            'scale_factors': self.scale_factors.copy()
        }


def train(sample, target_net, meta, alpha, cascade_iterations_1, cascade_iterations_2, train_or_test):
    """
    Trains the DQN model with optional metacognitive components (MAPS architecture)
    
    Args:
        sample: Batch of experiences from replay buffer
        target_net: Target network for Q-learning
        meta: Boolean indicating whether to use 2nd order metacognitive network
        alpha: Smoothing factor for exponential moving average used in wagering
        cascade_iterations_1: Number of iterations for cascade model in main DQN
        cascade_iterations_2: Number of iterations for cascade model in metacognitive network
        train_or_test: Boolean indicating whether to update weights (True) or just evaluate (False)
        
    Returns:
        Various loss values depending on configuration (main task loss, metacognitive loss, component losses)
    """
    
    global counter_list_losses, list_losses_first, list_losses_first_distillation, list_losses_first_features, list_losses_second, list_losses_second_distillation, list_losses_second_features
    
    # Global loss weighters for balancing different loss components in continuous learning
    global loss_weighter, loss_weighter_second, scheduler1, scheduler2

    with torch.autograd.set_detect_anomaly(True):
        
        counter_list_losses+=1
                
        # Calculate cascade rates for iterative information flow
        cascade_rate_1 = float(1.0/cascade_iterations_1)
        cascade_rate_2 = float(1.0/cascade_iterations_2)

        # Initialize outputs for cascade model connections
        comparison_out = None
        main_task_out = None
        target_task_out = None

        # Initialize teacher network outputs for continuous learning
        comparison_out_teacher = None
        main_task_out_teacher = None
        
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
                
        # For continuous learning, run teacher network (previous task model)
        if previous_loss != None:
            
            with torch.no_grad():
                # Forward pass through teacher network (no gradient updates)
                for j in range(cascade_iterations_1):
                    _, h1_teacher, comparison_1_teacher, main_task_out_teacher = teacher_first_net(states, main_task_out_teacher, cascade_rate_1) 
                

        # Gather Q-values for the actions taken in each state
        Q_s_a = output_DQN_policy.gather(1, actions)

        # Handle target calculation for Q-learning (max Q-value in next state)
        # Find non-terminal states to calculate future rewards
        none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0], dtype=torch.int64, device=device)
        none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)

        # Initialize target Q-values
        Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=device)
        if len(none_terminal_next_states) != 0:
            # Compute target network output with cascade model
            for k in range(cascade_iterations_1):
                output_DQN_target, _, _, target_task_out = target_net(none_terminal_next_states, target_task_out, cascade_rate_1)
                
            # Get max Q-value for next state (detached from computation graph)
            Q_s_prime_a_prime[none_terminal_next_state_index] = output_DQN_target.detach().max(1)[0].unsqueeze(1)

        # Compute the TD target (reward + discounted future reward)
        target = rewards + GAMMA * Q_s_prime_a_prime

        # Setup for contractive autoencoder loss
        W = policy_net.state_dict()['fc_hidden.weight']
        lam = 1e-4  # Regularization parameter for contractive loss
        
        # Weights for different loss components in continuous learning
        ce_loss_weight = WEIGHT1      # Current task loss weight
        soft_target_loss_weight = WEIGHT2  # Distillation/regularization weight
        feature_weight = WEIGHT3      # Feature preservation weight
        
        # Continuous learning branch (when teacher network exists)
        if previous_loss != None and (not(evaluation_continuous_learning)):
            
            # Weight regularization to prevent catastrophic forgetting (EWC-inspired)
            loss_previous_task = compute_weight_regularization(policy_net, teacher_first_net)
                                          
            # Feature preservation loss between current and teacher networks
            feature_loss = f.mse_loss(h1, h1_teacher) 
                        
            # Current task loss (contractive autoencoder loss)
            loss_task = CAE_loss(W, target, Q_s_a, h1, lam) 
            
            # Organize losses for the loss weighter
            current_losses = {
                'task': loss_task,
                'distillation': loss_previous_task,
                'feature': feature_loss
            }
            
            # Dynamic loss weighting based on observed magnitudes
            loss_weighter.update(current_losses)
            weighted_losses = loss_weighter.weight_losses(current_losses)
            
            # Apply the weighted losses
            loss_task = weighted_losses['task']
            loss_previous_task = weighted_losses['distillation']
            feature_loss = weighted_losses['feature']

            # Combine losses with predefined weights
            loss = (soft_target_loss_weight * loss_previous_task + 
                    ce_loss_weight * loss_task + 
                    feature_weight * feature_loss)
        
        else: 
            # Standard DQN loss when not using continuous learning
            loss = CAE_loss(W, target, Q_s_a, h1, lam)        
            
        # Metacognitive (2nd order) network branch
        if meta:
            
            with torch.set_grad_enabled(True):
                # Process through cascade model for metacognitive network
                for i in range(cascade_iterations_2):
                    output_second, comparison_out = second_order_net(comparison_1, comparison_out, cascade_rate_2)

            # Continuous learning for metacognitive network
            if previous_loss != None and (not(evaluation_continuous_learning)):
                
                with torch.no_grad():
                    # Forward pass through teacher's metacognitive network
                    for i in range(cascade_iterations_2):
                        _, comparison_out_teacher = \
                            teacher_second_net(comparison_1_teacher, comparison_out_teacher, cascade_rate_2)
                
                # Weight regularization for metacognitive network
                previous_task_loss_second = compute_weight_regularization(second_order_net, teacher_second_net)
                
                # Current task loss for metacognitive network (wagering)
                task_loss_second = f.binary_cross_entropy_with_logits(output_second, targets_wagering)
                                
                # Feature preservation for metacognitive network
                feature_loss_second = f.mse_loss(comparison_out, comparison_out_teacher)
                
                # Organize metacognitive losses for weighting
                current_losses_second = {
                    'task': task_loss_second,
                    'distillation': previous_task_loss_second,
                    'feature': feature_loss_second
                }
                
                # Dynamic loss weighting for metacognitive network
                loss_weighter_second.update(current_losses_second)
                weighted_losses_second = loss_weighter_second.weight_losses(current_losses_second)
                
                # Apply weighted losses for metacognitive network
                task_loss_second = weighted_losses_second['task']
                previous_task_loss_second = weighted_losses_second['distillation']
                feature_loss_second = weighted_losses_second['feature']
                
                # Combine metacognitive losses with predefined weights
                loss_second = (soft_target_loss_weight * previous_task_loss_second +
                            ce_loss_weight * task_loss_second +
                            feature_weight * feature_loss_second)
            
            else: 
                # Standard metacognitive loss (wagering) when not using continuous learning
                loss_second = f.binary_cross_entropy_with_logits(output_second, targets_wagering)
                
            # Update weights if in training mode
            if train_or_test == True: 
                # Backward pass for metacognitive network first
                loss_second.backward(retain_graph=True)
                optimizer2.step()
                
                # Then backward pass for main network
                loss.backward()
                optimizer.step()
                
                # Step the learning rate schedulers
                scheduler1.step()
                scheduler2.step()

            # Return appropriate losses based on configuration
            if previous_loss != None and (not(evaluation_continuous_learning)):
                return loss, loss_second, (loss_task, loss_previous_task, feature_loss), (task_loss_second, previous_task_loss_second, feature_loss_second)
            else: 
                return loss, loss_second
            
        else:
            # Non-metacognitive branch (standard DQN only)
            if train_or_test == True:            
                loss.backward()
                optimizer.step()
                scheduler1.step()
            
            # Return appropriate losses based on configuration
            if previous_loss != None and (not(evaluation_continuous_learning)):
                return loss, (loss_task, loss_previous_task, feature_loss) 
            return loss


def evaluation(env, meta, t, target_off, target_net, alpha, cascade_iterations_1, cascade_iterations_2, replay_start_size, num_actions, replay_off, r_buffer):
   """
   Evaluates the agent's performance on the environment for a single episode
   
   Args:
       env: MinAtar environment being used
       meta: Boolean indicating whether the metacognitive (2nd order) network is used
       t: Current time step counter
       target_off: Boolean indicating whether target network is disabled
       target_net: Target network for Q-learning stability
       alpha: Smoothing factor for exponential moving average used in wagering
       cascade_iterations_1: Number of iterations for cascade model in main DQN
       cascade_iterations_2: Number of iterations for cascade model in metacognitive network
       replay_start_size: Minimum buffer size before learning starts
       num_actions: Number of possible actions in the environment
       replay_off: Boolean indicating whether replay buffer is disabled
       r_buffer: Replay buffer for experience storage
       
   Returns:
       G: Total episodic return
       loss_one: Average main task (DQN) loss for the episode
       loss_two: Average metacognitive network loss (if meta=True)
       Additional component losses if continuous learning is enabled
   """
   
   # Set networks to evaluation mode
   policy_net.eval()
   if meta:
       second_order_net.eval()
   if not target_off:
       target_net.eval()
       
   # Set teacher networks to evaluation mode if using continuous learning
   if previous_loss!=None:
       teacher_first_net.eval()
       if meta:
           teacher_second_net.eval()
           
   # Initialize loss tracking variables
   loss_one = None  # Main DQN loss
   loss_two = None  # Metacognitive network loss
       
   # Initialize component losses for continuous learning
   loss_one_distillation = None  # Weight regularization loss for main network
   loss_one_feature = None       # Feature preservation loss for main network
   loss_one_task = None          # Task loss for main network
   loss_two_distillation = None  # Weight regularization loss for metacognitive network
   loss_two_feature = None       # Feature preservation loss for metacognitive network
   loss_two_task = None          # Task loss for metacognitive network
   
   
   policy_net_update_counter = 0
   
   # Create a copy of the replay buffer for evaluation
   r_buffer_copie = r_buffer
   
   
   # Initialize the return for the episode
   G = 0.0
   counter_episode = 0

   # Initialize the environment and get starting state
   env.reset()
   s = get_state(env.state())
   is_terminated = False
   
   # Run until episode terminates
   while(not is_terminated):
       counter_episode += 1
       
       # Get next state, action, reward, and termination status
       s_prime, action, reward, is_terminated = world_dynamics(t, replay_start_size, num_actions, s, env, cascade_iterations_1, cascade_iterations_2)

       # Handle experience collection with or without replay buffer
       sample = None
       if replay_off:
           # Use current transition directly if replay is off
           sample = [transition(s, s_prime, action, reward, is_terminated)]
       else:
           # Add to replay buffer and sample a batch
           r_buffer_copie.add(s, s_prime, action, reward, is_terminated)
           # Sample a batch when buffer has enough data
           sample = r_buffer_copie.sample(BATCH_SIZE)
           
       # Evaluate model on samples at regular intervals
       if t % TRAINING_FREQ == 0 and sample is not None:
           
           # Initialize losses if this is the first training step
           if loss_one == None:
               loss_one = 0
               loss_one_distillation = 0
               loss_one_feature = 0
               loss_one_task = 0
               
           if loss_two == None:
               loss_two = 0
               loss_two_distillation = 0
               loss_two_feature = 0
               loss_two_task = 0
               
           if not target_off:
               policy_net_update_counter += 1

           # Evaluate model with metacognitive network
           if meta: 
               if previous_loss != None and (not(evaluation_continuous_learning)):
                   # Get all component losses for continuous learning with metacognition
                   loss_first, loss_second, (loss_task, previous_task_loss, feature_loss), (task_loss_second, previous_task_loss_second, feature_loss_second) = train(sample, policy_net, meta, alpha, cascade_iterations_1, cascade_iterations_2, False)
                   
                   # Accumulate component losses for main network
                   loss_one_task += loss_task.item()
                   loss_one_feature += feature_loss.item()
                   loss_one_distillation += previous_task_loss.item()
                   
                   # Accumulate component losses for metacognitive network
                   loss_two_task += task_loss_second.item()
                   loss_two_feature += feature_loss_second.item()
                   loss_two_distillation += previous_task_loss_second.item()  
                   
               else: 
                   # Standard losses without continuous learning
                   loss_first, loss_second = train(sample, policy_net, meta, alpha, cascade_iterations_1, cascade_iterations_2, False)

               # Accumulate total losses
               loss_one += loss_first.item()
               loss_two += loss_second.item()

           # Evaluate model without metacognitive network
           else:
               if previous_loss != None and (not(evaluation_continuous_learning)):
                   # Get component losses for continuous learning without metacognition
                   loss_first, (loss_task, previous_task_loss, feature_loss) = train(sample, policy_net, meta, alpha, cascade_iterations_1, cascade_iterations_2, False)
                   
                   # Accumulate component losses
                   loss_one_task += loss_task.item()
                   loss_one_feature += feature_loss.item()
                   loss_one_distillation += previous_task_loss.item() 
               else:
                   # Standard loss without continuous learning
                   loss_first = train(sample, policy_net, meta, alpha, cascade_iterations_1, cascade_iterations_2, False)

               # Accumulate total loss
               loss_one += loss_first.item()

       # Update return and time step
       G += reward.item()
       t += 1
       
       # Update current state
       s = s_prime

   # Calculate average losses over the episode
   if loss_one != None:
       loss_one = loss_one / counter_episode
       loss_one_task = loss_one_task / counter_episode
       loss_one_feature = loss_one_feature / counter_episode
       loss_one_distillation = loss_one_distillation / counter_episode
       
   if meta:
       if loss_two != None:
           loss_two = loss_two / counter_episode
           loss_two_task = loss_two_task / counter_episode
           loss_two_feature = loss_two_feature / counter_episode
           loss_two_distillation = loss_two_distillation / counter_episode

   # Return appropriate metrics based on configuration
   if previous_loss != None and (not(evaluation_continuous_learning)):
       # Return detailed component losses for continuous learning
       return G, loss_one, loss_two, (loss_one_task, loss_one_distillation, loss_one_feature), (loss_two_task, loss_two_distillation, loss_two_feature)
   else:
       # Return basic losses for standard learning
       return G, loss_one, loss_two


################################################################################################################
# dqn
#
# DQN algorithm with the option to disable replay and/or target network, and the function saves the training data.
# This implementation supports the MAPS (Metacognitive Architecture for Improved Perceptual and Social Learning)
# architecture, which includes second-order network integration and a cascade model approach.
#
# Inputs:
#   env: environment of the game
#   replay_off: disable the replay buffer and train on each state transition
#   target_off: disable target network
#   output_file_name: directory and file name prefix to output data and network weights, file saved as 
#       <output_file_name>_data_and_weights
#   store_intermediate_result: a boolean, if set to true will store checkpoint data every 1000 episodes
#       to a file named <output_file_name>_checkpoint
#   load_path: file path for a checkpoint to load, and continue training from
#   step_size: step-size for RMSProp optimizer
#
#################################################################################################################



def dqn(env, replay_off, target_off, output_file_name, meta, alpha, cascade_iterations_1, cascade_iterations_2, game_name, curriculum=False, curriculum_evaluation=False, store_intermediate_result=True, load_path=None, checkpoints=False, adapnet=False):
    
    # Define global variables to maintain state between function calls and share across the module
    global TARGET_NETWORK_UPDATE_FREQ 
    global second_order_net , policy_net , optimizer , optimizer2 , previous_loss, previous_loss_two, checkpoint
    global evaluation_continuous_learning

    # Loss tracking lists for both first-order and second-order networks
    global list_losses_first, counter_list_losses, list_losses_first_distillation, list_losses_first_features
    global list_losses_second, list_losses_second_distillation, list_losses_second_features

    # Dynamic loss weightings and curriculum-related parameters
    global loss_weighter, loss_weighter_second, reduce_curriculum, scheduler1, scheduler2, STEP_SIZE, STEP_SIZE_SECOND
    
    # Initialize curriculum reduction value (reduces frames for curriculum learning)
    reduce_curriculum=0
    
    # Initialize dynamic loss weighters to balance loss components during training
    loss_weighter = DynamicLossWeighter()
    loss_weighter_second = DynamicLossWeighter()

    # Initialize loss tracking lists for first-order network
    list_losses_first=[]
    list_losses_first_distillation=[]
    list_losses_first_features=[]
    
    # Initialize loss tracking lists for second-order (metacognitive) network
    list_losses_second=[]
    list_losses_second_distillation=[]
    list_losses_second_features=[]
    
    counter_list_losses=0
    
    # Set flag for curriculum evaluation mode
    evaluation_continuous_learning= curriculum_evaluation
    
    # For curriculum learning, we need teacher networks to retain knowledge of previous environments
    if curriculum:
        global teacher_first_net, teacher_second_net
        
        # Set up game-specific optimizer and replay buffer names for continuous learning
        current_optimizer= 'optimizer_' + game_name
        current_optimizer2= 'optimizer2_' + game_name
        current_replay_buffer= 'replay_buffer_' + game_name
        
        # Determine which components to load based on whether meta (2nd order network) is used
        if meta:
            # Load list includes optimizer states for both networks across all environments
            load_list = ['optimizer_space_invaders',
                    'optimizer_breakout',
                    'optimizer_seaquest',
                    'optimizer_freeway',
                    'optimizer_asterix',
                    
                    'optimizer2_space_invaders',
                    'optimizer2_breakout',
                    'optimizer2_seaquest',
                    'optimizer2_freeway',
                    'optimizer2_asterix',
                    
                    'replay_buffer_space_invaders',
                    'replay_buffer_breakout',
                    'replay_buffer_seaquest',
                    'replay_buffer_freeway',
                    'replay_buffer_asterix']
        else:
            # Without meta, we only need first-order optimizers and replay buffers
            load_list = ['optimizer_space_invaders',
                    'optimizer_breakout',
                    'optimizer_seaquest',
                    'optimizer_freeway',
                    'optimizer_asterix',
                    
                    'replay_buffer_space_invaders',
                    'replay_buffer_breakout',
                    'replay_buffer_seaquest',
                    'replay_buffer_freeway',
                    'replay_buffer_asterix']
        
    # Initialize loss tracking variables
    previous_loss=None
    previous_loss_two=None
    
    print("runining env ", game_name)
    if curriculum:
        print("training following a continous learning approach and a curriculum")
    if curriculum_evaluation:
        print("evaluating continous learning models")

    # Environment-specific parameters (Freeway needs different settings)
    if game_name=='freeway':
        print("we are using the freeway setting")
        # Freeway requires more frequent target network updates and fewer iterations between checkpoints
        TARGET_NETWORK_UPDATE_FREQ = 500
        checkpoint_iteration = 15
        validation_iterations = 2 
    else:
        # Standard parameters for other environments
        TARGET_NETWORK_UPDATE_FREQ = 500
        checkpoint_iteration = 600
        validation_iterations = 2
    
    # Special parameters for curriculum evaluation mode
    if curriculum_evaluation:
       # Increase validation iterations for certain environments to ensure reliable evaluation
       if game_name=='freeway' or game_name=='space_invaders':
           validation_iterations = 10
       else:
           validation_iterations = 10
           
       number_validation_curriculum = 1
       addition = 1

    # Print configuration details    
    print("Number of cascade iterations(1st order) are: ", cascade_iterations_1)
    print("Number of cascade iterations(1st order) are: ", cascade_iterations_2)
    print("Meta values is: ", meta)
    print("Alpha value for EMA is: ", float(alpha/100))
    
    # Get environment-specific parameters
    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()
    print("num actions ", num_actions, " in channels ", in_channels)

    # Initialize network architectures based on configuration
    if adapnet:
        print("using adaptive Q Network")
        # Adaptive Q-Network can handle variable input channels (used for continuous learning)
        policy_net = AdaptiveQNetwork(MAX_INPUT_CHANNELS, num_actions).to(device)
    else: 
        print("using default Q Network")
        # Standard Q-Network with fixed input channels
        policy_net = QNetwork(in_channels, num_actions).to(device)
        
    # For curriculum learning, initialize teacher networks to preserve knowledge
    if curriculum and adapnet:
        teacher_first_net = AdaptiveQNetwork(MAX_INPUT_CHANNELS, num_actions).to(device)
    else:
        teacher_first_net = QNetwork(in_channels, num_actions).to(device)

    # Initialize second-order (metacognitive) network if meta flag is set
    if meta:
        second_order_net = SecondOrderNetwork(in_channels).to(device)
        if curriculum:
            # For curriculum learning, we need a teacher second-order network too
            teacher_second_net = SecondOrderNetwork(in_channels).to(device)
    else:
        second_order_net = None

    # Initialize replay buffer parameters
    replay_start_size = 0
    
    # Initialize target network if enabled
    if not target_off:
        if adapnet:
            target_net = AdaptiveQNetwork(MAX_INPUT_CHANNELS, num_actions).to(device)
        else:
            target_net = QNetwork(in_channels, num_actions).to(device)
            
        # Initialize target network with policy network weights
        target_net.load_state_dict(policy_net.state_dict())

    # Initialize replay buffer if enabled
    if not replay_off:
        r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
        replay_start_size = REPLAY_START_SIZE
    else:
        r_buffer = None

    # Load checkpoint if path is provided
    if load_path is not None and isinstance(load_path, str):
        print("loading model stored in ", load_path)
        try:
            # Multiple attempts to load the checkpoint with different methods
            # to handle potential compatibility issues
            checkpoint = torch.load(load_path)
        except Exception as e1:
            try:
                checkpoint = torch.load(load_path, weights_only=False, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            except Exception as e2:
                checkpoint = torch.load(load_path, pickle_module=torch.serialization.pickle, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                checkpoint = torch.load(load_path)
                
        # Extract learning rates from the checkpoint
        STEP_SIZE = checkpoint['lr_first']
        STEP_SIZE_SECOND = checkpoint['lr_second']
        
        # Load previous loss values for continuous learning if available
        if len(checkpoint['validation_loss_one']) > 0:
            previous_loss = checkpoint['validation_loss_one'][-1]
        else:
            previous_loss = checkpoint['training_loss_one'][-1]
        
        if meta and len(checkpoint['validation_loss_two']) > 0:
            previous_loss_two = checkpoint['validation_loss_two'][-1]
        else: 
            previous_loss_two = None

    else:
        # Initialize learning rates if no checkpoint is loaded
        STEP_SIZE = step_size1
        STEP_SIZE_SECOND = step_size2
        
    print("current lr 1 is: ", STEP_SIZE, " and lr 2 is: ", STEP_SIZE_SECOND)
        
    # Initialize optimizers with appropriate learning rates
    optimizer = optim.Adam(policy_net.parameters(), lr=STEP_SIZE, eps=MIN_SQUARED_GRAD)
    scheduler1 = StepLR(optimizer, step_size=1000, gamma=scheduler_step)  

    if meta:
        # Initialize second optimizer for metacognitive network if enabled
        optimizer2 = optim.Adam(second_order_net.parameters(), lr=STEP_SIZE_SECOND, eps=MIN_SQUARED_GRAD)
        scheduler2 = StepLR(optimizer2, step_size=1000, gamma=scheduler_step)  

    # Initialize training metrics and counters
    e_init = 0               # Initial episode counter
    t_init = 0               # Initial frame counter
    policy_net_update_counter_init = 0  # Counter for policy network updates
    avg_return_init = 0.0    # Initial average return
    data_return_init = []    # List to store returns
    frame_stamp_init = []    # List to store frame counts
    episode_stamps_init = [] # List to store episode counts
    loss_ones_init = []      # List to store first-order network losses
    loss_twos_init = []      # List to store second-order network losses
    validation_returns_init = [] # List to store validation returns
    
    # Additional validation metrics
    validation_losses_one_init = []
    validation_loss_two_init = []
    validation_frames_init = []
    validation_episodes_init = []

    # Load model state and metrics from checkpoint if provided
    if load_path is not None and isinstance(load_path, str):
        print("loading model stored in ", load_path)

        # Helper function to load partial state dictionaries when architectures may not match exactly
        def load_partial_state_dict(model_original, state_dict):
            # Create a deep copy of the model's state_dict, not just the model's state dict
            model = copy.deepcopy(model_original)
            model_state = model.state_dict()
            for name, param in state_dict.items():
                # Only copy weights with the same shape
                if name in model_state and model_state[name].shape == param.shape:
                    model_state[name].copy_(param)
                    print(f"Successful copy for {name}")
                else:
                    print(f"Skipping {name} due to shape mismatch")
                    
            model.load_state_dict(model_state)
            return model
    
        # Load teacher networks for curriculum learning
        if curriculum and not(curriculum_evaluation):
            teacher_first_net = load_partial_state_dict(policy_net, checkpoint['policy_net_state_dict'])

        # Load policy network weights based on configuration
        if not adapnet:
            policy_net = load_partial_state_dict(policy_net, checkpoint['policy_net_state_dict'])
        elif (not curriculum) or curriculum_evaluation:
            print("load first order weights")
            policy_net = copy.deepcopy(policy_net)
            policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            
        load_optimizer_copie = copy.deepcopy(checkpoint['optimizer_state_dict'])
        
        # Load optimizer states based on the learning mode
        if (not curriculum) or (curriculum_evaluation and (current_optimizer not in checkpoint)):
            print("load optimizers")
            
            optimizer.load_state_dict(load_optimizer_copie)
            if meta:
                load_optimizer2_copie = copy.deepcopy(checkpoint['optimizer2_state_dict'])
                optimizer2.load_state_dict(load_optimizer2_copie)
                
        elif curriculum_evaluation:
            print("load optimizers CURRICULUM EVALUATION")
            optimizer_current_load = copy.deepcopy(checkpoint[current_optimizer])
            optimizer.load_state_dict(optimizer_current_load)
            if meta:
                optimizer2_current_load = copy.deepcopy(checkpoint[current_optimizer2])
                optimizer2.load_state_dict(optimizer2_current_load)
                
        elif curriculum and meta:
            print("load optimizer 2 CURRICULUM")
            optimizer2_current_load = copy.deepcopy(checkpoint['optimizer2_state_dict'])
            optimizer2.load_state_dict(optimizer2_current_load)
            
        # Load second-order network states if meta is enabled
        if meta:
            if curriculum and (not(curriculum_evaluation)):
                teacher_second_net_copie = copy.deepcopy(checkpoint['second_net_state_dict'])
                teacher_second_net.load_state_dict(teacher_second_net_copie)
            
            if (not curriculum) or curriculum_evaluation:
                print("starting evaluation, loaded second order")
                second_order_net_copie = copy.deepcopy(checkpoint['second_net_state_dict'])
                second_order_net.load_state_dict(second_order_net_copie)
            
        # Load target network if enabled
        if not target_off:
            if not adapnet:
                target_net = load_partial_state_dict(target_net, checkpoint['policy_net_state_dict'])
            else:
                target_net = copy.deepcopy(target_net)
                target_net.load_state_dict(checkpoint['policy_net_state_dict'])
                        
        # Load replay buffer if enabled
        if (not replay_off) and ((not curriculum) or curriculum_evaluation):
            print("loading replay buffer")
            if curriculum_evaluation and (current_replay_buffer in checkpoint):
                
                buffer_load_current = copy.deepcopy(checkpoint[current_replay_buffer])
                
                if isinstance(buffer_load_current, tuple):
                    buffer_load_current = buffer_load_current[0]
                
                r_buffer = replay_buffer(buffer_load_current['buffer_size'])
                r_buffer.buffer = buffer_load_current['buffer']
                r_buffer.location = buffer_load_current['location']
                
            else:
                buffer_load_previous = copy.deepcopy(checkpoint['replay_buffer'])
                r_buffer = replay_buffer(buffer_load_previous['buffer_size'])
                r_buffer.buffer = buffer_load_previous['buffer']
                r_buffer.location = buffer_load_previous['location']
        
        # Initialize counters based on curriculum learning or normal training
        if curriculum and (not curriculum_evaluation):
            avg_return_init = 0
            policy_net_update_counter_init = 0
            t_init = 0
            reduce_curriculum = checkpoint['frame_stamps'][-1]
            e_init = 0
            
        elif curriculum_evaluation:
            t_init = NUM_FRAMES
        else: 
            t_init = checkpoint['frame_stamps'][-1]
            e_init = checkpoint['episode_stamps'][-1]
            avg_return_init = checkpoint['avg_return']
            policy_net_update_counter_init = checkpoint['policy_net_update_counter']

        # Load training metrics    
        data_return_init = checkpoint['return_per_run']
        frame_stamp_init = checkpoint['frame_stamp_per_run']

        # Load additional metrics
        episode_stamps_init = checkpoint['episode_stamps']
        loss_ones_init = checkpoint['training_loss_one']
        validation_returns_init = checkpoint['validation_returns']
        validation_losses_one_init = checkpoint['validation_loss_one']
        validation_frames_init = checkpoint['validation_frame_stamps']
        validation_episodes_init = checkpoint['validation_episode_stamps']
        
        if meta:
            loss_twos_init = checkpoint['training_loss_two']
            validation_loss_two_init = checkpoint['validation_loss_two']

    # Initialize data containers with loaded values or empty lists
    data_return = data_return_init
    frame_stamp = frame_stamp_init
    episode_stamps = episode_stamps_init
    avg_return = avg_return_init
    loss_ones = loss_ones_init
    loss_twos = loss_twos_init
    
    validation_returns = validation_returns_init
    validation_losses_one = validation_losses_one_init
    validation_losses_two = validation_loss_two_init
    validation_frames = validation_frames_init
    validation_episodes = validation_episodes_init
    
    # Additional tracking for detailed loss components
    validation_losses_one_task = []
    validation_losses_two_task = []
    validation_losses_one_distillation = []
    validation_losses_two_distillation = []
    validation_losses_one_feature = []
    validation_losses_two_feature = []

    # Initialize training counters
    t = t_init
    e = e_init
    policy_net_update_counter = policy_net_update_counter_init
    t_start = time.time()

    # Flags for curriculum termination
    is_curriculum_terminated = False
    evaluation_curriculum_counter = 0
    
    # Main training loop - continues until frame limit or curriculum termination
    while ((t < (NUM_FRAMES - reduce_curriculum)) and (not is_curriculum_terminated)) or (curriculum_evaluation and (evaluation_curriculum_counter < number_validation_curriculum)):
        
        # Learning rate decay every 10000 frames
        if t % 10000 == 0:
            STEP_SIZE = STEP_SIZE * scheduler_step
            STEP_SIZE_SECOND = STEP_SIZE_SECOND * scheduler_step
        
        # Set teacher networks to evaluation mode for curriculum learning
        if curriculum:
            teacher_first_net.eval()
            if meta:
                teacher_second_net.eval()
        
        # Set networks to training mode
        policy_net.train()
        if meta:
            second_order_net.train()
        if not target_off:
            target_net.train()
            
        loss_one = None
        loss_two = None
        
        # Initialize episode return and counter
        G = 0.0
        counter_episode = 0

        # Initialize the environment and start state
        env.reset()
        s = get_state(env.state())
        is_terminated = False
        
        # Episode loop - continues until termination or frame limit reached
        while(not is_terminated) and t < (NUM_FRAMES - reduce_curriculum) and (not curriculum_evaluation):
            counter_episode += 1
            
            # Generate interaction with environment
            s_prime, action, reward, is_terminated = world_dynamics(t, replay_start_size, num_actions, s, env, cascade_iterations_1, cascade_iterations_2)

            sample = None
            if replay_off:
                # Direct training on current transition if replay is disabled
                sample = [transition(s, s_prime, action, reward, is_terminated)]
            else:
                # Add transition to replay buffer
                r_buffer.add(s, s_prime, action, reward, is_terminated)

                # Start learning when buffer has enough data
                if t > REPLAY_START_SIZE and len(r_buffer.buffer) >= BATCH_SIZE:
                    # Sample a batch from replay buffer
                    sample = r_buffer.sample(BATCH_SIZE)

            # Train every TRAINING_FREQ frames when a sample is available
            if t % TRAINING_FREQ == 0 and sample is not None:
                
                if loss_one == None:
                    loss_one = 0
                    
                if loss_two == None:
                    loss_two = 0
                    
                if not target_off:
                    policy_net_update_counter += 1

                # Training with metacognitive network if enabled
                if meta: 
                    if previous_loss != None and (not(evaluation_continuous_learning)):
                        # Full training with distillation and feature preservation for continuous learning
                        loss_first, loss_second, (loss_task, previous_task_loss, feature_loss), (task_loss_second, previous_task_loss_second, feature_loss_second) = train(sample, policy_net, meta, alpha, cascade_iterations_1, cascade_iterations_2, True)
                    
                    else:
                        # Standard training without distillation
                        loss_first, loss_second = train(sample, policy_net, meta, alpha, cascade_iterations_1, cascade_iterations_2, True)

                    loss_one += loss_first.item()
                    loss_two += loss_second.item()

                else:
                    # Training without metacognitive network
                    if previous_loss != None and (not(evaluation_continuous_learning)):
                        # Training with distillation for continuous learning
                        loss_first, (loss_task, previous_task_loss, feature_loss) = train(sample, policy_net, meta, alpha, cascade_iterations_1, cascade_iterations_2, True)
                    
                    else:
                        # Standard training
                        loss_first = train(sample, policy_net, meta, alpha, cascade_iterations_1, cascade_iterations_2, True)

                    loss_one += loss_first.item()
                
            # Update target network at specified intervals
            if not target_off and policy_net_update_counter > 0 and policy_net_update_counter % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Update cumulative reward for the episode
            G += reward.item()

            # Increment frame counter
            t += 1

            # Update current state
            s = s_prime

        # Episode completed - update metrics
        if loss_one != None:
            # Calculate average loss for the episode
            loss_one = loss_one / counter_episode
            
        loss_ones.append(loss_one)

        # Increment episode counter
        e += 1

        # Store episode metrics
        data_return.append(G)
        frame_stamp.append(t)
        episode_stamps.append(e)
        
        # Store second-order network loss if enabled
        if meta:
            if loss_two != None:
                loss_two = loss_two / counter_episode
            loss_twos.append(loss_two)

        # Update exponential moving average of return
        avg_return = 0.99 * avg_return + 0.01 * G
        
        # Periodic validation and checkpoint saving
        if e % checkpoint_iteration == 0 or curriculum_evaluation:
            
            if curriculum_evaluation:
                print(t, alpha, replay_start_size, num_actions, "SPACE")

            # Initialize validation metric containers
            v_rewards = []
            v_losses_one = []
            v_losses_two = []
            
            v_losses_one_task = []
            v_losses_one_distillation = []
            v_losses_one_feature = []
            
            v_losses_two_task = []
            v_losses_two_distillation = []
            v_losses_two_feature = []
            
            # Run multiple validation episodes
            for _ in range(validation_iterations):
                random_number = random.randint(t, t*2)
                
                if curriculum_evaluation:
                    # Simplified evaluation for curriculum evaluation mode
                    val_reward, val_loss_one, val_loss_two = evaluation(
                        env, meta, random_number, target_off, target_net, alpha,
                        cascade_iterations_1, cascade_iterations_2, replay_start_size, num_actions, replay_off,
                        r_buffer)
                    
                elif previous_loss != None and (not(evaluation_continuous_learning)):
                    # Detailed evaluation with loss component breakdown for continuous learning
                    val_reward, val_loss_one, val_loss_two, (loss_one_task, loss_one_distillation, loss_one_feature), (loss_two_task, loss_two_distillation, loss_two_feature) = evaluation(
                        env, meta, random_number, target_off, target_net, alpha,
                        cascade_iterations_1, cascade_iterations_2, replay_start_size, num_actions, replay_off,
                        r_buffer)
                    
                    # Store detailed loss components
                    v_losses_one_task.append(loss_one_task)
                    v_losses_one_distillation.append(loss_one_distillation)
                    v_losses_one_feature.append(loss_one_feature)
                    
                    v_losses_two_task.append(loss_two_task)
                    v_losses_two_distillation.append(loss_two_distillation)
                    v_losses_two_feature.append(loss_two_feature)

                else: 
                    # Standard evaluation
                    val_reward, val_loss_one, val_loss_two = evaluation(
                        env, meta, random_number, target_off, target_net, alpha,
                        cascade_iterations_1, cascade_iterations_2, replay_start_size, num_actions, replay_off,
                        r_buffer)
                    
                # Store individual validation results
                v_rewards.append(val_reward)
                v_losses_one.append(val_loss_one)
                v_losses_two.append(val_loss_two)
                
            # Increment curriculum evaluation counter
            evaluation_curriculum_counter += 1
            
            # Calculate validation metric means
            validation_reward = numpy.mean(v_rewards)
            validation_loss_one = numpy.mean(v_losses_one)
            
            # Calculate validation metric standard deviations
            validation_reward_std = numpy.std(v_rewards)
            validation_loss_one_std = numpy.std(v_losses_one)
            
            # Process detailed loss components if available (continuous learning)
            if previous_loss != None and (not(evaluation_continuous_learning)):
                # Calculate means and standard deviations for detailed loss components
                validation_loss_one_task = numpy.mean(v_losses_one_task)
                validation_loss_one_distillation = numpy.mean(v_losses_one_distillation)
                validation_loss_one_feature = numpy.mean(v_losses_one_feature)
                
                validation_loss_one_task_std = numpy.std(v_losses_one_task)
                validation_loss_one_distillation_std = numpy.std(v_losses_one_distillation)
                validation_loss_one_feature_std = numpy.std(v_losses_one_feature)
                
                # Store detailed loss components
                validation_losses_one_task.append(validation_loss_one_task)
                validation_losses_one_distillation.append(validation_loss_one_distillation)
                validation_losses_one_feature.append(validation_loss_one_feature)
            
                # Process second-order network loss components if meta is enabled
                if meta: 
                    validation_loss_two_task = numpy.mean(v_losses_two_task)
                    validation_loss_two_distillation = numpy.mean(v_losses_two_distillation)
                    validation_loss_two_feature = numpy.mean(v_losses_two_feature)  
                    
                    validation_loss_two_task_std = numpy.std(v_losses_two_task)
                    validation_loss_two_distillation_std = numpy.std(v_losses_two_distillation)
                    validation_loss_two_feature_std = numpy.std(v_losses_two_feature)
                    
                    validation_losses_two_task.append(validation_loss_two_task)
                    validation_losses_two_distillation.append(validation_loss_two_distillation)
                    validation_losses_two_feature.append(validation_loss_two_feature)
                    
                else:
                    # Initialize as None if meta is disabled
                    validation_loss_two_task = None
                    validation_loss_two_distillation = None
                    validation_loss_two_feature = None
                    
                    validation_loss_two_task_std = None
                    validation_loss_two_distillation_std = None
                    validation_loss_two_feature_std = None
                    
            # Process second-order network overall loss if meta is enabled
            if meta:
                validation_loss_two = numpy.mean(v_losses_two)
                validation_loss_two_std = numpy.std(v_losses_two)
                validation_losses_two.append(validation_loss_two)
                 
            else: 
                validation_loss_two = None
                validation_loss_two_std = None
            
            # Store overall validation metrics
            validation_returns.append(validation_reward)
            validation_losses_one.append(validation_loss_one)
            validation_losses_two.append(validation_loss_two)
            validation_frames.append(t)
            validation_episodes.append(e)
            
            # Special handling for curriculum evaluation mode
            if curriculum_evaluation:
                # Format and print validation results
                if meta: 
                    result_string = (
                        f" Validation metrics:\n"
                        f" Reward: {numpy.around(validation_reward,2)} ± {numpy.around(validation_reward_std,2)}\n"
                        f" Loss 1: {numpy.around(validation_loss_one,5)} ± {numpy.around(validation_loss_one_std,5)}\n"
                        f" Loss 2: {numpy.around(validation_loss_two,5)} ± {numpy.around(validation_loss_two_std,5)}"
                    )
                else: 
                    result_string = (
                        f" Validation metrics:\n"
                        f" Reward: {numpy.around(validation_reward,2)} ± {numpy.around(validation_reward_std,2)}\n"
                        f" Loss 1: {numpy.around(validation_loss_one,5)} ± {numpy.around(validation_loss_one_std,5)}"
                    )         
                    
                print(result_string)
                logging.info(result_string)
                
                # Save only validation-related metrics
                torch.save({
                    'validation_returns': validation_reward,
                    'validation_returns_std': validation_reward_std,
                    'validation_loss_one': validation_loss_one,
                    'validation_loss_one_std': validation_loss_one_std,
                    'validation_loss_two': validation_loss_two,
                    'validation_loss_two_std': validation_loss_two_std,
                    'lr_first':STEP_SIZE,
                    'lr_second':STEP_SIZE_SECOND
                    
                }, output_file_name + "_evaluation_results_curriculum")
                
                is_curriculum_terminated=True
                
            elif meta:
                
                if loss_one!=None and loss_two!=None:
                    print("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                            str(numpy.around(avg_return, 2)) + " ( val "+str(numpy.around(validation_reward,2)) +" )"  + " | loss first order: " +
                            str(numpy.around(loss_one,5))  + " ( val "+str(numpy.around(validation_loss_one,5)) +" )" + " | loss second order: " +
                            str(numpy.around(loss_two,5)) + " ( val "+str(numpy.around(validation_loss_two,5)) +" )" + " | Frame: " + str(t)+" | Time per frame: " +str( numpy.around( (time.time()-t_start)/t  ,5)) )
            
                    
                    logging.info("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                            str(numpy.around(avg_return, 2)) + " ( val "+str(numpy.around(validation_reward,2)) +" )" + " | loss first order: " +
                            str(numpy.around(loss_one,5)) + " ( val "+str(numpy.around(validation_loss_one,5)) +" )"  + " | loss second order: " +
                            str(numpy.around(loss_two,5))+ " ( val "+str(numpy.around(validation_loss_two,5)) +" )" + " | Frame: " + str(t)+" | Time per frame: " +str( numpy.around( (time.time()-t_start)/t  ,5)) )


                    if previous_loss!=None and (not(evaluation_continuous_learning)):

                        print("Task loss: " +  " (val "+str(numpy.around(validation_loss_one_task, 5)) +")" + " | Distillation loss: " +
                                " (val "+str(numpy.around(validation_loss_one_distillation, 5)) +")" + " | Feature loss: " +
                                " (val "+str(numpy.around(validation_loss_one_feature, 5)) +")"  + '| |  Task loss second: '  +  
                                " (val "+str(numpy.around(validation_loss_two_task, 5)) +")" + " | Distillation loss: " +
                                " (val "+str(numpy.around(validation_loss_two_distillation, 5)) +")" + " | Feature loss: " +
                                " (val "+str(numpy.around(validation_loss_two_feature, 5)) +")" )
                        
                        logging.info("Task loss: " +  " (val "+str(numpy.around(validation_loss_one_task, 5)) +")" + " | Distillation loss: " +
                                " (val "+str(numpy.around(validation_loss_one_distillation, 5)) +")" + " | Feature loss: " +
                                " (val "+str(numpy.around(validation_loss_one_feature, 5)) +")"  + '| |  Task loss second: '  +
                                " (val "+str(numpy.around(validation_loss_two_task, 5)) +")" + " | Distillation loss: " +
                                " (val "+str(numpy.around(validation_loss_two_distillation, 5)) +")" + " | Feature loss: " +
                                " (val "+str(numpy.around(validation_loss_two_feature, 5)) +")" )
                    
                if checkpoints:
                    torch.save({
                                'returns': data_return,
                                'frame_stamps': frame_stamp,
                                'policy_net_state_dict': policy_net.state_dict(),
                                'second_net_state_dict': second_order_net.state_dict(),
                                'episode_stamps': episode_stamps,
                                'training_loss_one': loss_ones,
                                'training_loss_two': loss_twos,
                                'validation_returns': validation_returns,
                                'validation_frame_stamps': validation_frames,
                                'validation_episode_stamps': validation_episodes,
                                'validation_loss_one': validation_losses_one,
                                'validation_loss_two': validation_losses_two,
                                'target_net_state_dict': target_net.state_dict() if not target_off else [],
                                'optimizer_state_dict': optimizer.state_dict(),
                                'optimizer2_state_dict': optimizer2.state_dict(),
                                'avg_return': avg_return,
                                'return_per_run': data_return,
                                'frame_stamp_per_run': frame_stamp,
                                'replay_buffer': {
                                    'buffer_size': r_buffer.buffer_size, 
                                    'location': r_buffer.location, 
                                    'buffer': r_buffer.buffer
                                } if not replay_off else 0,
                                'policy_net_update_counter':policy_net_update_counter,
                                'lr_first':STEP_SIZE,
                                'lr_second':STEP_SIZE_SECOND

                    }, output_file_name + "_last_checkpoint")
                
                
            else: 
                
                if loss_one!=None :
                    print("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                            str(numpy.around(avg_return, 2))  + " ( val "+str(numpy.around(validation_reward,2)) +" )"  + " | loss first order: " +
                            str(numpy.around(loss_one,5)) + " ( val "+str(numpy.around(validation_loss_one,5)) +" )" + " | Frame: " + str(t)+" | Time per frame: " +str( numpy.around( (time.time()-t_start)/t  ,5)) )
                    

                    logging.info("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                            str(numpy.around(avg_return, 2)) + " ( val "+str(numpy.around(validation_reward,2)) +" )"  + " | loss first order: " +
                            str(numpy.around(loss_one,5))  + " ( val "+str(numpy.around(validation_loss_one,5)) +" )"+  " | Frame: " + str(t)+" | Time per frame: " +str( numpy.around( (time.time()-t_start)/t  ,5)) )
                    
                    if previous_loss!=None and (not(evaluation_continuous_learning)):
                    
                        print("Task loss: " +  " (val "+str(numpy.around(validation_loss_one_task, 5)) +")" + " | Distillation loss: " +
                            " (val "+str(numpy.around(validation_loss_one_distillation, 5)) +")" + " | Feature loss: " +
                            " (val "+str(numpy.around(validation_loss_one_feature, 5)) +")"  )
                        logging.info("Task loss: " +  " (val "+str(numpy.around(validation_loss_one_task, 5)) +")" + " | Distillation loss: " +
                            " (val "+str(numpy.around(validation_loss_one_distillation, 5)) +")" + " | Feature loss: " +
                            " (val "+str(numpy.around(validation_loss_one_feature, 5)) +")"   )
                
                if checkpoints:
                    torch.save({
                                'returns': data_return,
                                'frame_stamps': frame_stamp,
                                'policy_net_state_dict': policy_net.state_dict(),
                                'episode_stamps': episode_stamps,
                                'training_loss_one': loss_ones,
                                'validation_returns': validation_returns,
                                'validation_frame_stamps': validation_frames,
                                'validation_episode_stamps': validation_episodes,
                                'validation_loss_one': validation_losses_one,
                                'target_net_state_dict': target_net.state_dict() if not target_off else [],
                                'optimizer_state_dict': optimizer.state_dict(),
                                'avg_return': avg_return,
                                'return_per_run': data_return,
                                'frame_stamp_per_run': frame_stamp,
                                'replay_buffer': {
                                    'buffer_size': r_buffer.buffer_size, 
                                    'location': r_buffer.location, 
                                    'buffer': r_buffer.buffer
                                } if not replay_off else 0,
                                'policy_net_update_counter':policy_net_update_counter,
                                'lr_first':STEP_SIZE,
                                'lr_second':STEP_SIZE_SECOND
                                
                    }, output_file_name + "_last_checkpoint")              

    if not curriculum_evaluation:
        # Print final logging info
        
        if not curriculum: 
            # Write data to file
            if meta: 
                torch.save({
                    'returns': data_return,
                    'frame_stamps': frame_stamp,
                    'policy_net_state_dict': policy_net.state_dict(),
                    'second_net_state_dict': second_order_net.state_dict(),
                    'episode_stamps': episode_stamps,
                    'training_loss_one': loss_ones,
                    'training_loss_two': loss_twos,
                    'validation_returns': validation_returns,
                    'validation_frame_stamps': validation_frames,
                    'validation_episode_stamps': validation_episodes,
                    'validation_loss_one': validation_losses_one,
                    'validation_loss_two': validation_losses_two,
                    'target_net_state_dict': target_net.state_dict() if not target_off else 0,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer2_state_dict': optimizer2.state_dict(),
                    'avg_return': avg_return,
                    'return_per_run': data_return,
                    'frame_stamp_per_run': frame_stamp,
                    'replay_buffer': {
                        'buffer_size': r_buffer.buffer_size, 
                        'location': r_buffer.location, 
                        'buffer': r_buffer.buffer
                    } if not replay_off else 0,
                    'policy_net_update_counter':policy_net_update_counter,
                    'lr_first':STEP_SIZE,
                    'lr_second':STEP_SIZE_SECOND


                }, output_file_name + "_data_and_weights")
            else:
                torch.save({
                    'returns': data_return,
                    'frame_stamps': frame_stamp,
                    'policy_net_state_dict': policy_net.state_dict(),
                    'episode_stamps': episode_stamps,
                    'training_loss_one': loss_ones,
                    'validation_returns': validation_returns,
                    'validation_frame_stamps': validation_frames,
                    'validation_episode_stamps': validation_episodes,
                    'validation_loss_one': validation_losses_one,
                    'validation_loss_two': validation_losses_two,
                    'target_net_state_dict': target_net.state_dict() if not target_off else 0,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_return': avg_return,
                    'return_per_run': data_return,
                    'frame_stamp_per_run': frame_stamp,
                    'replay_buffer': {
                        'buffer_size': r_buffer.buffer_size, 
                        'location': r_buffer.location, 
                        'buffer': r_buffer.buffer
                    } if not replay_off else 0,
                    
                    'policy_net_update_counter':policy_net_update_counter,
                    'lr_first':STEP_SIZE,
                    'lr_second':STEP_SIZE_SECOND
                }, output_file_name + "_data_and_weights")
        else:
                # Write data to file
            
            first_game_curriculum='breakout'
            second_game_curriculum='space_invaders'
            
            if meta:
                save_dict = {
                    'returns': data_return,
                    'frame_stamps': frame_stamp,
                    'policy_net_state_dict': policy_net.state_dict(),
                    'second_net_state_dict': second_order_net.state_dict(),
                    'episode_stamps': episode_stamps,
                    'training_loss_one': loss_ones,
                    'training_loss_two': loss_twos,
                    'validation_returns': validation_returns,
                    'validation_frame_stamps': validation_frames,
                    'validation_episode_stamps': validation_episodes,
                    'validation_loss_one': validation_losses_one,
                    'validation_loss_two': validation_losses_two,
                    'target_net_state_dict': target_net.state_dict() if not target_off else [],
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer2_state_dict': optimizer2.state_dict(),
                    'avg_return': avg_return,
                    'return_per_run': data_return,
                    'frame_stamp_per_run': frame_stamp,
                    'replay_buffer': {
                        'buffer_size': r_buffer.buffer_size, 
                        'location': r_buffer.location, 
                        'buffer': r_buffer.buffer
                    } if not replay_off else 0,
                    'policy_net_update_counter': policy_net_update_counter,
                    'lr_first': STEP_SIZE,
                    'lr_second': STEP_SIZE_SECOND,
                    'optimizer_space_invaders': None,
                    'optimizer_breakout': None,
                    'optimizer_seaquest': None,
                    'optimizer_freeway': None,
                    'optimizer_asterix': None,
                    'optimizer2_space_invaders': None,
                    'optimizer2_breakout': None,
                    'optimizer2_seaquest': None,
                    'optimizer2_freeway': None,
                    'optimizer2_asterix': None,
                    'replay_buffer_space_invaders': None,
                    'replay_buffer_breakout': None,
                    'replay_buffer_seaquest': None,
                    'replay_buffer_freeway': None,
                    'replay_buffer_asterix': None
                }
                
                # Update the dynamic keys with current optimizer and replay buffer
                if game_name!=second_game_curriculum:
                    #validation done for 2nd game only, as for the 1st game in the curriculum it's not neccesary to do curriculum = True
                    for i in range(len(load_list)):
                        save_dict[load_list[i]] = checkpoint[load_list[i]]
                else:
                    save_dict['optimizer_' + first_game_curriculum] = checkpoint['optimizer_state_dict']
                    save_dict['optimizer2_' + first_game_curriculum] = checkpoint['optimizer2_state_dict']
                    save_dict['replay_buffer_' + first_game_curriculum] =  checkpoint['replay_buffer']
                    
                save_dict[current_optimizer] = optimizer.state_dict()
                save_dict[current_optimizer2] = optimizer2.state_dict()
                save_dict[current_replay_buffer] =  {
                        'buffer_size': r_buffer.buffer_size, 
                        'location': r_buffer.location, 
                        'buffer': r_buffer.buffer
                    } if not replay_off else 0
                
            else:
                save_dict = {
                    'returns': data_return,
                    'frame_stamps': frame_stamp,
                    'policy_net_state_dict': policy_net.state_dict(),
                    'episode_stamps': episode_stamps,
                    'training_loss_one': loss_ones,
                    'validation_returns': validation_returns,
                    'validation_frame_stamps': validation_frames,
                    'validation_episode_stamps': validation_episodes,
                    'validation_loss_one': validation_losses_one,
                    'validation_loss_two': validation_losses_two,
                    'target_net_state_dict': target_net.state_dict() if not target_off else [],
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_return': avg_return,
                    'return_per_run': data_return,
                    'frame_stamp_per_run': frame_stamp,
                    'replay_buffer': {
                        'buffer_size': r_buffer.buffer_size, 
                        'location': r_buffer.location, 
                        'buffer': r_buffer.buffer
                    } if not replay_off else 0,
                    'policy_net_update_counter': policy_net_update_counter,
                    'lr_first': STEP_SIZE,
                    'lr_second': STEP_SIZE_SECOND,
                    'optimizer_space_invaders': None,
                    'optimizer_breakout': None,
                    'optimizer_seaquest': None,
                    'optimizer_freeway': None,
                    'optimizer_asterix': None,
                    'replay_buffer_space_invaders': None,
                    'replay_buffer_breakout': None,
                    'replay_buffer_seaquest': None,
                    'replay_buffer_freeway': None,
                    'replay_buffer_asterix': None
                }
                # Update the dynamic keys with current optimizer and replay buffer
                if game_name!=second_game_curriculum:
                    #validation done for 2nd game only, as for the 1st game in the curriculum it's not neccesary to do curriculum = True
                    for i in range(len(load_list)):
                        save_dict[load_list[i]] = checkpoint[load_list[i]]
                else:
                    save_dict['optimizer_' + first_game_curriculum] = checkpoint['optimizer_state_dict']
                    save_dict['replay_buffer_' + first_game_curriculum] = checkpoint['replay_buffer']
                    
                save_dict[current_optimizer] = optimizer.state_dict()
                save_dict[current_replay_buffer] =  {
                        'buffer_size': r_buffer.buffer_size, 
                        'location': r_buffer.location, 
                        'buffer': r_buffer.buffer
                    } if not replay_off else 0
                
            torch.save(save_dict, output_file_name + "_data_and_weights")



def process_return_stamps(num_runs, window_size, returns_runs_episodes, frame_stamps_runs_episodes):


    # Acquire unique frames across all runs
    flatten_frame_stamps = [item for sublist in frame_stamps_runs_episodes for item in sublist]
    flatten_frame_stamps.sort()
    unique_frame_stamps = numpy.unique(flatten_frame_stamps).tolist()

    # Get smoothed performance measures
    mv_avg = []
    mv_frames = []
    for _, val in enumerate(zip(returns_runs_episodes, frame_stamps_runs_episodes)):
        mv_avg_per_run = numpy.convolve(val[0], numpy.ones((window_size,)) / window_size, mode='valid')
        mv_frames_per_run = [val[1][i+window_size-1] for i, frame
                             in enumerate(numpy.arange(len(val[1]) - window_size + 1))]

        mv_avg.append(mv_avg_per_run)
        mv_frames.append(mv_frames_per_run)

    run_index = [0]*num_runs
    returns_by_frame = numpy.zeros((len(unique_frame_stamps), num_runs))

    # Fill measure into unique frames for each run
    for index, bucket in enumerate(unique_frame_stamps):
        for run in numpy.arange(num_runs):
            if len(mv_frames[run])-1 > run_index[run] and mv_frames[run][run_index[run]] == bucket:
                run_index[run] += 1
            returns_by_frame[index][run] = mv_avg[run][run_index[run]]
            
    return returns_by_frame, unique_frame_stamps


def process_data(num_runs, file_name, window_size, window_size_validation, setting):

    # Aggregate all the runs from different files
    returns_runs_episodes = []
    frame_stamps_runs_episodes = []
    
    returns_runs_episodes_validation = []
    frame_stamps_runs_episodes_validation = []
    
    print(file_name)
    for i in numpy.arange(1, num_runs+1):
        print(i)
        
        # First try with normal loading
        if setting in ['setting2', 'setting3', 'setting4', 'setting6']:
            data_and_weights = torch.load(file_name + "_" + str(i) + "_last_checkpoint")
        else:
            data_and_weights = torch.load(file_name + "_" + str(i) + "_data_and_weights")
       
        #print(data_and_weights['frame_stamps'][-1], len(data_and_weights['frame_stamps']))

        returns = data_and_weights['returns']
        frame_stamps = data_and_weights['frame_stamps']
        returns_validation = data_and_weights['validation_returns']
        frame_stamps_validation = data_and_weights['validation_frame_stamps']

        returns_runs_episodes.append(returns)
        frame_stamps_runs_episodes.append(frame_stamps)
        returns_runs_episodes_validation.append(returns_validation)
        frame_stamps_runs_episodes_validation.append(frame_stamps_validation)

    returns_by_frame , unique_frame_stamps = process_return_stamps(num_runs, window_size, returns_runs_episodes, frame_stamps_runs_episodes)
    returns_by_frame_validation , unique_frame_stamps_validation = process_return_stamps(num_runs, window_size_validation, returns_runs_episodes_validation, frame_stamps_runs_episodes_validation)


    # Save the processed data into a file
    torch.save({
        'returns': returns_by_frame,
        'unique_frames': unique_frame_stamps,
        'returns_validation': returns_by_frame_validation,
        'unique_frames_validation': unique_frame_stamps_validation
        
    }, file_name+"_processed_data")
    
    
def find_closest_index(arr, value):
    # Find index of closest value to target using numpy
    return numpy.searchsorted(arr, value, side='left')

# Modified plot_avg_return to accept game_idx parameter
def plot_avg_return(file_name, granularity, granularity_validation, setting, game_idx):
    dictionary_settings = {
        'setting1': '2nd Order Net',
        'setting2': '2nd Order Net\n(cascade, 1st Net)',
        'setting3': '2nd Order Net\n(cascade, 2nd Net)',
        'setting4': 'MAPS* - 2nd Order\n(cascade, both Nets)',
        'setting5': 'Baseline* - DQN',
        'setting6': 'DQN\n(cascade)'
    }
    
    setting_label = dictionary_settings[setting]
    
    # Load data
    plotting_data = torch.load(file_name + "_processed_data")
    returns = plotting_data['returns']
    unique_frames = plotting_data['unique_frames']
    returns_validation = plotting_data['returns_validation']
    unique_frames_validation = plotting_data['unique_frames_validation']
    
    cutoff_idx = find_closest_index(unique_frames, max_step)
    cutoff_idx_validation = find_closest_index(unique_frames_validation, max_step)
    
    unique_frames = unique_frames[:cutoff_idx]
    returns = returns[:cutoff_idx]
    unique_frames_validation = unique_frames_validation[:cutoff_idx_validation]
    returns_validation = returns_validation[:cutoff_idx_validation]
    
    # Prepare indices for granularity
    x_len = len(unique_frames)
    x_index = [i for i in numpy.arange(0, x_len, granularity)]
    x_len_validation = len(unique_frames_validation)
    x_index_validation = [i for i in numpy.arange(0, x_len_validation, granularity_validation)]
    
    # Prepare x-axis values
    x = numpy.array(unique_frames[::granularity])
    x_validation = numpy.array(unique_frames_validation[::granularity_validation])
    
    # Convert returns to numpy array and select values according to granularity
    returns_array = numpy.array(returns)[x_index]
    returns_array_validation = numpy.array(returns_validation)[x_index_validation]
    
    # Calculate mean and std across seeds (axis 1)
    mean_returns = numpy.mean(returns_array, axis=1)
    std_returns = numpy.std(returns_array, axis=1)
    mean_returns_validation = numpy.mean(returns_array_validation, axis=1)
    std_returns_validation = numpy.std(returns_array_validation, axis=1)
    
    # Find the order of magnitude of the last frame
    order = 6
    range_frames = int(unique_frames[-1] / (10**order))
    order_validation = 6
    range_frames_validation = int(unique_frames_validation[-1] / (10**order_validation))
    
    # Scale x-axis values
    x_scaled = x / (10**order)
    x_scaled_validation = x_validation / (10**order_validation)
    
    # Access the correct axes for this game
    train_ax = ax[game_idx * 2]
    val_ax = ax[game_idx * 2 + 1]
    
    # Plot mean line for training
    line_train = train_ax.plot(x_scaled, mean_returns)[0]
    line_color = line_train.get_color()
    
    # Add shaded error bars for training (±1 std)
    train_ax.fill_between(
        x_scaled,
        mean_returns - std_returns,
        mean_returns + std_returns, 
        alpha=0.2, 
        color=line_color
    )
    
    # Create ticks at regular intervals
    tick_positions = numpy.arange(0, range_frames + 1, 1)
    train_ax.set_xticks(tick_positions)
    tick_labels = [str(int(x)) for x in tick_positions]
    train_ax.set_xticklabels(tick_labels)
    
    # Plot validation data with same color
    val_ax.plot(x_scaled_validation, mean_returns_validation, color=line_color)
    val_ax.fill_between(
        x_scaled_validation,
        mean_returns_validation - std_returns_validation,
        mean_returns_validation + std_returns_validation, 
        alpha=0.2,
        color=line_color
    )
    
    # Create ticks at regular intervals for validation
    val_tick_positions = numpy.arange(0, range_frames_validation + 1, 1)
    val_ax.set_xticks(val_tick_positions)
    val_tick_labels = [str(int(x)) for x in val_tick_positions]
    val_ax.set_xticklabels(val_tick_labels)
    
    # Add to legend handles list only for the first game to avoid duplicates
    global legend_handles, legend_labels
    if game_idx == 0:
        legend_handles.append(line_train)
        legend_labels.append(setting_label)
        
    # Calculate final values for z-scores and return strings
    returns_4_zcores = returns_array[-1]
    returns_4_zcores_validation = returns_array_validation[-1]
    return_string = str(mean_returns[-1]) + ' + ' + str(std_returns[-1])
    return_string_validation = str(mean_returns_validation[-1]) + ' + ' + str(std_returns_validation[-1])
    
    return returns_4_zcores, returns_4_zcores_validation, return_string, return_string_validation

def calculate_z_score(data1, data2):
    mean1, mean2 = numpy.mean(data1), numpy.mean(data2)
    std1, std2 = numpy.std(data1, ddof=1), numpy.std(data2, ddof=1)
    n1, n2 = len(data1), len(data2)
    pooled_se = numpy.sqrt((std1**2 / n1) + (std2**2 / n2))
    z_score = (mean2 - mean1) / pooled_se
    return z_score

def z_score_results(returns_dict, label):
    baseline_setting = 'baseline DQN'
    baseline_returns = returns_dict[baseline_setting][label]
    
    table_data = {
        'Condition': [],
        'Z-Score': [],
        'Significant': []
    }
    
    for setting, data in returns_dict.items():
        if setting != baseline_setting:        
            z_score = calculate_z_score(baseline_returns, data[label])
            significance = abs(z_score) > norm.ppf(0.975)  # 95% confidence level
            table_data['Condition'].append(f'{baseline_setting} vs {setting}')
            table_data['Z-Score'].append(z_score)
            table_data['Significant'].append(significance)
    
    return table_data

def plot(filename):
    #base_games = ["breakout", "space_invaders",  "seaquest"]
    base_games = ["seaquest","asterix"]

    settings = ['setting1', 'setting2', 'setting3', 'setting4', 'setting5','setting6']
    #settings = ['setting1', 'setting5']
    
    # Global parameters
    global max_step, ax, legend_handles, legend_labels
    max_step = 1000000
    numruns = 3
    granularity = 20
    windowsize = 500
    windowsize_validation = 5
    granularity_validation = 1
    
    # Settings dictionary
    dictionary_settings = {
        'setting1': '2nd Order Net',
        'setting2': '2nd Order Net (cascade, 1st Net)',
        'setting3': '2nd Order Net (cascade, 2nd Net)',
        'setting4': '2nd Order Net (cascade, both Nets)',
        'setting5': 'baseline DQN',
        'setting6': 'baseline DQN (cascade)'
    }
    
    # Create figure with gridspec to have a separate area for the legend
    fig = plt.figure(figsize=[11, 6 * len(base_games)], dpi=300)
    
    
    # Create gridspec with 3 columns - one for each plot type and one for legend
    gs = fig.add_gridspec(len(base_games), 3, width_ratios=[5, 5, 1])
    
    # Initialize axes list and legend lists
    ax = []
    legend_handles = []
    legend_labels = []
    
    # Set style
    sns.set_style("ticks")
    sns.set_context("paper")
    
    # Process each game
    for game_idx, game in enumerate(base_games):
        # Create subplot axes for this game (training and validation)
        ax.append(fig.add_subplot(gs[game_idx, 0]))
        ax.append(fig.add_subplot(gs[game_idx, 1]))
        
        if game_idx == len(base_games)-1:
            fig.text(0.25, 0.035, 'Frames (×10^6)', va='center', ha='center', fontsize=7, fontweight='bold')
            fig.text(0.65, 0.035, 'Frames (×10^6)', va='center', ha='center', fontsize=7, fontweight='bold')

        
        # Add game name to the y-axis
        ax[game_idx*2].set_ylabel(game.replace('_', ' ').title(), fontsize=8, fontweight='bold')
        
        # Set specific parameters for the freeway game
        if game == 'freeway':
            granularity = 2
            windowsize = 50
        else:
            granularity = 20
            windowsize = 500
        
        # Dictionary to store results for this game
        returns_dict = {}
        
        for setting_idx, setting in enumerate(settings):
            # Construct path as per your version
            game_name = filename + game
            setting_path = game_name + '_' + settings[setting_idx] + '_steps2000000'
            file_path = Path(setting_path + "_processed_data")
            
            # Process data if needed
            if not file_path.is_file():
                process_data(int(numruns), setting_path, int(windowsize), int(windowsize_validation), setting)
            
            # Plot returns for this setting
            return_z_scores, return_z_scores_validation, return_string, return_string_validation = plot_avg_return(
                setting_path, int(granularity), int(granularity_validation), setting, game_idx)
            
            # Store results using your indexing
            returns_dict[dictionary_settings[settings[setting_idx]]] = {
                'returns': return_z_scores,
                'returns_validation': return_z_scores_validation,
                'return_string': return_string,
                'return_string_validation': return_string_validation
            }
            
            print(f"{game} - {dictionary_settings[settings[setting_idx]]} TRAINING: {return_string} VALIDATION: {return_string_validation}")
        
        # Calculate statistical results
        table_results_training = z_score_results(returns_dict, 'returns')
        print(f"{game} - TRAINING RESULTS", table_results_training)
        table_results_validation = z_score_results(returns_dict, 'returns_validation')
        print(f"{game} - VALIDATION RESULTS", table_results_validation)
        
        # Configure x-axis for all rows
        tick_positions = numpy.arange(0, 1.6, 0.5)  # 0, 0.5, 1.0, 1.5
        
        # Set x-axis labels for all plots but hide them except on bottom row
        
        # Set ticks and format them
        ax[game_idx*2].set_xticks(tick_positions)
        ax[game_idx*2 + 1].set_xticks(tick_positions)
        ax[game_idx*2].set_xticklabels([f"{x:.1f}" for x in tick_positions], fontsize=6)
        ax[game_idx*2 + 1].set_xticklabels([f"{x:.1f}" for x in tick_positions], fontsize=6)
        
        # Hide tick labels for all but bottom row
        if game_idx < len(base_games) - 1:
            plt.setp(ax[game_idx*2].get_xticklabels(), visible=False)
            plt.setp(ax[game_idx*2 + 1].get_xticklabels(), visible=False)
        else:
            plt.setp(ax[game_idx*2].get_xticklabels())
            plt.setp(ax[game_idx*2 + 1].get_xticklabels()) 
            ax[game_idx*2].tick_params(axis='x',direction="in", labelsize=6)  # Reduces the size of x-axis ticks on the first subplot
            ax[game_idx*2 + 1].tick_params(axis='x', direction="in",labelsize=6)  # Reduces the size of x-axis ticks on the second subplot
    
        ax[game_idx*2].set_xlim(0, max_step/1000000)  # Modify these limits as per your requirement
        ax[game_idx*2+1].set_xlim(0, max_step/1000000)  # Modify these limits as per your requirement

                   
        # Add titles only to the top row
        if game_idx == 0:
            ax[0].set_title('Training Rewards', fontsize=8,y=0.96, fontweight='bold')
            ax[1].set_title('Validation Rewards', fontsize=8,y=0.96, fontweight='bold')

        # Add "Average Return" label to the left side of the first column, centered vertically
    
    # Create the legend in the first row, third column of gridspec
    legend_ax = fig.add_subplot(gs[0:2, 2])  # Span first two rows for the legend
    legend_ax.axis('off')  # Hide the axis
    
    # Add the legend
    legend = legend_ax.legend(
        legend_handles[:len(settings)],  # Only use one set of handles (first game's)
        legend_labels[:len(settings)],   # Only use one set of labels
        loc='center',
        title="Models",
        frameon=True,
        fontsize='x-small',
        title_fontsize='small'
    )

    
    legend.get_frame().set_alpha(0.8)
    
    # Save and show plot
    #plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the title
    plt.tight_layout()  # Make room for the title

    plt.subplots_adjust(hspace=0.05)  # Minimal space between rows
    plt.show()
    if len(base_games)==5:
        
        fig.savefig('all_games_maxstep_AAAI_' + str(max_step) + "_avg_return.jpg", bbox_inches="tight")
    else:
        fig.savefig(base_games[0]+'_maxstep_AAAI_' + str(max_step) + "_avg_return.jpg", bbox_inches="tight")
        
    plt.close(fig)


def compare_saved_dicts(dict1_path, dict2_path):
    # Load the dictionaries from the given paths
    dict1 = torch.load(dict1_path)
    dict2 = torch.load(dict2_path)
    
    # Print out the top-level keys in both dictionaries for inspection
    print("Keys in dict1:", dict1.keys())
    print("Keys in dict2:", dict2.keys())
    
    
    def compare_dicts(val1, val2, parent_key=""):
        # Check if both values are tensors and compare them
        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            if not torch.allclose(val1, val2, atol=1e-5):  # Allowing some tolerance in numerical comparison
                print(f"Values at '{parent_key}' are DIFFERENT.")
            else:
                print(f"Values at '{parent_key}' are SAME.")

        # Check if both values are OrderedDict and recurse into their items and compare
        elif isinstance(val1, OrderedDict) and isinstance(val2, OrderedDict):
            # Compare each sub-key in the OrderedDict recursively
            for sub_key in val1:
                if sub_key not in val2:
                    print(f"Sub-key '{parent_key}.{sub_key}' is missing in the second dictionary.")
                    continue  # Skip to the next sub-key if it's missing in dict2

                sub_val1 = val1[sub_key]
                sub_val2 = val2[sub_key]
                # Recursively compare nested structures (either tensor or OrderedDict)
                compare_dicts(sub_val1, sub_val2, parent_key=f"{parent_key}.{sub_key}")
            else:
                print(f"Values at '{parent_key}' are SAME.")

        # Check if both are dicts and contain a 'state_dict' key
        elif isinstance(val1, dict) and isinstance(val2, dict):
            if 'state_dict' in val1 and 'state_dict' in val2:
                # Compare the state_dicts recursively
                compare_dicts(val1['state_dict'], val2['state_dict'], parent_key=f"{parent_key}.state_dict")
            else:
                if val1 != val2:
                    print(f"Values at '{parent_key}' are DIFFERENT.")
                else:
                    print(f"Values at '{parent_key}' are SAME.")

        # Case when values are not tensors or OrderedDicts and are directly comparable
        elif val1 != val2:
            print(f"Values at '{parent_key}' are DIFFERENT.")
        else:
            print(f"Values at '{parent_key}' are SAME.")    
                

    # Iterate over the top-level keys in dict1 and compare with dict2
    for key in dict1.keys():
        print(f"Comparing key: '{key}'")
        if key not in dict2:
            print(f"Key '{key}' is missing in the second dictionary.")
            continue
        
        val1 = dict1[key]
        val2 = dict2[key]
        
        # Start comparing each key using the recursive function
        if key not in ['optimizer_breakout','optimizer2_breakout']:
            
            '''
            if key== 'replay_buffer_breakout':
                print(val1)
                print("SPACE")
                print(val2)
            '''
                
            compare_dicts(val1, val2, parent_key=key)

        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", type=str)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--loadfile", "-l", type=str)
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--replayoff", "-r", action="store_true")
    parser.add_argument("--targetoff", "-t", action="store_true")
    parser.add_argument("--seed", "-seed", type=str)
    parser.add_argument("--ema", "-ema", type=float)
    parser.add_argument("--steps", "-steps", type=int)
    parser.add_argument("--cascade", "-cascade", type=int)
    parser.add_argument("--setting", "-setting", type=int)
    parser.add_argument("--curriculum", "-c", action="store_true")
    parser.add_argument("--curriculum_evaluation", "-ce", action="store_true")
    parser.add_argument("--checkpoint", "-point", action="store_true")
    parser.add_argument("--plot", "-pl", action="store_true")
    parser.add_argument("--plot_file", "-plfile",  type=str)
    parser.add_argument("--adapnet", "-anet", action="store_true")
    parser.add_argument("--weight1", "-w1", type=float, default=40)
    parser.add_argument("--weight2", "-w2", type=float, default=40)
    parser.add_argument("--weight3", "-w3", type=float, default=20)

    

    global NUM_FRAMES, WEIGHT1, WEIGHT2, WEIGHT3

    args = parser.parse_args()
    WEIGHT1 = float(args.weight1/100)
    WEIGHT2 = float(args.weight2/100)
    WEIGHT3 = float(args.weight3/100)

    print("WEIGHTS", WEIGHT1, WEIGHT2, WEIGHT3)
    NUM_FRAMES= args.steps
    
    if not args.plot:

        print("number of steps to run is", NUM_FRAMES)

        
        if args.verbose:
            logging.basicConfig(level=logging.INFO)

        # If there's an output specified, then use the user specified output.  Otherwise, create file in the current
        # directory with the game's name.
        if args.output:
            file_name = args.output
        else:
            file_name = os.getcwd() + "/" + args.game + '_setting' + str(args.setting) + '_steps' + str(NUM_FRAMES) +'_' + args.seed

        load_file_path = None
                
        if args.loadfile:
            load_file_path = args.loadfile
            
        #2nd order network, but no cascade model
        if args.setting== 1:
            cascade_1=1
            cascade_2=1 
            meta=True
        
        #2nd order network, and a cascade model on the 1st order network only
        elif args.setting== 2:
            cascade_1=args.cascade
            cascade_2=1
            meta=True

        #2nd order network, and a cascade model on the 2nd order network only
        elif args.setting== 3:
            cascade_1=1
            cascade_2=args.cascade
            meta=True

        #2nd order network, and a cascade model on both networks
        elif args.setting== 4:
            cascade_1=args.cascade
            cascade_2=args.cascade
            meta=True
            
        #No 2nd order network and no cascade model    
        if args.setting== 5:
            cascade_1=1
            cascade_2=1 
            meta=False
        
        #Cascade model, but no 2nd order network
        elif args.setting== 6:
            cascade_1=args.cascade
            cascade_2=1
            meta=False
            
        env = Environment(args.game)

        print('Cuda available?: ' + str(torch.cuda.is_available()))
        dqn(env, args.replayoff, args.targetoff, file_name,  meta , args.ema, cascade_1, cascade_2, args.game, args.curriculum, args.curriculum_evaluation, args.save, load_file_path, args.checkpoint, args.adapnet)
    
    else:
        plot(args.plot_file)
        
if __name__ == '__main__':
    main()