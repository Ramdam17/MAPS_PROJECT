import torch
import torch.nn as nn
import torch.nn.functional as f
import time

import numpy, argparse, logging, os

from collections import namedtuple
from minatar import Environment


#####################################################################################################################
# Constants
#
#####################################################################################################################
ALPHA = 0.00048828125
LAMBDA = 0.8
GAMMA = 0.99
BETA = 0.01
GAMMA_RMS = 0.999
EPS_RMS = 0.0001
MIN_DENOM = 0.0001

dSiLU = lambda x: torch.sigmoid(x)*(1+x*(1-torch.sigmoid(x)))
SiLU = lambda x: x*torch.sigmoid(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#####################################################################################################################
# ACNetwork
#
# Setup the AC-network with one hidden 2D conv with variable number of input channels. We use 16 filters, a quarter of
# the original DQN paper of 64. One hidden fully connected linear layer with a quarter of the original DQN paper of
# 512 rectified units. Finally, we use one output layer which is a fully connected softmax layer with a single output 
# for each valid action for the policy network, and another output which is a fully connected linear layer, with a
# single output for the state value. 
#
#####################################################################################################################
class ACNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):

        super(ACNetwork, self).__init__()
        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        # Output layer:
        self.policy = nn.Linear(in_features=128, out_features=num_actions)
        self.value = nn.Linear(in_features=128, out_features=1)

    # As per implementation instructions, the forward function should be overwritten by all subclasses
    def forward(self, x):
        # Output from the first conv with sigmoid linear activation
        x = SiLU(self.conv(x))

        # Output from the final hidden layer with derivative of sigmoid linear activation
        x = dSiLU(self.fc_hidden(x.view(x.size(0), -1)))

        # Return policy and value outputs
        return f.softmax(self.policy(x), dim=1), self.value(x)

transition = namedtuple('transition', 'state, last_state, action, reward, is_terminal')


#####################################################################################################################
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
#####################################################################################################################
def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


#####################################################################################################################
# world_dynamics
#
# Responsible for world dynamics. It generates next state and reward after taking an action according to the behavior
# policy. The behavior policy is specified by the policy network output. Reward should be casted to float, otherwise
# it is LongTensor, which is used for indexing.
#
# Inputs:
#   s: current state
#   env: environment of the game
#   network: combined policy and value netork, an instance of ACNetwork
#
# Output: next state, action, reward, is_terminated
#
#####################################################################################################################
def world_dynamics(s, env, network):
    # network(s)[0] specifies the policy network, which we use to draw an action according to a multinomial 
    # distribution over axis 1, (axis 0 iterates over samples, and is unused in this case. torch._no_grad() 
    # avoids tracking history in autograd.
    with torch.no_grad():
        action = torch.multinomial(network(s)[0],1)[0]

    # Act according to the action and observe the transition and reward
    reward, terminated = env.act(action)

    # Obtain s_prime
    s_prime = get_state(env.state())

    return s_prime, action, torch.tensor([[reward]], device=device).float(), torch.tensor([[terminated]], device=device)

#####################################################################################################################
# train
#
# This is where learning happens. More specifically, this function updates the weights of the policy/value network.
#
# Inputs:
#   sample: a single transition
#   traces: an instance of QNetwork
#   grads: a list of tensors, one for each network parameter. Used as temporary storage of computed gradient
#   network: an instance of ACNetwork, to be trained
#   alpha: learning rate for actor-critic update
#
#####################################################################################################################
def train(sample, traces, grads, MSGs, network, alpha, time_step):
    # states, next_states: (1, in_channel, 10, 10) - inline with pytorch NCHW format
    # actions, rewards, is_terminal: (1, 1)
    last_state = sample.last_state
    state = sample.state
    action = sample.action
    reward = sample.reward
    is_terminal = sample.is_terminal

    pi, V_curr = network(state)

    # Compute the targets
    trace_potential = V_curr+0.5*torch.log(pi[0,action]+MIN_DENOM)
    entropy = -torch.sum(torch.log(pi+MIN_DENOM)*pi)

    network.zero_grad()
    trace_potential.backward(retain_graph=True)

    with torch.no_grad():
        for param, grad in zip(network.parameters(), grads):
            grad.data.copy_(param.grad)

    # Update parameters except for on the first observation
    if(last_state is not None):
        network.zero_grad()
        entropy.backward()
        with torch.no_grad():
            V_last = network(last_state)[1]
            delta = GAMMA*(0 if is_terminal else V_curr)+reward-V_last

            # Update uses RMSProp with initialization debiasing
            for param, trace, MSG in zip(network.parameters(), traces, MSGs):
                # Check if param.grad is None and use 0 in that case
                param_grad = 0 if param.grad is None else param.grad
                grad = trace*delta[0]+BETA*param_grad
                MSG.copy_(GAMMA_RMS*MSG+(1-GAMMA_RMS)*grad*grad)
                param.copy_(param+alpha*grad/(torch.sqrt(MSG/(1-GAMMA_RMS**(time_step+1))+EPS_RMS)))

    # Always update trace
    with torch.no_grad():
        for grad, trace in zip(grads, traces):
            trace.copy_(LAMBDA*GAMMA*trace+grad)

#####################################################################################################################
# validation
#
# Evaluates the agent's performance on a set of validation episodes.
#
# Inputs:
#   env: environment of the game
#   network: an instance of ACNetwork to evaluate
#   num_episodes: number of episodes to run for validation
#
# Output: average return across all validation episodes
#
#####################################################################################################################
def validation(env, network, num_episodes):
    # Set network to evaluation mode
    network.eval()
    
    validation_returns = []
    
    for _ in range(num_episodes):
        # Initialize environment and state
        env.reset()
        s = get_state(env.state())
        G = 0.0
        is_terminated = False
        
        # Run until episode terminates
        while not is_terminated:
            # Get action based on policy
            with torch.no_grad():
                action = torch.multinomial(network(s)[0], 1)[0]
            
            # Execute action
            reward, is_terminated = env.act(action)
            
            # Update return and state
            G += reward
            if not is_terminated:
                s = get_state(env.state())
        
        # Store episode return
        validation_returns.append(G)
    
    # Set network back to training mode
    network.train()
    
    # Return average validation return
    return numpy.mean(validation_returns), numpy.std(validation_returns)

#####################################################################################################################
# AC_lambda
#
# Online actor-critic algorithm with eligibiltiy traces but with no experience replay.
#
# Inputs:
#   env: environment of the game
#   replay_off: disable the replay buffer and train on each state transition
#   target_off: disable target network
#   output_file_name: directory and file name prefix to output data and network weights, file saved as 
#       <output_file_name>_data_and_weights
#   store_intermediate_result: a boolean, if set to true will store checkpoint data every 1000 episodes
#       to a file named <output_file_name>_checkpoint
#   load_path: directory of the file plus the file name of the saved model
#   alpha: step-size for use in actor-critic update
#
#####################################################################################################################
def AC_lambda(env, output_file_name, store_intermediate_result=False, load_path=None, alpha=ALPHA):

    # Get in_channels and num_actions
    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()

    # Instantiate networks, optimizer, loss and buffer
    network = ACNetwork(in_channels, num_actions).to(device)

    # Eligibility traces are stored here
    traces = [torch.zeros(x.size(), dtype=torch.float32, device=device) for x in network.parameters()]

    # Space allocated to store gradients used in training
    grads = [torch.zeros(x.size(), dtype=torch.float32, device=device) for x in network.parameters()]

    # Running average of mean squared gradient for use in RMSProp
    MSG = [torch.zeros(x.size(), dtype=torch.float32, device=device) for x in network.parameters()]

    # Set initial values
    e = 0
    t = 0
    avg_return = 0.0
    returns = []
    frame_stamps = []
    
    # New validation metrics storage
    validation_returns = []
    validation_frames = []
    validation_episodes = []
    validation_returns_std = []

    # Load model and optimizer if load_path is not None
    if load_path is not None and isinstance(load_path, str):
        checkpoint = torch.load(load_path)
        network.load_state_dict(checkpoint['network_state_dict'])
        e = checkpoint['episode']
        t = checkpoint['frame']
        avg_return = checkpoint['avg_return']
        returns = checkpoint['returns']
        frame_stamps = checkpoint['frame_stamps']
        
        # Load validation metrics if available
        if 'validation_returns' in checkpoint:
            validation_returns = checkpoint['validation_returns']
            validation_frames = checkpoint['validation_frames']
            validation_episodes = checkpoint['validation_episodes']
            validation_returns_std = checkpoint['validation_returns_std']

        # Set to training mode
        network.train()

    # Start the simulation
    # Train for a number of frames
    t_start = time.time()
    while t < NUM_FRAMES:
        # Initialize the return for every episode (we should see this eventually increase)
        G = 0.0

        # Initialize the environment and start state
        env.reset()
        s = get_state(env.state())
        is_terminated = False
        s_last = None
        r_last = None
        term_last = None
        while(not is_terminated) and t < NUM_FRAMES:
            # Generate data
            s_prime, action, reward, is_terminated = world_dynamics(s, env, network)

            sample = transition(s, s_last, action, r_last, term_last)

            train(sample, traces, grads, MSG, network, alpha, t)

            G += reward.item()

            t += 1

            # Continue the process
            s_last = s
            r_last = reward
            term_last = is_terminated
            s = s_prime

        # Increment the episodes
        e += 1
        sample = transition(s, s_last, action, r_last, term_last)
        train(sample, traces, grads, MSG, network, alpha, t)

        # Clear elligibility traces after each episode
        for trace in traces:
            trace.zero_()

        # Save the return for each episode
        returns.append(G)
        frame_stamps.append(t)

        # Logging exponentiated return only when verbose is turned on and only at 100 episode intervals
        avg_return = 0.99 * avg_return + 0.01 * G
        
        # Run validation every 100 episodes
        if e % CHECKPOINT_ITERATION == 0:
            # Run validation episodes
            val_reward, val_reward_std = validation(env, network, VALIDATION_ITERATIONS)
            
            # Store validation metrics
            validation_returns.append(val_reward)
            validation_returns_std.append(val_reward_std)
            validation_frames.append(t)
            validation_episodes.append(e)
            
            # Log training and validation metrics
            print("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                     str(numpy.around(avg_return, 2)) + " | Val return: " +
                     str(numpy.around(val_reward, 2)) +
                     " | Frame: " + str(t) + " | Time per frame: " +
                     str((time.time()-t_start)/t))
            
            logging.info("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                     str(numpy.around(avg_return, 2)) + " | Val return: " +
                     str(numpy.around(val_reward, 2))  +
                     " | Frame: " + str(t) + " | Time per frame: " +
                     str((time.time()-t_start)/t))
            
            # Save model data and other intermediate data if specified
            if store_intermediate_result:
                torch.save({
                    'episode': e,
                    'frame': t,
                    'network_state_dict': network.state_dict(),
                    'avg_return': avg_return,
                    'returns': returns,
                    'frame_stamps': frame_stamps,
                    'validation_returns': validation_returns,
                    'validation_returns_std': validation_returns_std,
                    'validation_frames': validation_frames,
                    'validation_episodes': validation_episodes,
                }, output_file_name + "_checkpoint")
        
        elif e % CHECKPOINT_ITERATION == 0:
            # Log regular training metrics (non-validation episodes)
            logging.info("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                     str(numpy.around(avg_return, 2)) + " | Frame: " + str(t) + " | Time per frame: " +
                     str((time.time()-t_start)/t))

    # Print final logging info
    logging.info("Avg return: " + str(numpy.around(avg_return, 2)) + " | Time per frame: " +
                 str((time.time()-t_start)/t))
        
    # Write data to file
    torch.save({
        'returns': returns,
        'unique_frames': frame_stamps,
        'network_state_dict': network.state_dict(),
        'returns_validation': validation_returns,
        'validation_returns_std': validation_returns_std,
        'unique_frames_validation': validation_frames,
        'validation_episodes': validation_episodes,
    }, output_file_name + "_data_and_weights")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", type=str)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--loadfile", "-l", type=str)
    parser.add_argument("--alpha", "-a", type=float, default=ALPHA)
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--steps", "-steps", type=int)

    args = parser.parse_args()

    global NUM_FRAMES, VALIDATION_ITERATIONS, CHECKPOINT_ITERATION
    
    if args.game== 'freeway':
        CHECKPOINT_ITERATION = 15
        VALIDATION_ITERATIONS=2
    else:
        CHECKPOINT_ITERATION=600
        VALIDATION_ITERATIONS=2

    NUM_FRAMES = args.steps
    print("number of steps to run is", NUM_FRAMES)
    print("iterations before a checkpoint:", CHECKPOINT_ITERATION)
    print("validation iterations per checkpoint:", VALIDATION_ITERATIONS)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # If there's an output specified, then use the user specified output. Otherwise, create file in the current
    # directory with the game's name.
    if args.output:
        file_name = args.output
    else:
        file_name = os.getcwd() + "/" + args.game

    load_file_path = None
    if args.loadfile:
        load_file_path = args.loadfile

    env = Environment(args.game)

    print('Cuda available?:'+str(torch.cuda.is_available()))
    AC_lambda(env, file_name, args.save, load_file_path, alpha=args.alpha)


if __name__ == '__main__':
    main()