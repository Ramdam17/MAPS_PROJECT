import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check, calculate_conv_params
from onpolicy.algorithms.utils.cnn import CNNBase, Encoder
from onpolicy.algorithms.utils.modularity import SCOFF
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.rnn_meta import RNNLayer_Meta

from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.algorithms.utils.rim_cell import RIM
from absl import logging


class R_Actor_Meta(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, action_space,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(R_Actor_Meta, self).__init__()

        # new parameters
        self.drop_out = args.drop_out
        self.rnn_attention_module = args.rnn_attention_module
        self.use_bidirectional = args.use_bidirectional
        self.n_rollout = args.n_rollout_threads
        #self.hidden_size = int(args.hidden_size / 4)
        self.hidden_size = args.hidden_size

        self.layer_input=nn.Linear(args.hidden_size, self.hidden_size )
        self.layer_output=nn.Linear(self.hidden_size, args.hidden_size)
        
        
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._recurrent_N = args.recurrent_N
        self._use_version_scoff = args.use_version_scoff
        self.tpdv = dict(dtype=torch.float32, device=device)
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy

        obs_shape = get_shape_from_obs_space(obs_space)

        self.use_attention = args.use_attention
        self._attention_module = args.attention_module

        self._obs_shape = obs_shape

        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)


        self.rnn = RNNLayer_Meta(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, args.hidden_size, self._use_orthogonal, self._gain)
        self.to(device)
        self.algo = args.algorithm_name

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False, wager=0):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        
        #print("obs shape", obs.size(), rnn_states.size(), masks.size() if masks is not None else None, available_actions.size() if available_actions is not None else None)
        #rnn_states=self.layer_input(rnn_states)
        #print("obs shape", obs.size(), rnn_states.size(), masks.size() if masks is not None else None, available_actions.size() if available_actions is not None else None)

        obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)
        #actor_features=self.layer_input(actor_features)
        

        output = self.rnn(actor_features, rnn_states, masks=masks, wager=False)
        #print("output critic shape", output[0].size(), output[1].size() ) 

            
        actor_features, rnn_states = output[:2]
        #actor_features=self.layer_output(actor_features)
        #rnn_states=self.layer_output(rnn_states)
        
        if self.rnn_attention_module == "LSTM":
            c = output[-1]

        if not self.use_attention and (self._use_naive_recurrent_policy or self._use_recurrent_policy):
            rnn_states = rnn_states.permute(1, 0, 2)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        
        

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        
        
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)
        actor_features=self.layer_input(actor_features)
        rnn_states=self.layer_input(rnn_states)


        output = self.rnn(actor_features, rnn_states, masks=masks, wager=True)
                
        actor_features, rnn_states = output[:2]
        wager= output[0]
        
        #print("wager", wager.size())
        #print(output[0].size(), output[1].size() , "shape output")
        
        return wager


class R_Critic_Meta(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, cent_obs_space, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(R_Critic_Meta, self).__init__()

        # new parameters
        self.drop_out = args.drop_out
        self.rnn_attention_module = args.rnn_attention_module
        self.use_bidirectional = args.use_bidirectional
        self.n_rollout = args.n_rollout_threads
        #self.hidden_size = int(args.hidden_size / 4)
        self.hidden_size = args.hidden_size
        
        self.layer_input=nn.Linear(args.hidden_size, self.hidden_size )
        self.layer_output=nn.Linear(self.hidden_size, args.hidden_size)
        
        self._use_orthogonal = args.use_orthogonal
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)

        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy

        ## Zahra added
        self._use_version_scoff = args.use_version_scoff
        self.use_attention = args.use_attention
        self._attention_module = args.attention_module

        self._obs_shape = cent_obs_shape

        base = CNNBase if len(self._obs_shape) == 3 else MLPBase
        self.base = base(args, self._obs_shape)

 
        self.rnn = RNNLayer_Meta(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(args.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(args.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks, wager=0):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        
        
        
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        #rnn_states=self.layer_input(rnn_states)

        critic_features = self.base(cent_obs)
        #critic_features=self.layer_input(critic_features)
        

        output = self.rnn(critic_features, rnn_states, masks=masks, wager=False)
        #print("output critic shape", output[0].size(), output[1].size() ) 
        
        
        critic_features, rnn_states = output[:2]
        #critic_features=self.layer_output(critic_features)
        #rnn_states=self.layer_output(rnn_states)
        
        
        if self.rnn_attention_module == "LSTM":
            c = output[-1]

        if not self.use_attention and (self._use_naive_recurrent_policy or self._use_recurrent_policy):
            rnn_states = rnn_states.permute(1, 0, 2)

        critic_features = critic_features.unsqueeze(0)
        values = self.v_out(critic_features)

        return values, rnn_states
