import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_meta import R_Actor_Meta, R_Critic_Meta

from onpolicy.utils.util import update_linear_schedule

import torch_optimizer as optim2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.optimizer=args.optimizer



        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_meta = R_Actor_Meta(args, self.obs_space, self.act_space, self.device)
        self.critic_meta = R_Critic_Meta(args, self.share_obs_space, self.device)
        

        total_parameters_normal= count_parameters(self.actor) + count_parameters(self.critic)
        total_parameters_meta= count_parameters(self.actor_meta) + count_parameters(self.critic_meta)
        print("parameters normal ", total_parameters_normal, " |||  parameters meta ", total_parameters_meta)



        if self.optimizer == 'ADAM':
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay,amsgrad=False)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                     lr=self.critic_lr,
                                                     eps=self.opti_eps,
                                                     weight_decay=self.weight_decay,amsgrad=False)
            
            self.actor_meta_optimizer = torch.optim.Adam(self.actor_meta.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay,amsgrad=False)
            self.critic_meta_optimizer = torch.optim.Adam(self.critic_meta.parameters(),
                                                        lr=self.critic_lr,
                                                        eps=self.opti_eps,
                                                        weight_decay=self.weight_decay,amsgrad=False)
        elif self.optimizer== 'ADAMAX':
            self.actor_optimizer = torch.optim.Adamax(self.actor.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.Adamax(self.critic.parameters(),
                                                     lr=self.critic_lr,
                                                     eps=self.opti_eps,
                                                     weight_decay=self.weight_decay)
            self.actor_meta_optimizer = torch.optim.Adamax(self.actor_meta.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)
            self.critic_meta_optimizer = torch.optim.Adamax(self.critic_meta.parameters(),
                                                        lr=self.critic_lr,
                                                        eps=self.opti_eps,
                                                        weight_decay=self.weight_decay)
        elif self.optimizer == 'RANGERVA':
            self.actor_optimizer = optim2.RangerVA(self.actor.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)
            self.critic_optimizer = optim2.RangerVA(self.critic.parameters(),
                                                     lr=self.critic_lr,
                                                     eps=self.opti_eps,
                                                     weight_decay=self.weight_decay)
            self.actor_meta_optimizer = optim2.RangerVA(self.actor_meta.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)
            self.critic_meta_optimizer = optim2.RangerVA(self.critic_meta.parameters(),
                                                        lr=self.critic_lr,
                                                        eps=self.opti_eps,
                                                        weight_decay=self.weight_decay)
        
            
        elif self.optimizer == 'AMS':
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay, amsgrad=True)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                     lr=self.critic_lr,
                                                     eps=self.opti_eps,
                                                     weight_decay=self.weight_decay, amsgrad=True)
            self.actor_meta_optimizer = torch.optim.Adam(self.actor_meta.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay,amsgrad=True)
            self.critic_meta_optimizer = torch.optim.Adam(self.critic_meta.parameters(),    
                                                        lr=self.critic_lr,
                                                        eps=self.opti_eps,
                                                        weight_decay=self.weight_decay,amsgrad=True)
            
        elif self.optimizer == 'ADAMW':
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay,amsgrad=False)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(),
                                                     lr=self.critic_lr,
                                                     eps=self.opti_eps,
                                                     weight_decay=self.weight_decay,amsgrad=False)
            self.actor_meta_optimizer = torch.optim.AdamW(self.actor_meta.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay,amsgrad=False)
            self.critic_meta_optimizer = torch.optim.AdamW(self.critic_meta.parameters(),
                                                        lr=self.critic_lr,
                                                        eps=self.opti_eps,
                                                        weight_decay=self.weight_decay,amsgrad=False)
            
        elif self.optimizer == 'AMSW':
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay, amsgrad=True)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(),
                                                     lr=self.critic_lr,
                                                     eps=self.opti_eps,
                                                     weight_decay=self.weight_decay, amsgrad=True)
            self.actor_meta_optimizer = torch.optim.AdamW(self.actor_meta.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay,amsgrad=True)
            self.critic_meta_optimizer = torch.optim.AdamW(self.critic_meta.parameters(),
                                                        lr=self.critic_lr,
                                                        eps=self.opti_eps,
                                                        weight_decay=self.weight_decay,amsgrad=True)
            
        elif self.optimizer == 'RMS':
            self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(),
                                                      lr=self.lr, eps=self.opti_eps,
                                                      weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(),
                                                       lr=self.critic_lr,
                                                       eps=self.opti_eps,
                                                       weight_decay=self.weight_decay)  
            self.actor_meta_optimizer = torch.optim.RMSprop(self.actor_meta.parameters(),
                                                        lr=self.lr, eps=self.opti_eps,
                                                        weight_decay=self.weight_decay)
            self.critic_meta_optimizer = torch.optim.RMSprop(self.critic_meta.parameters(),
                                                         lr=self.critic_lr,
                                                         eps=self.opti_eps,
                                                         weight_decay=self.weight_decay)
            
        elif self.optimizer == 'POO':
            self.actor_optimizer = optim2.Shampoo(self.actor.parameters(),
                                                      lr=self.lr, epsilon=self.opti_eps,
                                                      weight_decay=self.weight_decay)
            self.critic_optimizer = optim2.Shampoo(self.critic.parameters(),
                                                       lr=self.critic_lr,
                                                       epsilon=self.opti_eps,
                                                       weight_decay=self.weight_decay)
            self.actor_meta_optimizer = optim2.Shampoo(self.actor_meta.parameters(),
                                                        lr=self.lr, epsilon=self.opti_eps,
                                                        weight_decay=self.weight_decay)
            self.critic_meta_optimizer = optim2.Shampoo(self.critic_meta.parameters(), 
                                                            lr=self.critic_lr,
                                                            epsilon=self.opti_eps,
                                                            weight_decay=self.weight_decay)
            
        elif self.optimizer == 'SWT':
            self.actor_optimizer = optim2.SWATS(self.actor.parameters(),
                                                      lr=self.lr, eps=self.opti_eps,
                                                      weight_decay=self.weight_decay)   
            self.critic_optimizer = optim2.SWATS(self.critic.parameters(),
                                                       lr=self.critic_lr,
                                                       eps=self.opti_eps,
                                                       weight_decay=self.weight_decay)
            self.actor_meta_optimizer = optim2.SWATS(self.actor_meta.parameters(),
                                                        lr=self.lr, eps=self.opti_eps,
                                                        weight_decay=self.weight_decay)
            self.critic_meta_optimizer = optim2.SWATS(self.critic_meta.parameters(),
                                                        lr=self.critic_lr,
                                                        eps=self.opti_eps,
                                                        weight_decay=self.weight_decay)
            
        elif self.optimizer == 'SGD':
            self.actor_optimizer = torch.optim.SGD(self.actor.parameters(),
                                                      lr=self.lr, weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.SGD(self.critic.parameters(),
                                                       lr=self.critic_lr,
                                                       weight_decay=self.weight_decay)
            self.actor_meta_optimizer = torch.optim.SGD(self.actor_meta.parameters(),
                                                        lr=self.lr, weight_decay=self.weight_decay)
            self.critic_meta_optimizer = torch.optim.SGD(self.critic_meta.parameters(),
                                                        lr=self.critic_lr,
                                                        weight_decay=self.weight_decay)
            
        
        
            

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    # def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
    #                 deterministic=False):
    #     """
    #     Compute actions and value function predictions for the given inputs.
    #     :param cent_obs (np.ndarray): centralized input to the critic.
    #     :param obs (np.ndarray): local agent inputs to the actor.
    #     :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
    #     :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
    #     :param masks: (np.ndarray) denotes points at which RNN states should be reset.
    #     :param available_actions: (np.ndarray) denotes which actions are available to agent
    #                               (if None, all actions available)
    #     :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

    #     :return values: (torch.Tensor) value function predictions.
    #     :return actions: (torch.Tensor) actions to take.
    #     :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
    #     :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
    #     :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
    #     """
    #     actions, action_log_probs, rnn_states_actor = self.actor(obs,
    #                                                              rnn_states_actor,
    #                                                              masks,
    #                                                              available_actions,
    #                                                              deterministic
    #                                                              )

    #     values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
    #     return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic



    def lr_decay_meta(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_meta_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_meta_optimizer, episode, episodes, self.critic_lr)
        
        
    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_cells_actor, rnn_states_critic, rnn_cells_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        param cent_obs (np.ndarray): centralized input to the critic.
         :param obs (np.ndarray): local agent inputs to the actor.
         :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
         :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
         :param masks: (np.ndarray) denotes points at which RNN states should be reset.
         :param available_actions: (np.ndarray) denotes which actions are available to agent
                                   (if None, all actions available)
         :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

         :return values: (torch.Tensor) value function predictions.
         :return actions: (torch.Tensor) actions to take.
         :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
         :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
         :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        # Convert numpy arrays to PyTorch tensors and move them to the correct device
        cent_obs = torch.tensor(cent_obs).to(self.device)
        obs = torch.tensor(obs).to(self.device)
        rnn_states_actor = torch.tensor(rnn_states_actor).to(self.device)
        rnn_cells_actor = torch.tensor(rnn_cells_actor).to(self.device)
        rnn_states_critic = torch.tensor(rnn_states_critic).to(self.device)
        rnn_cells_critic = torch.tensor(rnn_cells_critic).to(self.device)
        masks = torch.tensor(masks).to(self.device)

        if available_actions is not None:
            available_actions = torch.tensor(available_actions).to(self.device)

        # Now call the actor and critic with tensors on the correct device
        actions, action_log_probs, rnn_states_actor, rnn_cells_actor = self.actor(obs,
                                                                                  rnn_states_actor,
                                                                                  rnn_cells_actor,
                                                                                  masks,
                                                                                  available_actions,
                                                                                  deterministic
                                                                                  )

        values, rnn_states_critic, rnn_cells_critic = self.critic(cent_obs, rnn_states_critic, rnn_cells_critic, masks)
        
        return values, actions, action_log_probs, rnn_states_actor, rnn_cells_actor, rnn_states_critic, rnn_cells_critic
    

    def meta(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, wager_objective, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        [...]
        """
        # Convert numpy arrays to PyTorch tensors and move them to the correct device
        cent_obs = torch.tensor(cent_obs).to(self.device)
        obs = torch.tensor(obs).to(self.device)
        rnn_states_actor = torch.tensor(rnn_states_actor).to(self.device)
        rnn_states_critic = torch.tensor(rnn_states_critic).to(self.device)
        masks = torch.tensor(masks).to(self.device)

        if available_actions is not None:
            available_actions = torch.tensor(available_actions).to(self.device)

        # Now call the actor and critic with tensors on the correct device
        actions, action_log_probs, rnn_states_actor = self.actor_meta(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic , wager_objective
                                                                 )

        values, rnn_states_critic = self.critic_meta(cent_obs, rnn_states_critic, masks , wager_objective)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
    
    def get_values(self, cent_obs, rnn_states_critic, rnn_cells_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _, _ = self.critic(cent_obs, rnn_states_critic, rnn_cells_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_cells_actor, rnn_states_critic, rnn_cells_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     rnn_cells_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _, _ = self.critic(cent_obs, rnn_states_critic, rnn_cells_critic, masks)
        return values, action_log_probs, dist_entropy



    def evaluate_actions_meta(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        
        rnn_states_actor_input= torch.tensor(rnn_states_actor).to(self.device)
        
        #values, actions, action_log_probs, rnn_states_actor_output, rnn_states_critics = self.get_actions(cent_obs, obs, rnn_states_actor, rnn_states_critic, masks )
        
        #rnn_states_actor_output = rnn_states_actor_output.permute(1, 0, 2)
        
        
        #rnn_states_actor_meta = rnn_states_actor_input - rnn_states_actor_output
        #print("shape of rnn_states_actor_input", rnn_states_actor_input.shape , "shape of rnn_states_actor_output", rnn_states_actor_output.shape, "shape of rnn_states_actor_meta", rnn_states_actor_meta.shape)
        
        wager = self.actor_meta.evaluate_actions(obs,
                                                rnn_states_actor_input,
                                                action,
                                                masks,
                                                available_actions,
                                                active_masks)

        return wager
    
    
    
    
    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
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
        output_cascade1=None
        obs = check(obs).to(**self.tpdv)


        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)
        
        if self._attention_module == "SCOFF":
            output = self.rnn(actor_features, rnn_states)
            actor_features, rnn_states = output[:2]

        else:
            for j in range(self.cascade_one):
                actor_features, rnn_states, output_cascade1 = self.rnn(actor_features, rnn_states, masks, output_cascade1 , self.cascade_rate_one)
                            
        if not self.use_attention and (self._use_naive_recurrent_policy or self._use_recurrent_policy):
            rnn_states = rnn_states.permute(1, 0, 2)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def act(self, obs, rnn_states_actor, rnn_cells_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor, rnn_cells_actor = self.actor(obs, rnn_states_actor, rnn_cells_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor, rnn_cells_actor