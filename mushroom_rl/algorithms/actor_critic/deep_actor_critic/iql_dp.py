import torch
import numpy as np
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.rl_utils.replay_memory import ReplayMemory

from mushroom_rl.core.dataset import Dataset
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.rl_utils.parameters import Parameter, to_parameter
from mushroom_rl.utils.torch import TorchUtils
from tqdm import trange
from copy import deepcopy

# from torch.nn.functional import binary_cross_entropy_with_logits

class IQL_DP(DeepAC):
    """
    IQL Offline-to-Online RL algorithm.
    "Offline Reinforcement Learning with Implicit Q-Learning".
    Kostrikov I. et al.. 2022.
    Reference implementation: https://github.com/corl-team/CORL.
    Uses DiffusionPolicy (https://diffusion-policy.cs.columbia.edu/) as policy class.

    """
    def __init__(self, mdp_info, policy_class, policy_params,
                 actor_params, actor_optimizer, critic_params, value_func_params, value_func_optimizer,
                 batch_size, initial_replay_size, max_replay_size, tau,
                 squash_actions=False, discrete_action_dims=0, continuous_action_dims=0,
                 normalize_states=False, schedule_actor_lr=False, actor_loss_type='ddpg_plus_bc',
                 bc_weight_in_ddpg=0.25, iql_beta=1.0, iql_tau=0.7, max_clamp_adv=100.0,
                 critic_fit_params=None, actor_predict_params=None, critic_predict_params=None):
        """
        Constructor.

        Args:
            policy_class (Policy): class of the policy;
            policy_params (dict): parameters of the policy to build;
            actor_params (dict): parameters of the actor approximator to
                build;
            actor_optimizer (dict): parameters to specify the actor
                optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            value_func_params (dict): parameters of the value function approximator to build;
            value_func_optimizer (dict): parameters to specify the value function
                optimizer algorithm;
            batch_size ([int, Parameter]): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            tau ([float, Parameter]): value of coefficient for soft updates;
            squash_actions (bool, False): whether to squash the actions to [-1, 1] with tanh;
            discrete_action_dims (int, 0): number of discrete actions in the action space;
            continuous_action_dims (int, 0): number of continuous actions in the action space;
            normalize_states (bool, False): whether to normalize states;
            schedule_actor_lr (bool, False): whether to use a learning rate scheduler for the actor;
            actor_loss_type (str, 'ddpg_plus_bc' or 'awr'): type of actor loss to use. Options: 'ddpg_plus_bc', 'awr' (Advantage Weighted Regression);
            bc_weight_in_ddpg ([float, Parameter], 0.25): Weight for the BC loss in DDPG_BC;
            iql_beta ([float, Parameter], 0.25): For AWR: Inverse temperature. Small beta -> BC, big beta -> maximizing Q; when fitting on the offline dataset;
            iql_tau ([float, Parameter], 0.7): Coefficient for the asymmetric IQL loss;
            max_clamp_adv ([float, Parameter], 100.0): Maximum value considered for the advantage;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator;
            actor_predict_params (dict, None): parameters for the prediction with the
                actor approximator;
            critic_predict_params (dict, None): parameters for the prediction with the
                critic approximator.

        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params
        # self._actor_predict_params = dict() if actor_predict_params is None else actor_predict_params
        self._critic_predict_params = dict() if critic_predict_params is None else critic_predict_params

        if 'n_models' in critic_params.keys():
            assert(critic_params['n_models'] >= 2)
        else:
            critic_params['n_models'] = 2
        
        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator, **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator, **target_critic_params)

        # Add IQL value function approximator & optimizer
        # assert value_func_params['n_models'] == 1 # Single model
        self._value_func_approximator = Regressor(TorchApproximator, **value_func_params)
        value_func_network_params = self._value_func_approximator.model.network.parameters()
        self._value_func_optimizer = value_func_optimizer['class'](value_func_network_params, **value_func_optimizer['params'])

        # self._actor_approximator = Regressor(TorchApproximator, **actor_params)

        self._init_target(self._critic_approximator, self._target_critic_approximator)

        # policy = policy_class(self._actor_approximator, **policy_params)
        policy = policy_class(policy_params)

        # policy_parameters = self._actor_approximator.model.network.parameters()
        policy_parameters = policy._model.parameters()

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

        self._batch_size = to_parameter(batch_size)
        self._tau = to_parameter(tau)
        self._fit_count = 0
        self._actor_last_loss = None # Store actor loss for logging
        self._actor_last_bc_loss = None # Store BC loss for logging
        self._actor_last_q_loss = None # Store Q loss for logging
        self._value_last_loss = None # Store value loss for logging
        self._last_exp_adv = None # Store exp_adv for logging

        self._replay_memory = ReplayMemory(mdp_info, self.info, initial_replay_size, max_replay_size)

        self._squash_actions = squash_actions
        self._discrete_action_dims = discrete_action_dims
        assert discrete_action_dims == 0, 'Discrete actions not yet supported for IQL_DP'
        self._continuous_action_dims = continuous_action_dims
        self._normalize_states = normalize_states
        self._states_mean = None
        self._states_std = None

        # Optimizer deviations from mushroom_rl
        if policy_params['use_transformer'] is True:
            # create the transformer optimizers here and assign to self
            self._optimizer = policy._model.net.configure_optimizers(learning_rate=policy_params['transformer_lr_actor_net'],
                                                                weight_decay=policy_params['transformer_weight_decay_actor_net'],
                                                                betas=policy_params['transformer_betas_actor_net'])
            self._parameters = policy._model.parameters()
        # remove optimizer save attribute from super class since we will use our own method to save model and optimizer instead
        del self._save_attributes['_optimizer']

        self._schedule_actor_lr = schedule_actor_lr
        if self._schedule_actor_lr:
            max_steps = (max_replay_size * 100) // batch_size # heuristic. TODO: test
            self._actor_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, max_steps)
        else:
            self._actor_lr_scheduler = None

        self._actor_loss_type = actor_loss_type
        self._bc_weight_in_ddpg = to_parameter(bc_weight_in_ddpg)
        self._iql_beta = to_parameter(iql_beta)
        self._iql_tau = to_parameter(iql_tau)
        self._max_clamp_adv = to_parameter(max_clamp_adv)
        
        self.offline_dataset = None

        self._add_save_attr(
            _critic_fit_params='pickle',
            _critic_predict_params='pickle',
            # _actor_predict_params='pickle',
            _batch_size='mushroom',
            _tau='mushroom',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _value_func_approximator='mushroom',
            _value_func_optimizer='torch',
            # _actor_approximator='mushroom',
            _squash_actions='primitive',
            _discrete_action_dims='primitive',
            _continuous_action_dims='primitive',
            _normalize_states='primitive',
            _states_mean='primitive',
            _states_std='primitive',
            _schedule_actor_lr='primitive',
            _actor_lr_scheduler='torch',
            _actor_loss_type='primitive',
            _bc_weight_in_ddpg='mushroom',
            _iql_beta='mushroom',
            _iql_tau='mushroom',
            _max_clamp_adv='mushroom',
        )
    
    def load_dataset(self, datasets, debug=False):
        # there can be more than one dataset so loop over the list
        for dataset in datasets:
            # load & create mushroom dataset
            mushroom_dataset = Dataset.from_array(dataset['obs'], dataset['action'], dataset['reward'],
                                                    dataset['next_obs'], dataset['absorbing'], dataset['last'],
                                                    backend='torch')
            if self.offline_dataset is None:
                self.offline_dataset = mushroom_dataset
            else:
                self.offline_dataset += mushroom_dataset
        
        if self._normalize_states:
            self._compute_states_mean_std(self.offline_dataset.state)
            # update the dataset with the normalized states
            self.offline_dataset._data._states = self._norm_states(self.offline_dataset.state)
            self.offline_dataset._data._next_states = self._norm_states(self.offline_dataset.next_state)
            # move devices if needed
            self._states_mean = self._states_mean.to(TorchUtils.get_device())
            self._states_std = self._states_std.to(TorchUtils.get_device())
        
        # Action and Q chunking:
        # Make re-arranged chunked dataset to use DP-style training for policy and critic

        ## rearrange data based on obs_horizon, action_pred_horizon etc. as per diffusion policy
        n_obs_steps = self.policy._n_obs_steps
        action_horizon = self.policy._horizon
        # rearrange into episodes
        episodes = []
        episode = {'obs': torch.empty((0, self.offline_dataset.state.shape[1])),
                    'action': torch.empty((0, self.offline_dataset.action.shape[1])),
                    'reward': torch.empty((0, 1)),
                    'next_obs': torch.empty((0, self.offline_dataset.next_state.shape[1])),
                    'absorbing': torch.empty((0, 1)),
                    'last': torch.empty((0, 1))
                    }
        for i in range(len(self.offline_dataset.state)):
            episode['obs'] = torch.vstack((episode['obs'], self.offline_dataset.state[i].unsqueeze(0)))
            episode['action'] = torch.vstack((episode['action'], self.offline_dataset.action[i].unsqueeze(0)))
            episode['reward'] = torch.vstack((episode['reward'], self.offline_dataset.reward[i].unsqueeze(0)))
            episode['next_obs'] = torch.vstack((episode['next_obs'], self.offline_dataset.next_state[i].unsqueeze(0)))
            episode['absorbing'] = torch.vstack((episode['absorbing'], self.offline_dataset.absorbing[i].unsqueeze(0)))
            episode['last'] = torch.vstack((episode['last'], self.offline_dataset.last[i].unsqueeze(0)))
            if self.offline_dataset.last[i]:
                episodes.append(episode)
                episode = {'obs': torch.empty((0, self.offline_dataset.state.shape[1])),
                            'action': torch.empty((0, self.offline_dataset.action.shape[1])),
                            'reward': torch.empty((0, 1)),
                            'next_obs': torch.empty((0, self.offline_dataset.next_state.shape[1])),
                            'absorbing': torch.empty((0, 1)),
                            'last': torch.empty((0, 1))}
            if debug and i > 2000:
                print("[[Debugging so skipping time consuming data rearrangement]]")
                break
        # stack batches of size n_obs_steps for the obs and size horizon for the actions
        # Note: assumption is always that n_obs_steps < action_horizon
        # For example:
        # "observation.state": [-0.1, 0.0],
        # "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
        rearranged_dataset = {'obs': torch.empty((0, n_obs_steps, self.offline_dataset.state.shape[1])),
                                'action': torch.empty((0, action_horizon, self.offline_dataset.action.shape[1])),
                                'reward': torch.empty((0, self.offline_dataset.reward.shape[1])),
                                'next_obs': torch.empty((0, n_obs_steps, self.offline_dataset.next_state.shape[1])),
                                'absorbing': torch.empty((0, 1)),
                                'last': torch.empty((0, 1))
                                }
        for idx, episode in enumerate(episodes):
            # stack obs
            # compute obs indices
            obs_indices = torch.arange(len(episode['obs'])).unsqueeze(1) - torch.arange(n_obs_steps-1, -1, -1)
            # next obs indices are with a gap of the shift due to the action chunk
            next_obs_indices = obs_indices + action_horizon - n_obs_steps + 1
            # correct for indices out of range. Just pad with the first/last element
            obs_indices = torch.clip(obs_indices, 0, len(episode['obs'])-1)
            next_obs_indices = torch.clip(next_obs_indices, 0, len(episode['next_obs'])-1)
            obs_stack = episode['obs'][obs_indices]
            next_obs_stack = episode['next_obs'][next_obs_indices]
            rearranged_dataset['obs'] = torch.cat((rearranged_dataset['obs'], obs_stack), dim=0)
            rearranged_dataset['next_obs'] = torch.cat((rearranged_dataset['next_obs'], next_obs_stack), dim=0)
            # stack actions
            act_indices = torch.arange(len(episode['action'])).unsqueeze(1) - torch.arange(n_obs_steps-1, n_obs_steps-1-action_horizon, -1)
            # correct for indices out of range. Just pad with the first/last element
            act_indices = torch.clip(act_indices, 0, len(episode['action'])-1)
            act_stack = episode['action'][act_indices]
            rearranged_dataset['action'] = torch.cat((rearranged_dataset['action'], act_stack), dim=0)
            # accumulate rewards for the action chunk, zero rewards for out of range indices
            reward_indices = torch.arange(len(episode['reward'])).unsqueeze(1) - torch.arange(n_obs_steps-1, n_obs_steps-1-action_horizon, -1)
            episode_reward_array = torch.cat((episode['reward'], torch.zeros((1,1))), dim=0) # add zero reward at end for out of range indices
            out_of_range = (reward_indices < 0) | (reward_indices >= len(episode['reward']))
            reward_indices[out_of_range] = len(episode['reward']) # new end index will have a zero reward value
            discount_powers = self.mdp_info.gamma ** torch.arange(action_horizon).unsqueeze(0)
            discounted_rewards = episode_reward_array[reward_indices] * discount_powers.T
            acc_rewards = torch.sum(discounted_rewards, dim=1)
            rearranged_dataset['reward'] = torch.cat((rearranged_dataset['reward'], acc_rewards), dim=0)
            # rearrange last and absorbing: move them forward the same amount as we moved the next_obs since they are in sync
            absorbing_indices = torch.arange(len(episode['absorbing']))
            absorbing_indices = absorbing_indices + action_horizon - n_obs_steps + 1
            absorbing_indices = torch.clip(absorbing_indices, 0, len(episode['absorbing'])-1)
            rearranged_dataset['absorbing'] = torch.cat((rearranged_dataset['absorbing'], episode['absorbing'][absorbing_indices]), dim=0)
            last_indices = torch.arange(len(episode['last']))
            last_indices = last_indices + action_horizon - n_obs_steps + 1
            last_indices = torch.clip(last_indices, 0, len(episode['last'])-1)
            rearranged_dataset['last'] = torch.cat((rearranged_dataset['last'], episode['last'][last_indices]), dim=0)
            # TODO: make this function faster
            if debug and idx > 5:
                print("[[Debugging so skipping time consuming data rearrangement]]")
                break
        
        # move devices if needed
        rearranged_dataset['obs'] = rearranged_dataset['obs'].to(TorchUtils.get_device())
        rearranged_dataset['next_obs'] = rearranged_dataset['next_obs'].to(TorchUtils.get_device())
        rearranged_dataset['action'] = rearranged_dataset['action'].to(TorchUtils.get_device())
        rearranged_dataset['absorbing'] = rearranged_dataset['absorbing'].to(TorchUtils.get_device())
        rearranged_dataset['last'] = rearranged_dataset['last'].to(TorchUtils.get_device())
        
        TODO: roll into a single dimention for now.
        The intermediate dimension will be reintroduced when we batch before sending to DP
        chunked_offline_dataset = Dataset.from_array(rearranged_dataset
        
        # Create new replay memory object and copy over offline dataset to the replay buffer
        # Change mdp info state and action sizes to match the chunked dataset
        new_mdp_info = deepcopy(self.mdp_info)
        new new_mdp_info.observation_space = chunked_offline_dataset['obs'].shape[1]
        new new_mdp_info.action_space = chunked_offline_dataset['action'].shape[1]
        self._replay_memory = ReplayMemory(new_mdp_info, self.info, initial_size=len(chunked_offline_dataset), max_size=max(self._replay_memory._max_size, len(chunked_offline_dataset)))
        self._replay_memory.add(chunked_offline_dataset)
        
        self.offline_dataset = chunked_offline_dataset

        # else:
        #     # No chunking, just copy over offline dataset to the replay buffer
        #     self._replay_memory._initial_size = len(self.offline_dataset) # set initial size to the size of the offline dataset
        #     if self._replay_memory._max_size < len(self.offline_dataset):
        #         print('[[Warning: Offline dataset size exceeds max replay memory size. Resizing replay memory to fit dataset.]]')
        #         self._replay_memory = ReplayMemory(self.mdp_info, self.info, len(self.offline_dataset), len(self.offline_dataset))
        #     self._replay_memory.add(self.offline_dataset)
    
    def offline_fit(self, n_epochs, fit_critic=True, fit_actor=True):
        if self.offline_dataset is None:
            raise ValueError('No offline dataset loaded!. Call load_dataset() first.')
        
        # fit on the dataset (for n_epochs)
        for epoch in trange(n_epochs):
            state, action, reward, next_state, absorbing, _ = self._replay_memory.get(self._batch_size())

            # if self._normalize_states: # Assumed done at load time
            #     state_fit = self._norm_states(state)
            #     next_state_fit = self._norm_states(next_state)
            # else:
            state_fit = state
            next_state_fit = next_state

            self.iql_fit(state_fit, action, reward, next_state_fit, absorbing, fit_critic, fit_actor)
    
    def fit(self, dataset, fit_critic=True, fit_actor=True): # Online
        raise NotImplementedError('Online fitting not yet implemented for IQL_DP')
        # TODO: For online fitting, handle normalization and chunking
        # norm and chunk dataset before adding it to the replay memory
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = self._replay_memory.get(self._batch_size())
            # if self._normalize_states: # N.A.
            #     state_fit = self._norm_states(state)
            #     next_state_fit = self._norm_states(next_state)
            # else:
            state_fit = state
            next_state_fit = next_state
            
            self.iql_fit(state_fit, action, reward, next_state_fit, absorbing, fit_critic, fit_actor)

    def iql_fit(self, state, action, reward, next_state, absorbing, fit_critic=True, fit_actor=True):
        if self._actor_loss_type == 'awr':
            with torch.no_grad():
                next_v = self._value_func_approximator(next_state, **self._critic_predict_params)
                # next_v = next_v.cpu() # TODO: check if this is needed
            # Get advantage & update value function (if fit_critic)
            adv = self._get_adv_and_update_v(state, action, fit_critic)
            if fit_critic:
                # Update Q function
                self._update_q(next_v, state, action, reward, absorbing)
            if fit_actor:
                # Update actor
                self._update_actor_awr(adv, state, action)
        elif self._actor_loss_type == 'ddpg_plus_bc':
            assert fit_critic != fit_actor, 'fit_critic and fit_actor cannot be True at the same time for DDPG_BC'
            if fit_critic:
                with torch.no_grad():
                    next_v = self._value_func_approximator(next_state, **self._critic_predict_params)
                    # next_v = next_v.cpu() # TODO: check if this is needed
                # Get advantage & update value function (if fit_critic)
                adv = self._get_adv_and_update_v(state, action, fit_critic)
                # Update Q function
                self._update_q(next_v, state, action, reward, absorbing)
            elif fit_actor:
                # Update actor
                self._update_actor_ddpg_plus_bc(state, action)
        else:
            raise ValueError(f'Invalid actor loss type: {self._actor_loss_type}')

    def _get_adv_and_update_v(self, state, action, fit_critic=True):
        # Update value function
        with torch.no_grad():
            target_q = self._target_critic_approximator.predict(state, action,
                                                prediction='min', **self._critic_predict_params)

        if fit_critic:
            v = self._value_func_approximator(state, **self._critic_predict_params)
            adv = target_q - v

            v_loss = self._asymmetric_l2_loss(adv, self._iql_tau())
            self._value_func_optimizer.zero_grad()
            v_loss.backward()
            self._value_func_optimizer.step()

            self._value_last_loss = v_loss.detach().cpu().numpy() # for logging
        else:
            with torch.no_grad():
                v = self._value_func_approximator(state, **self._critic_predict_params)
                adv = target_q - v
        
        return adv
    
    def _update_q(self, next_v, state, action, reward, absorbing):
        # Compute q value
        # since we use chunking, actual gamma is gamma^action_horizon
        q = reward + (~absorbing) * (self.mdp_info.gamma ** self.policy._horizon) * next_v

        # Fit critic
        self._critic_approximator.fit(state, action, q, **self._critic_fit_params)      

        # Update target critic
        self._update_target(self._critic_approximator, self._target_critic_approximator)

    def _update_actor_awr(self, adv, state, action):
        # compute advantage weighted BC loss
        exp_adv = torch.exp(self._iql_beta() * adv.detach()).clamp(max=self._max_clamp_adv)

        # target action from data:
        act = torch.as_tensor(action, dtype=torch.float32, device=TorchUtils.get_device())
        
        # Query DP for BC loss
        TODO: roll into correct shape for DP
        batch = {'observation.state': state, 'action': act}
        bc_loss = self.policy.forward(batch, self._squash_actions)['loss']

        # Compute actor loss
        actor_loss = torch.mean(exp_adv * bc_loss)

        self._optimize_actor_parameters(actor_loss)
        
        if self._schedule_actor_lr:
            self._actor_lr_scheduler.step()

        self._actor_last_loss = actor_loss.detach().cpu().numpy() # Store actor loss for logging
        self._last_exp_adv = exp_adv.detach().mean().cpu().numpy() # Store exp_adv for logging

    def _update_actor_ddpg_plus_bc(self, state, action):
        # target action from data:
        act = torch.as_tensor(action, dtype=torch.float32, device=TorchUtils.get_device())
        
        # Query DP for predicted action and BC loss
        TODO: roll into correct shape for DP
        batch = {'observation.state': state, 'action': act}
        policy_forward_output = self.policy.forward(batch, self._squash_actions)
        act_pred = policy_forward_output['act_pred']
        bc_loss = policy_forward_output['loss']
        
        # DDPG loss
        q = self._critic_approximator(state, act_pred, **self._critic_predict_params)
        q_loss = -q

        # Total loss
        actor_loss = torch.mean(q_loss + self._bc_weight_in_ddpg() * bc_loss)

        self._optimize_actor_parameters(actor_loss)
        
        if self._schedule_actor_lr:
            self._actor_lr_scheduler.step()

        self._actor_last_loss = actor_loss.detach().cpu().numpy() # Store actor loss for logging
        self._actor_last_bc_loss = bc_loss.detach().mean().cpu().numpy() # Store BC loss for logging
        self._actor_last_q_loss = q_loss.detach().mean().cpu().numpy() # Store Q loss for logging

    def _asymmetric_l2_loss(self, u: torch.Tensor, tau: float):
        # loss is just L2 when u is positive, but (1 - tau) * L2 when u is negative.
        # Eg. When tau = 0.7, the loss is only 0.3 * L2 when u is negative.
        return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

    def _compute_states_mean_std(self, states: np.ndarray, eps: float = 1e-3):
        self._states_mean = states.mean(0)
        self._states_std = states.std(0) + eps

        # set them for the policy as well so that we use it when drawing actions
        self.policy._states_mean = self._states_mean
        self.policy._states_std = self._states_std

    def _norm_states(self, states: np.ndarray):
        if self._states_mean is None or self._states_std is None:
            raise ValueError('States mean and std not computed yet. Call _compute_states_mean_std() on the dataset first.')
        return (states - self._states_mean) / self._states_std
        
    def _post_load(self):
        self._actor_approximator = self.policy._approximator
        self._update_optimizer_parameters(self._actor_approximator.model.network.parameters())
