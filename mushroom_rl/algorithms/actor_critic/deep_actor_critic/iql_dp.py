import torch
import numpy as np
import sys
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.rl_utils.replay_memory import ReplayMemory

from mushroom_rl.core.dataset import Dataset
from mushroom_rl.rl_utils import spaces
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.rl_utils.parameters import Parameter, to_parameter
from mushroom_rl.utils.torch import TorchUtils
from tqdm import tqdm, trange
from copy import deepcopy

# Helper function to check if output is redirected (for disabling tqdm in log files)
def should_disable_tqdm():
    """Returns True if stderr is not a TTY (i.e., output is redirected to a file)"""
    return not sys.stderr.isatty()

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
        
        # Check if critic uses transformer and reconfigure optimizer accordingly
        # Check the first model to see if it's a transformer (all models in ensemble should be same type)
        if len(self._critic_approximator) > 0 and hasattr(self._critic_approximator[0].network, 'configure_optimizers'):
            # Use transformer optimizer configuration (similar to actor)
            # Get transformer parameters from policy_params if available, otherwise use defaults
            if isinstance(policy_params, dict):
                transformer_lr = policy_params.get('transformer_lr_critic_net', policy_params.get('transformer_lr_actor_net', 1e-4))
                transformer_weight_decay = policy_params.get('transformer_weight_decay_critic_net', policy_params.get('transformer_weight_decay_actor_net', 1e-3))
                transformer_betas = policy_params.get('transformer_betas_critic_net', policy_params.get('transformer_betas_actor_net', (0.9, 0.95)))
            else:
                transformer_lr = getattr(policy_params, 'transformer_lr_critic_net', getattr(policy_params, 'transformer_lr_actor_net', 1e-4))
                transformer_weight_decay = getattr(policy_params, 'transformer_weight_decay_critic_net', getattr(policy_params, 'transformer_weight_decay_actor_net', 1e-3))
                transformer_betas = getattr(policy_params, 'transformer_betas_critic_net', getattr(policy_params, 'transformer_betas_actor_net', (0.9, 0.95)))
            
            # Replace the optimizer in each critic model
            for i in range(len(self._critic_approximator)):
                self._critic_approximator[i]._optimizer = self._critic_approximator[i].network.configure_optimizers(
                    learning_rate=transformer_lr,
                    weight_decay=transformer_weight_decay,
                    betas=transformer_betas
                )
                # Remove optimizer from save attributes (too large, will be recreated on load)
                if '_optimizer' in self._critic_approximator[i]._save_attributes:
                    del self._critic_approximator[i]._save_attributes['_optimizer']
            
            # Also remove optimizers from target critic approximators (they don't need optimizers, but remove from save)
            for i in range(len(self._target_critic_approximator)):
                if '_optimizer' in self._target_critic_approximator[i]._save_attributes:
                    del self._target_critic_approximator[i]._save_attributes['_optimizer']

        # Add IQL value function approximator & optimizer
        # assert value_func_params['n_models'] == 1 # Single model
        self._value_func_approximator = Regressor(TorchApproximator, **value_func_params)
        
        # Check if value function uses transformer and configure optimizer accordingly
        # Access the model through the regressor (handles both single model and ensemble)
        value_func_model = self._value_func_approximator[0] if len(self._value_func_approximator) > 0 else self._value_func_approximator.model
        if hasattr(value_func_model.network, 'configure_optimizers'):
            # Use transformer optimizer configuration (similar to actor)
            # Get transformer parameters from policy_params if available, otherwise use defaults
            if isinstance(policy_params, dict):
                transformer_lr = policy_params.get('transformer_lr_critic_net', policy_params.get('transformer_lr_actor_net', 1e-4))
                transformer_weight_decay = policy_params.get('transformer_weight_decay_critic_net', policy_params.get('transformer_weight_decay_actor_net', 1e-3))
                transformer_betas = policy_params.get('transformer_betas_critic_net', policy_params.get('transformer_betas_actor_net', (0.9, 0.95)))
            else:
                transformer_lr = getattr(policy_params, 'transformer_lr_critic_net', getattr(policy_params, 'transformer_lr_actor_net', 1e-4))
                transformer_weight_decay = getattr(policy_params, 'transformer_weight_decay_critic_net', getattr(policy_params, 'transformer_weight_decay_actor_net', 1e-3))
                transformer_betas = getattr(policy_params, 'transformer_betas_critic_net', getattr(policy_params, 'transformer_betas_actor_net', (0.9, 0.95)))
            
            self._value_func_optimizer = value_func_model.network.configure_optimizers(
                learning_rate=transformer_lr,
                weight_decay=transformer_weight_decay,
                betas=transformer_betas
            )
        else:
            # Use standard optimizer
            if value_func_optimizer is None:
                raise ValueError("value_func_optimizer cannot be None when network does not have configure_optimizers method")
            value_func_network_params = value_func_model.network.parameters()
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
        self._q_last_loss = None # Store critic Q loss for logging
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
        self.optimal_dataset = None  # Store optimal dataset for critic error computation
        self.actor_dataset = None  # Store actor dataset for actor training and critic error computation
        self.offline_episode_starts = None  # Episode boundaries for offline dataset
        self.offline_episode_ends = None
        self.optimal_episode_starts = None  # Episode boundaries for optimal dataset
        self.optimal_episode_ends = None
        self.actor_episode_starts = None  # Episode boundaries for actor dataset
        self.actor_episode_ends = None

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
            # _value_func_optimizer='torch',
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
    
    def _get_episode_boundaries(self, dataset):
        """
        Find episode boundaries in a dataset using the 'last' flag.
        
        Args:
            dataset: Dataset object with 'last' attribute indicating episode ends
        
        Returns:
            tuple: (episode_starts, episode_ends) - lists of start and end indices for each episode
        """
        # Convert last to boolean if needed (handle both bool and numeric types)
        last_array = dataset.last.squeeze() if dataset.last.dim() > 1 else dataset.last
        if last_array.dtype != torch.bool:
            last_array = last_array.bool()
        # Find all episode end indices (where last[i] == True)
        episode_end_indices = torch.where(last_array)[0].cpu().numpy()
        # Compute episode start and end indices
        if len(episode_end_indices) > 0:
            episode_starts = [0] + (episode_end_indices[:-1] + 1).tolist()
            episode_ends = (episode_end_indices + 1).tolist()
            # Handle case where dataset doesn't end with last=True
            # Include remaining samples as the last episode
            if episode_ends[-1] < len(dataset.state):
                episode_starts.append(episode_ends[-1])
                episode_ends.append(len(dataset.state))
        else:
            # No episode boundaries found - treat entire dataset as one episode
            episode_starts = [0]
            episode_ends = [len(dataset.state)]
        
        return episode_starts, episode_ends
    
    def _load_and_chunk_dataset(self, datasets, compute_mean_std=True, debug=False):
        """
        Shared helper function to load datasets and perform chunking.
        Handles loading, normalization, episode rearrangement, and chunking.
        
        Args:
            datasets: list of dictionaries with keys: obs, action, reward, next_obs, absorbing, last
            compute_mean_std: whether to compute mean/std for normalization (only used if normalize_states=True)
            debug: if True, skip time-consuming data rearrangement for debugging
        
        Returns:
            chunked_dataset: Dataset object with chunked data ready for training
            episode_starts: list of start indices for each episode
            episode_ends: list of end indices for each episode
        """
        # there can be more than one dataset so loop over the list
        mushroom_dataset = None
        for dataset in datasets:
            # load & create mushroom dataset
            dataset_obj = Dataset.from_array(
                dataset['obs'].astype(np.float32),
                dataset['action'].astype(np.float32),
                dataset['reward'].astype(np.float32),
                dataset['next_obs'].astype(np.float32),
                dataset['absorbing'].astype(np.float32),
                dataset['last'].astype(np.float32),
                backend='torch'
            )
            if mushroom_dataset is None:
                mushroom_dataset = dataset_obj
            else:
                mushroom_dataset += dataset_obj
        
        if self._normalize_states:
            if compute_mean_std:
                self._compute_states_mean_std(mushroom_dataset.state)
            # update the dataset with the normalized states
            mushroom_dataset._data._states = self._norm_states(mushroom_dataset.state)
            mushroom_dataset._data._next_states = self._norm_states(mushroom_dataset.next_state)
        
        # Action and Q chunking:
        # Make re-arranged chunked dataset to use DP-style training for policy and critic

        ## rearrange data based on obs_horizon, action_pred_horizon etc. as per diffusion policy
        n_obs_steps = self.policy._n_obs_steps
        action_horizon = self.policy._horizon
        # Find episode boundaries using the 'last' array
        episode_starts, episode_ends = self._get_episode_boundaries(mushroom_dataset)
        # Limit episodes for debugging
        if debug:
            max_episodes = 500
            episode_starts = episode_starts[:max_episodes]
            episode_ends = episode_ends[:max_episodes]
            print(f"[[Debugging so processing only {len(episode_starts)} episodes]]")
        
        # Create episodes by slicing the original arrays directly
        episodes = []
        for start_idx, end_idx in zip(episode_starts, episode_ends):
            episode = {
                'obs': mushroom_dataset.state[start_idx:end_idx],
                'action': mushroom_dataset.action[start_idx:end_idx],
                'reward': mushroom_dataset.reward[start_idx:end_idx],
                'next_obs': mushroom_dataset.next_state[start_idx:end_idx],
                'absorbing': mushroom_dataset.absorbing[start_idx:end_idx],
                'last': mushroom_dataset.last[start_idx:end_idx]
            }
            episodes.append(episode)
        # stack batches of size n_obs_steps for the obs and size horizon for the actions
        # Note: assumption is always that n_obs_steps < action_horizon
        # For example:
        # "observation.state": [-0.1, 0.0],
        # "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
        # Use lists to accumulate tensors
        obs_list = []
        next_obs_list = []
        action_list = []
        reward_list = []
        absorbing_list = []
        last_list = []
        
        # Pre-compute discount powers (constant across all episodes)
        discount_powers = (self.mdp_info.gamma ** torch.arange(action_horizon)).unsqueeze(0)
        
        for idx, episode in enumerate(episodes):
            # stack obs
            # compute obs indices
            obs_indices = torch.arange(len(episode['obs'])).unsqueeze(1) - torch.arange(n_obs_steps-1, -1, -1)
            # next obs indices are with a gap of the shift due to the action chunk
            next_obs_indices = obs_indices + action_horizon - n_obs_steps
            # correct for indices out of range. Just pad with the first/last element
            obs_indices = torch.clip(obs_indices, 0, len(episode['obs'])-1)
            next_obs_indices = torch.clip(next_obs_indices, 0, len(episode['next_obs'])-1)
            obs_stack = episode['obs'][obs_indices]
            next_obs_stack = episode['next_obs'][next_obs_indices]
            obs_list.append(obs_stack)
            next_obs_list.append(next_obs_stack)
            # stack actions
            act_indices = torch.arange(len(episode['action'])).unsqueeze(1) - torch.arange(n_obs_steps-1, n_obs_steps-1-action_horizon, -1)
            # correct for indices out of range. Just pad with the first/last element
            act_indices = torch.clip(act_indices, 0, len(episode['action'])-1)
            act_stack = episode['action'][act_indices]
            action_list.append(act_stack)
            # accumulate rewards for the action chunk, zero rewards for out of range indices
            reward_indices = torch.arange(len(episode['reward'])).unsqueeze(1) - torch.arange(n_obs_steps-1, n_obs_steps-1-action_horizon, -1)
            episode_reward_array = torch.cat((episode['reward'], torch.zeros((1,1))), dim=0) # add zero reward at end for out of range indices
            out_of_range = (reward_indices < 0) | (reward_indices >= len(episode['reward']))
            reward_indices[out_of_range] = len(episode['reward']) # new end index will have a zero reward value
            discounted_rewards = episode_reward_array[reward_indices] * discount_powers.T
            acc_rewards = torch.sum(discounted_rewards, dim=1)
            reward_list.append(acc_rewards)
            # rearrange last and absorbing: move them forward the same amount as we moved the next_obs since they are in sync
            absorbing_indices = torch.arange(len(episode['absorbing']))
            absorbing_indices = absorbing_indices + action_horizon - n_obs_steps
            absorbing_indices = torch.clip(absorbing_indices, 0, len(episode['absorbing'])-1)
            episode_absorbing_array = episode['absorbing'][absorbing_indices]
            absorbing_list.append(episode_absorbing_array)
            last_indices = torch.arange(len(episode['last']))
            last_indices = last_indices + action_horizon - n_obs_steps
            last_indices = torch.clip(last_indices, 0, len(episode['last'])-1)
            episode_last_array = episode['last'][last_indices]
            last_list.append(episode_last_array)
            # TODO: make this function faster
            if debug and idx > 500:
                print("[[Debugging so skipping time consuming data rearrangement]]")
                break
        
        # Concatenate all accumulated tensors
        rearranged_dataset = {
            'obs': torch.cat(obs_list, dim=0),
            'next_obs': torch.cat(next_obs_list, dim=0),
            'action': torch.cat(action_list, dim=0),
            'reward': torch.cat(reward_list, dim=0),
            'absorbing': torch.cat(absorbing_list, dim=0),
            'last': torch.cat(last_list, dim=0)
        }
        
        # Squeeze rewards, absorbings and last into a single dimension for correct shapes during training
        rearranged_dataset['reward'] = rearranged_dataset['reward'].squeeze(1)
        rearranged_dataset['absorbing'] = rearranged_dataset['absorbing'].squeeze(1)
        rearranged_dataset['last'] = rearranged_dataset['last'].squeeze(1)

        # move devices if needed
        # rearranged_dataset['obs'] = rearranged_dataset['obs'].to(TorchUtils.get_device())
        # rearranged_dataset['action'] = rearranged_dataset['action'].to(TorchUtils.get_device())
        # rearranged_dataset['reward'] = rearranged_dataset['reward'].to(TorchUtils.get_device())
        # rearranged_dataset['next_obs'] = rearranged_dataset['next_obs'].to(TorchUtils.get_device())
        # rearranged_dataset['absorbing'] = rearranged_dataset['absorbing'].to(TorchUtils.get_device())
        # rearranged_dataset['last'] = rearranged_dataset['last'].to(TorchUtils.get_device())

        # Roll into a single dimension for now.
        # The intermediate dimension will be reintroduced when we batch before sending to DP
        # Flatten the obs and action dimensions: (batch, n_obs_steps, obs_dim) -> (batch, n_obs_steps * obs_dim)
        obs_flat = rearranged_dataset['obs'].view(rearranged_dataset['obs'].shape[0], -1)
        next_obs_flat = rearranged_dataset['next_obs'].view(rearranged_dataset['next_obs'].shape[0], -1)
        action_flat = rearranged_dataset['action'].view(rearranged_dataset['action'].shape[0], -1)
        
        chunked_dataset = Dataset.from_array(obs_flat, action_flat, rearranged_dataset['reward'],
                                                   next_obs_flat, rearranged_dataset['absorbing'], 
                                                   rearranged_dataset['last'], backend='torch')
        
        return chunked_dataset, episode_starts, episode_ends
    
    def load_dataset(self, datasets, compute_mean_std=True, debug=False):
        # Load and chunk the dataset using the shared helper function
        chunked_offline_dataset, offline_episode_starts, offline_episode_ends = self._load_and_chunk_dataset(datasets, compute_mean_std, debug)
        
        # Optional: load into replay memory for offline-to-online training (Later)
        # Create new replay memory object and copy over offline dataset to the replay buffer
        # For the replay memory, change mdp info state and action sizes to match the chunked dataset
        # chunked_obs_size = chunked_offline_dataset.state.shape[1]
        # chunked_act_size = chunked_offline_dataset.action.shape[1]
        # replay_mdp_info = deepcopy(self.mdp_info)
        # replay_mdp_info.observation_space = spaces.Box(
        #     -np.inf * np.ones(chunked_obs_size, dtype=np.float32),
        #     np.inf * np.ones(chunked_obs_size, dtype=np.float32))
        # replay_mdp_info.action_space = spaces.Box(
        #     -1.0 * np.ones(chunked_act_size, dtype=np.float32),
        #     1.0 * np.ones(chunked_act_size, dtype=np.float32))
        # self._replay_memory = ReplayMemory(replay_mdp_info, self.info, initial_size=len(chunked_offline_dataset), max_size=max(self._replay_memory._max_size, len(chunked_offline_dataset)))
        # self._replay_memory.add(chunked_offline_dataset)
        
        self.offline_dataset = chunked_offline_dataset
        self.offline_episode_starts = offline_episode_starts
        self.offline_episode_ends = offline_episode_ends

        # else:
        #     # No chunking, just copy over offline dataset to the replay buffer
        #     self._replay_memory._initial_size = len(self.offline_dataset) # set initial size to the size of the offline dataset
        #     if self._replay_memory._max_size < len(self.offline_dataset):
        #         print('[[Warning: Offline dataset size exceeds max replay memory size. Resizing replay memory to fit dataset.]]')
        #         self._replay_memory = ReplayMemory(self.mdp_info, self.info, len(self.offline_dataset), len(self.offline_dataset))
        #     self._replay_memory.add(self.offline_dataset)
    
    def load_optimal_dataset(self, datasets, compute_mean_std=True, debug=False):
        """
        Load optimal dataset for computing critic errors.
        Similar to load_dataset but stores the optimal dataset separately.
        
        Args:
            datasets: list of dictionaries with keys: obs, action, reward, next_obs, absorbing, last
            compute_mean_std: whether to compute mean/std for normalization (uses same normalization as offline dataset)
            debug: if True, skip time-consuming data rearrangement for debugging
        """
        # Check that mean/std are already computed if normalize_states is True
        if self._normalize_states and compute_mean_std:
            if self._states_mean is None or self._states_std is None:
                raise ValueError('States mean and std not computed yet. Call load_dataset() first.')
        
        # Load and chunk the dataset using the shared helper function
        # Pass compute_mean_std=False since we should use existing normalization
        chunked_optimal_dataset, optimal_episode_starts, optimal_episode_ends = self._load_and_chunk_dataset(datasets, compute_mean_std=False, debug=debug)

        self.optimal_dataset = chunked_optimal_dataset
        self.optimal_episode_starts = optimal_episode_starts
        self.optimal_episode_ends = optimal_episode_ends
    
    def load_actor_dataset(self, datasets, compute_mean_std=True, debug=False):
        """
        Load actor dataset for actor training and critic error computation.
        Similar to load_optimal_dataset but stores the actor dataset separately.
        
        Args:
            datasets: list of dictionaries with keys: obs, action, reward, next_obs, absorbing, last
            compute_mean_std: whether to compute mean/std for normalization (uses same normalization as offline dataset)
            debug: if True, skip time-consuming data rearrangement for debugging
        """
        # Check that mean/std are already computed if normalize_states is True
        if self._normalize_states and compute_mean_std:
            if self._states_mean is None or self._states_std is None:
                raise ValueError('States mean and std not computed yet. Call load_dataset() first.')
        
        # Load and chunk the dataset using the shared helper function
        # Pass compute_mean_std=False since we should use existing normalization
        chunked_actor_dataset, actor_episode_starts, actor_episode_ends = self._load_and_chunk_dataset(datasets, compute_mean_std=False, debug=debug)

        self.actor_dataset = chunked_actor_dataset
        self.actor_episode_starts = actor_episode_starts
        self.actor_episode_ends = actor_episode_ends
    
    def offline_fit(self, n_epochs, fit_critic=True, fit_actor=True):
        # Determine which dataset to use based on what we're training
        if fit_actor and self.actor_dataset is not None:
            # Use actor dataset for actor training
            dataset = self.actor_dataset
        elif fit_critic and self.offline_dataset is not None:
            # Use offline dataset for critic training
            dataset = self.offline_dataset
        else:
            # Fallback to offline_dataset if actor_dataset not available
            if self.offline_dataset is None:
                raise ValueError('No offline dataset loaded!. Call load_dataset() first.')
            dataset = self.offline_dataset
        
        # Initialize lists to accumulate losses for averaging
        acc_actor_loss = []
        acc_q_loss = []
        acc_value_loss = []
        acc_exp_adv = []
        acc_actor_bc_loss = []
        acc_actor_q_loss = []
        
        # fit on the dataset (for n_epochs)
        # for epoch in trange(n_epochs):
        #     state, action, reward, next_state, absorbing, _ = self._replay_memory.get(self._batch_size())
        epoch_count = 0
        with tqdm(total=n_epochs, disable=should_disable_tqdm()) as pbar:
            for state, action, reward, next_state, absorbing in minibatch_generator(
                self._batch_size(), dataset.state, dataset.action,
                dataset.reward, dataset.next_state, dataset.absorbing):

                # if self._normalize_states: # Assumed done at load time
                #     state_fit = self._norm_states(state)
                #     next_state_fit = self._norm_states(next_state)
                # else:
                state_fit = state
                next_state_fit = next_state

                if self._actor_loss_type == 'ddpg_plus_bc' and fit_critic == fit_actor:
                    # For DDPG+BC, fit critic first, then actor
                    self.iql_fit(state_fit, action, reward, next_state_fit, absorbing, fit_critic, False)
                    self.iql_fit(state_fit, action, reward, next_state_fit, absorbing, False, fit_actor)
                else:
                    self.iql_fit(state_fit, action, reward, next_state_fit, absorbing, fit_critic, fit_actor)
                
                # Accumulate losses for averaging
                if self._actor_last_loss is not None:
                    acc_actor_loss.append(self._actor_last_loss)
                if self._q_last_loss is not None:
                    acc_q_loss.append(self._q_last_loss)
                if self._value_last_loss is not None:
                    acc_value_loss.append(self._value_last_loss)
                if self._last_exp_adv is not None:
                    acc_exp_adv.append(self._last_exp_adv)
                if self._actor_last_bc_loss is not None:
                    acc_actor_bc_loss.append(self._actor_last_bc_loss)
                if self._actor_last_q_loss is not None:
                    acc_actor_q_loss.append(self._actor_last_q_loss)
                
                epoch_count += 1
                pbar.update(1)
                if epoch_count >= n_epochs:
                    break
        
        # Store averaged losses for logging
        if len(acc_actor_loss) > 0:
            self._actor_last_loss = np.mean(acc_actor_loss)
        if len(acc_q_loss) > 0:
            self._q_last_loss = np.mean(acc_q_loss)
        if len(acc_value_loss) > 0:
            self._value_last_loss = np.mean(acc_value_loss)
        if len(acc_exp_adv) > 0:
            self._last_exp_adv = np.mean(acc_exp_adv)
        if len(acc_actor_bc_loss) > 0:
            self._actor_last_bc_loss = np.mean(acc_actor_bc_loss)
        if len(acc_actor_q_loss) > 0:
            self._actor_last_q_loss = np.mean(acc_actor_q_loss)

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
            
            if self._actor_loss_type == 'ddpg_plus_bc' and fit_critic == fit_actor:
                # For DDPG+BC, fit critic first, then actor
                self.iql_fit(state_fit, action, reward, next_state_fit, absorbing, fit_critic, False)
                self.iql_fit(state_fit, action, reward, next_state_fit, absorbing, False, fit_actor)
            else:
                self.iql_fit(state_fit, action, reward, next_state_fit, absorbing, fit_critic, fit_actor)

    def iql_fit(self, state, action, reward, next_state, absorbing, fit_critic=True, fit_actor=True):
        if self._actor_loss_type == 'awr':
            with torch.no_grad():
                next_v = self._value_func_approximator(next_state, **self._critic_predict_params)
                next_v = next_v.cpu()
            # Get advantage & update value function (if fit_critic)
            adv = self._get_adv_and_update_v(state, action, fit_critic)
            if fit_critic:
                # Update Q function
                self._update_q(next_v, state, action, reward, absorbing)
            if fit_actor:
                # Update actor
                self._update_actor_awr(adv, state, action)
        elif self._actor_loss_type == 'ddpg_plus_bc':
            if fit_critic:
                with torch.no_grad():
                    next_v = self._value_func_approximator(next_state, **self._critic_predict_params)
                    next_v = next_v.cpu()
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
        
        # Store Q loss for logging
        if hasattr(self._critic_approximator[0], 'loss_fit'):
            self._q_last_loss = self._critic_approximator[0].loss_fit
        else:
            self._q_last_loss = None

        # Update target critic
        self._update_target(self._critic_approximator, self._target_critic_approximator)

    def _update_actor_awr(self, adv, state, action):
        # compute advantage weighted BC loss
        exp_adv = torch.exp(self._iql_beta() * adv.detach()).clamp(max=self._max_clamp_adv())

        # target action from data:
        act = torch.as_tensor(action, dtype=torch.float32, device=TorchUtils.get_device())
        
        # Query DP for BC loss
        batch = self._create_batch_for_dp(state, act)
        bc_loss = self.policy.forward(batch, self._squash_actions)['loss']

        # Compute actor loss
        actor_loss = torch.mean(exp_adv * bc_loss)

        self._optimize_actor_parameters(actor_loss)
        
        if self._schedule_actor_lr:
            self._actor_lr_scheduler.step()

        self._actor_last_loss = actor_loss.detach().cpu().numpy() # Store actor loss for logging
        self._last_exp_adv = exp_adv.detach().mean().cpu().numpy() # Store exp_adv for logging

    def _update_actor_ddpg_plus_bc(self, state, action):
        # target state, action from data:
        act = torch.as_tensor(action, dtype=torch.float32, device=TorchUtils.get_device())
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=TorchUtils.get_device())
        
        # Query DP for predicted action and BC loss
        batch = self._create_batch_for_dp(state_tensor, act)
        policy_forward_output = self.policy.forward(batch, self._squash_actions)
        act_pred = policy_forward_output['act_pred']
        bc_loss = policy_forward_output['loss']
        bc_loss = bc_loss.mean((1,2)) # mean over action dim and horizon dim to get (batch,)

        # DEBUG:
        # actor_loss = bc_loss.mean()
        # q_loss = torch.tensor(0.0).to(TorchUtils.get_device())
        # END DEBUG
        # DDPG loss - need to flatten the predicted action for the critic
        act_pred_flat = self._flatten_action_for_critic(act_pred).cpu()
        q = self._critic_approximator(state_tensor, act_pred_flat, **self._critic_predict_params)
        q_loss = -q

        # Total loss
        actor_loss = torch.mean(q_loss + self._bc_weight_in_ddpg() * bc_loss)

        self._optimize_actor_parameters(actor_loss)
        
        if self._schedule_actor_lr:
            self._actor_lr_scheduler.step()

        self._actor_last_loss = actor_loss.detach().cpu().numpy() # Store actor loss for logging
        self._actor_last_bc_loss = bc_loss.detach().mean().cpu().numpy() # Store BC loss for logging
        self._actor_last_q_loss = q_loss.detach().mean().cpu().numpy() # Store Q loss for logging

    def _create_batch_for_dp(self, state, action):
        """Create batch dictionary for Diffusion Policy with reshaped tensors."""
        n_obs_steps = self.policy._n_obs_steps
        action_horizon = self.policy._horizon
        obs_dim = self.mdp_info.observation_space.shape[0]  # Original observation dimension
        act_dim = self.mdp_info.action_space.shape[0]  # Original action dimension
        
        # Ensure tensors are on the correct device
        device = TorchUtils.get_device()
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        action = torch.as_tensor(action, dtype=torch.float32, device=device)
        
        state_reshaped = state.view(-1, n_obs_steps, obs_dim)
        action_reshaped = action.view(-1, action_horizon, act_dim)
        
        batch = {'observation.state': state_reshaped, 'action': action_reshaped}
        return batch

    def _flatten_action_for_critic(self, action_tensor):
        """Flatten action tensor for critic network."""
        action_horizon = self.policy._horizon
        act_dim = self.mdp_info.action_space.shape[0]
        return action_tensor.view(-1, action_horizon * act_dim)

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
    
    def _compute_q_values_from_rewards(self, rewards_tensor, absorbing_tensor, gamma_horizon, shift=1):
        """
        Compute Q values from rewards and absorbing flags using backward recurrence.
        
        Args:
            rewards_tensor: Tensor of rewards for each state-action pair
            absorbing_tensor: Tensor of absorbing flags (boolean or 0/1)
            gamma_horizon: Discount factor raised to horizon power
            shift: Number of steps to shift forward (horizon - n_obs_steps for chunked data)
            
        Returns:
            Tensor of Q values for each state-action pair
        """
        episode_length = len(rewards_tensor)
        
        # Convert absorbing to boolean if needed
        if absorbing_tensor.dtype != torch.bool:
            absorbing_tensor = absorbing_tensor.bool()
        
        # Compute future returns backwards
        future_returns = torch.zeros(episode_length, device=rewards_tensor.device, dtype=torch.float32)
        for i in range(episode_length - shift - 1, -1, -1):
            next_idx = i + shift
            if next_idx < episode_length:
                # Future return is the reward at next chunk + discounted future return
                future_returns[i-1] = rewards_tensor[next_idx] + (~absorbing_tensor[next_idx]) * gamma_horizon * future_returns[next_idx]
        
        # Compute Q values
        q_values = rewards_tensor + (~absorbing_tensor) * gamma_horizon * future_returns
        
        return q_values
    
    def _get_valid_episode_indices(self, dataset, episode_starts, episode_ends, dataset_name):
        """
        Filter episodes to only include those ending with absorbing=True.
        
        Args:
            dataset: Dataset object
            episode_starts: List of episode start indices
            episode_ends: List of episode end indices
            dataset_name: Name of dataset for error messages
        
        Returns:
            numpy array of valid episode indices
        """
        end_indices = np.array(episode_ends)
        valid_mask = end_indices > 0
        if valid_mask.any():
            last_absorbing_indices = end_indices[valid_mask] - 1
            absorbing_flags = dataset.absorbing[last_absorbing_indices]
            if isinstance(absorbing_flags, torch.Tensor):
                absorbing_flags = absorbing_flags.squeeze().cpu().numpy()
            else:
                absorbing_flags = np.array(absorbing_flags).squeeze()
            absorbing_mask = np.zeros(len(episode_ends), dtype=bool)
            absorbing_mask[valid_mask] = absorbing_flags > 0.5
            valid_episode_indices = np.where(absorbing_mask)[0]
        else:
            valid_episode_indices = np.array([], dtype=int)
        
        if len(valid_episode_indices) == 0:
            raise ValueError(f'No episodes found with absorbing=True at the end in {dataset_name} dataset.')
        
        return valid_episode_indices
    
    def _compute_critic_errors_for_episodes(self, dataset, episode_starts, episode_ends, sampled_indices, gamma_horizon, shift):
        """
        Compute critic errors for a set of sampled episodes.
        
        Args:
            dataset: Dataset object
            episode_starts: List of episode start indices
            episode_ends: List of episode end indices
            sampled_indices: Array of episode indices to process
            gamma_horizon: Discount factor raised to horizon power
            shift: Number of steps to shift forward
        
        Returns:
            tuple: (errors_list, values_list) - lists of errors and values
        """
        errors_list = []
        values_list = []
        
        for ep_idx in sampled_indices:
            start_idx = episode_starts[ep_idx]
            end_idx = episode_ends[ep_idx]
            
            # Extract episode data
            episode_states = dataset.state[start_idx:end_idx]
            episode_actions = dataset.action[start_idx:end_idx]
            episode_rewards = dataset.reward[start_idx:end_idx]
            episode_absorbing = dataset.absorbing[start_idx:end_idx]
            
            # Process rewards and absorbing flags
            rewards_tensor = episode_rewards.squeeze() if (hasattr(episode_rewards, 'dim') and episode_rewards.dim() > 1) else episode_rewards
            absorbing_tensor = episode_absorbing.squeeze() if (hasattr(episode_absorbing, 'dim') and episode_absorbing.dim() > 1) else episode_absorbing
            
            # Compute true Q values
            true_q_values = self._compute_q_values_from_rewards(rewards_tensor, absorbing_tensor, gamma_horizon, shift)
            
            # Compute network's Q values using critic
            with torch.no_grad():
                critic_q_values = self._critic_approximator(episode_states, episode_actions, **self._critic_predict_params)
            
            # Compute errors
            if critic_q_values.device != true_q_values.device:
                critic_q_values = critic_q_values.to(true_q_values.device)
            errors = critic_q_values - true_q_values
            errors_list.extend(errors.cpu().numpy().tolist())
            values_list.extend(critic_q_values.cpu().numpy().tolist())
        
        return errors_list, values_list
    
    def compute_critic_errors(self, optimal_data_percent=0.1):
        """
        Compute critic errors on the optimal, offline, and actor datasets.
        Args:
            optimal_data_percent: percentage of the optimal dataset to sample
        
        Returns:
            dict: dictionary containing critic error metrics
        """
        if self.optimal_dataset is None:
            raise ValueError('No optimal dataset loaded!. Call load_optimal_dataset() first.')
        if self.offline_dataset is None:
            raise ValueError('No offline dataset loaded!. Call load_dataset() first.')

        # Get valid episode indices for each dataset
        optimal_valid_indices = self._get_valid_episode_indices(
            self.optimal_dataset, self.optimal_episode_starts, self.optimal_episode_ends, 'optimal')
        offline_valid_indices = self._get_valid_episode_indices(
            self.offline_dataset, self.offline_episode_starts, self.offline_episode_ends, 'offline')
        
        # Sample episodes
        n_samples_optimal = max(1, int(len(optimal_valid_indices) * optimal_data_percent))
        sampled_optimal_indices = np.random.choice(
            optimal_valid_indices, size=min(n_samples_optimal, len(optimal_valid_indices)), replace=False)
        n_samples_offline = min(len(sampled_optimal_indices), len(offline_valid_indices))
        sampled_offline_indices = np.random.choice(offline_valid_indices, size=n_samples_offline, replace=False)
        
        # Compute critic errors
        gamma_horizon = self.mdp_info.gamma ** self.policy._horizon
        shift = self.policy._horizon - self.policy._n_obs_steps
        
        optimal_errors, optimal_values = self._compute_critic_errors_for_episodes(
            self.optimal_dataset, self.optimal_episode_starts, self.optimal_episode_ends,
            sampled_optimal_indices, gamma_horizon, shift)
        offline_errors, offline_values = self._compute_critic_errors_for_episodes(
            self.offline_dataset, self.offline_episode_starts, self.offline_episode_ends,
            sampled_offline_indices, gamma_horizon, shift)
        
        # Process actor dataset if available
        actor_errors, actor_values = [], []
        if self.actor_dataset is not None:
            actor_valid_indices = self._get_valid_episode_indices(
                self.actor_dataset, self.actor_episode_starts, self.actor_episode_ends, 'actor')
            if len(actor_valid_indices) > 0:
                n_samples_actor = min(len(sampled_optimal_indices), len(actor_valid_indices))
                sampled_actor_indices = np.random.choice(actor_valid_indices, size=n_samples_actor, replace=False)
                actor_errors, actor_values = self._compute_critic_errors_for_episodes(
                    self.actor_dataset, self.actor_episode_starts, self.actor_episode_ends,
                    sampled_actor_indices, gamma_horizon, shift)
            else:
                print(f'No actor episodes found with absorbing=True at the end in actor dataset.')
        
        # Convert to numpy arrays and compute statistics
        optimal_errors = np.array(optimal_errors)
        optimal_values = np.array(optimal_values)
        offline_errors = np.array(offline_errors)
        offline_values = np.array(offline_values)
        actor_errors = np.array(actor_errors) if len(actor_errors) > 0 else np.array([])
        actor_values = np.array(actor_values) if len(actor_values) > 0 else np.array([])
        
        # Resample to ensure equal number of values from each dataset
        if len(optimal_values) > 0 and len(offline_values) > 0:
            if len(actor_values) > 0:
                # Find minimum length among all three
                min_length = min(len(optimal_values), len(offline_values), len(actor_values))
            else:
                # Find minimum length between optimal and offline
                min_length = min(len(optimal_values), len(offline_values))
            
            # Resample all arrays to have equal length
            if len(optimal_values) > min_length:
                indices = np.random.choice(len(optimal_values), size=min_length, replace=False)
                optimal_errors = optimal_errors[indices]
                optimal_values = optimal_values[indices]
            if len(offline_values) > min_length:
                indices = np.random.choice(len(offline_values), size=min_length, replace=False)
                offline_errors = offline_errors[indices]
                offline_values = offline_values[indices]
            if len(actor_values) > 0 and len(actor_values) > min_length:
                indices = np.random.choice(len(actor_values), size=min_length, replace=False)
                actor_errors = actor_errors[indices]
                actor_values = actor_values[indices]
        
        # Build results dictionary
        critic_errors_dict = {
            'optimal_critic_error_mean': np.mean(optimal_errors) if len(optimal_errors) > 0 else 0.0,
            'optimal_critic_error_std': np.std(optimal_errors) if len(optimal_errors) > 0 else 0.0,
            'offline_critic_error_mean': np.mean(offline_errors) if len(offline_errors) > 0 else 0.0,
            'offline_critic_error_std': np.std(offline_errors) if len(offline_errors) > 0 else 0.0,
            'optimal_minus_offline_value_mean': np.mean(optimal_values) - np.mean(offline_values) if len(optimal_values) > 0 and len(offline_values) > 0 else 0.0,
        }
        
        if len(actor_errors) > 0:
            critic_errors_dict['actor_critic_error_mean'] = np.mean(actor_errors)
            critic_errors_dict['actor_critic_error_std'] = np.std(actor_errors)
            critic_errors_dict['optimal_minus_actor_value_mean'] = np.mean(optimal_values) - np.mean(actor_values)
            critic_errors_dict['offline_minus_actor_value_mean'] = np.mean(offline_values) - np.mean(actor_values)
        
        return critic_errors_dict
        
    def create_new_model(self, model_config):
        """
        Create a new model and assign it to the policy
        """
        # Create new model
        self.policy._model = model_config['model_class'](model_config)
        # Move model to correct device
        self.policy._model.to(TorchUtils.get_device())
    
    def _post_load(self):
        # reset tracking variables
        self._actor_last_loss = None
        self._actor_last_bc_loss = None
        self._actor_last_q_loss = None
        self._q_last_loss = None
        self._value_last_loss = None
        self._last_exp_adv = None

        if not hasattr(self, 'offline_dataset'):
            self.offline_dataset = None
        if not hasattr(self, 'optimal_dataset'):
            self.optimal_dataset = None
        if not hasattr(self, 'actor_dataset'):
            self.actor_dataset = None
        if not hasattr(self, 'offline_episode_starts'):
            self.offline_episode_starts = None
        if not hasattr(self, 'offline_episode_ends'):
            self.offline_episode_ends = None
        if not hasattr(self, 'optimal_episode_starts'):
            self.optimal_episode_starts = None
        if not hasattr(self, 'optimal_episode_ends'):
            self.optimal_episode_ends = None
        if not hasattr(self, 'actor_episode_starts'):
            self.actor_episode_starts = None
        if not hasattr(self, 'actor_episode_ends'):
            self.actor_episode_ends = None

        # self._actor_approximator = self.policy._approximator
        # self._update_optimizer_parameters(self._actor_approximator.model.network.parameters())
