import torch
import numpy as np

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.rl_utils.parameters import Parameter, to_parameter
from mushroom_rl.utils.torch import TorchUtils
from tqdm import tqdm, trange

# from torch.nn.functional import binary_cross_entropy_with_logits

class BC_DP(DeepAC):
    """
    BEHAVIOR CLONING algorithm with DiffusionPolicy that builds on top of the DeepAC class.
    Even though the algorithm is not an actor-critic method, it is implemented
    here to compare against other methods such as TD3+BC.
    Normally uses DiffusionPolicy (https://diffusion-policy.cs.columbia.edu/)

    """
    def __init__(self, mdp_info, policy_class, policy_params,
                 actor_params, actor_optimizer, critic_params=None,
                 batch_size=1, n_epochs_policy=1, patience=1, squash_actions=False,
                 discrete_action_dims=0, continuous_action_dims=0,
                 normalize_states=False, normalize_actions=False,
                 critic_fit_params=None, actor_predict_params=None, critic_predict_params=None):
        """
        Constructor.

        Args:
            policy_class (Policy): class of the policy;
            policy_params (dict): parameters of the policy to build;
            actor_params (dict): parameters of the actor approximator to
                build;
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            critic_params (dict): Unused parameter; Left for future
            batch_size ([int, Parameter]): size of minibatches for every optimization step
            n_epochs_policy ([int, Parameter]): number of policy update epochs on the whole dataset at every call;
            patience (float, 1.): (Optional) the number of epochs to wait until stop
                the learning if not improving;
            squash_actions (bool, True): (Optional) whether to squash the actions to [-1, 1] with tanh;
            discrete_action_dims (int, 0): number of discrete actions in the action space;
            continuous_action_dims (int, 0): number of continuous actions in the action space;
            normalize_states (bool, False): whether to normalize states;
            normalize_actions (bool, False): whether to normalize actions;
            critic_fit_params (dict, None): Unused parameter; Left for future
            actor_predict_params (dict, None): Unused parameter; Left for future
            critic_predict_params (dict, None): Unused parameter; Left for future

        """

        # self._actor_predict_params = dict() if actor_predict_params is None else actor_predict_params
        
        # self._actor_approximator = Regressor(TorchApproximator, **actor_params)

        # policy = policy_class(self._actor_approximator, **policy_params)
        policy = policy_class(policy_params)

        policy_parameters = policy._model.parameters()

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)


        self._batch_size = to_parameter(batch_size)
        self._n_epochs_policy = to_parameter(n_epochs_policy)
        # self._patience = to_parameter(patience)
        self._squash_actions = squash_actions
        self._normalize_states = normalize_states
        self._states_mean = None
        self._states_std = None
        self._normalize_actions = normalize_actions
        self._actions_mean = None
        self._actions_std = None
        self._discrete_action_dims = discrete_action_dims # unused for now
        self._continuous_action_dims = continuous_action_dims # unused for now

        self._fit_count = 0
        self._actor_last_loss = None # Store actor loss for logging

        # Optimizer deviations from mushroom_rl
        if policy_params['use_transformer'] is True:
            # create the transformer optimizers here and assign to self
            self._optimizer = policy._model.net.configure_optimizers(learning_rate=policy_params['transformer_lr_actor_net'],
                                                                weight_decay=policy_params['transformer_weight_decay_actor_net'],
                                                                betas=policy_params['transformer_betas_actor_net'])
            self._parameters = policy._model.parameters()
        # remove optimizer save attribute from super class since we will use our own method to save model and optimizer instead
        del self._save_attributes['_optimizer']

        self._add_save_attr(
            _batch_size='mushroom',
            _n_epochs_policy='mushroom',
            # _patience='mushroom',
            _squash_actions='primitive',
            _normalize_states='primitive',
            _states_mean='primitive',
            _states_std='primitive',
            _normalize_actions='primitive',
            _actions_mean='torch',
            _actions_std='torch',
            _discrete_action_dims='primitive',
            _continuous_action_dims='primitive',
            # _actor_predict_params='pickle',
            # _actor_approximator='mushroom',
            _fit_count='primitive',
        )
    
    def _get_episode_boundaries(self, last_array):
        """
        Find episode boundaries in a dataset using the 'last' flag.
        
        Args:
            last_array: Tensor or array with 'last' flags indicating episode ends
            
        Returns:
            tuple: (episode_starts, episode_ends) - lists of start and end indices for each episode
        """
        # Convert last to boolean if needed (handle both bool and numeric types)
        if isinstance(last_array, torch.Tensor):
            last_array_squeezed = last_array.squeeze() if last_array.dim() > 1 else last_array
            if last_array_squeezed.dtype != torch.bool:
                last_array_squeezed = last_array_squeezed.bool()
            # Find all episode end indices (where last[i] == True)
            episode_end_indices = torch.where(last_array_squeezed)[0].cpu().numpy()
        else:
            last_array_squeezed = last_array.squeeze() if last_array.ndim > 1 else last_array
            if last_array_squeezed.dtype != bool:
                last_array_squeezed = last_array_squeezed.astype(bool)
            # Find all episode end indices (where last[i] == True)
            episode_end_indices = np.where(last_array_squeezed)[0]
        
        # Compute episode start and end indices
        if len(episode_end_indices) > 0:
            episode_starts = [0] + (episode_end_indices[:-1] + 1).tolist()
            episode_ends = (episode_end_indices + 1).tolist()
            # Handle case where dataset doesn't end with last=True
            # Include remaining samples as the last episode
            dataset_length = len(last_array_squeezed)
            if episode_ends[-1] < dataset_length:
                episode_starts.append(episode_ends[-1])
                episode_ends.append(dataset_length)
        else:
            # No episode boundaries found - treat entire dataset as one episode
            dataset_length = len(last_array_squeezed) if isinstance(last_array_squeezed, torch.Tensor) else len(last_array_squeezed)
            episode_starts = [0]
            episode_ends = [dataset_length]
        
        return episode_starts, episode_ends
    
    def load_dataset(self, dataset, debug=False):
        # Copy over dataset. Convert to torch tensors
        self.dataset = dict()
        self.dataset['obs'] = torch.as_tensor(dataset['obs'], dtype=torch.float32)
        self.dataset['action'] = torch.as_tensor(dataset['action'], dtype=torch.float32)
        self.dataset['last'] = torch.as_tensor(dataset['last'], dtype=torch.bool)
        
        # normalize if needed
        if self._normalize_states:
            self._compute_states_mean_std(self.dataset['obs'])
            # save the normalized states in the dataset
            self.dataset['obs'] = self._norm_states(self.dataset['obs'])
            # move devices if needed
            self._states_mean = self._states_mean.to(TorchUtils.get_device())
            self._states_std = self._states_std.to(TorchUtils.get_device())
        if self._normalize_actions:
            self._compute_actions_mean_std(self.dataset['action'])
            # save the normalized actions in the dataset
            self.dataset['action'] = self._norm_actions(self.dataset['action'])
            # move devices if needed
            self._actions_mean = self._actions_mean.to(TorchUtils.get_device())
            self._actions_std = self._actions_std.to(TorchUtils.get_device())

        ## rearrange data based on obs_horizon, action_pred_horizon etc. as per diffusion policy
        n_obs_steps = self.policy._n_obs_steps
        action_horizon = self.policy._horizon
        # Find episode boundaries using the 'last' array
        episode_starts, episode_ends = self._get_episode_boundaries(self.dataset['last'])
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
                'obs': self.dataset['obs'][start_idx:end_idx],
                'action': self.dataset['action'][start_idx:end_idx]
            }
            episodes.append(episode)
        # stack batches of size n_obs_steps for the obs and size horizon for the actions
        # Note: assumption is always that n_obs_steps < action_horizon
        # For example:
        # "observation.state": [-0.1, 0.0],
        # "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
        # Use lists to accumulate tensors (O(1) append) instead of repeated torch.cat (O(n²))
        obs_list = []
        action_list = []
        for idx, episode in enumerate(episodes):
            # stack obs
            obs_indices = torch.arange(len(episode['obs'])).unsqueeze(1) - torch.arange(n_obs_steps-1, -1, -1)
            # correct for indices out of range. Just pad with the first/last element
            obs_indices = torch.clip(obs_indices, 0, len(episode['obs'])-1)
            obs_stack = episode['obs'][obs_indices]
            obs_list.append(obs_stack)
            # stack actions
            act_indices = torch.arange(len(episode['action'])).unsqueeze(1) - torch.arange(n_obs_steps-1, n_obs_steps-1-action_horizon, -1)
            # correct for indices out of range. Just pad with the first/last element
            act_indices = torch.clip(act_indices, 0, len(episode['action'])-1)
            act_stack = episode['action'][act_indices]
            action_list.append(act_stack)
            # TODO: make this function faster
            if debug and idx > 5:
                print("[[Debugging so skipping time consuming data rearrangement]]")
                break
        
        # Concatenate all accumulated tensors in a single operation (O(n) instead of O(n²))
        rearranged_dataset = {
            'obs': torch.cat(obs_list, dim=0),
            'action': torch.cat(action_list, dim=0)
        }
        
        # move devices if needed
        rearranged_dataset['obs'] = rearranged_dataset['obs'].to(TorchUtils.get_device())
        rearranged_dataset['action'] = rearranged_dataset['action'].to(TorchUtils.get_device())
        
        self.dataset = rearranged_dataset
    
    def fit(self, demo_dataset=None, n_epochs=None):
        if demo_dataset is None:
            demo_dataset = self.dataset
        if n_epochs is None:
            n_epochs = self._n_epochs_policy()
        
        acc_loss = []
        # fit on the dataset (for n_epochs)
        for epoch in trange(n_epochs):
            minibatches = len(demo_dataset['obs']) // self._batch_size()
            with tqdm(total=minibatches) as pbar:
                for obs, act in minibatch_generator(self._batch_size(), demo_dataset['obs'], demo_dataset['action']):
                    # if self._normalize_states:
                    #     state_fit = self._norm_states(obs)
                    # else:
                    #     state_fit = obs
                    # if self._normalize_actions:
                    #     act_fit = self._norm_actions(act)
                    # else:
                    #     act_fit = act
                    
                    # loss = self._loss(state_fit, act_fit)
                    
                    batch = {'observation.state': obs, 'action': act}
                    loss = self.policy.forward(batch, self._squash_actions)['loss'].mean()
                    self._optimize_actor_parameters(loss)

                    self._fit_count += 1
                    # losses for logging
                    acc_loss.append(loss.detach().cpu().numpy())

                    # early stopping: (Optional)
                    # check loss reduction, self._patience

                    # Update the progress bar
                    pbar.update(1)

        # Store mean actor loss for logging
        self._actor_last_loss = np.mean(acc_loss)
    
    # def _loss(self, state, act):
    #     # loss for behavior cloning
        
    #     # if isinstance(state, np.ndarray):
    #     #     state = torch.as_tensor(state, dtype=torch.float32)
    #     act = torch.as_tensor(act, dtype=torch.float32, device=TorchUtils.get_device())
    #     # act_disc = act[:, :self._discrete_action_dims]
    #     # act_cont = act[:, -self._continuous_action_dims:]

    #     # get predicted actions
    #     act_pred_cont = self.policy.draw_action(state)

    #     # act_pred = self._actor_approximator(state, **self._actor_predict_params)
    #     # act_pred_disc = act_pred[:, :self._discrete_action_dims]
    #     # act_pred_cont = act_pred[:, -self._continuous_action_dims:]
        
    #     if self._squash_actions:
    #         # Squash the continuous actions to [-1, 1] (Needed if RL policy squashes actions)
    #         act_pred_cont = torch.tanh(act_pred_cont)

    #     bc_loss = torch.zeros(1, device=TorchUtils.get_device())
    #     # if self._discrete_action_dims > 0:
    #     #     # ensure targets are binary
    #     #     act_disc = (act_disc > 0.5).float()
    #     #     # treating discrete actions as logits. Use binary cross entropy loss
    #     #     act_pred_disc = torch.sigmoid(act_pred_disc)
    #     #     # bc_loss += binary_cross_entropy_with_logits(act_pred_disc, act_disc)
    #     #     bc_loss += torch.mean(-act_disc * torch.log(act_pred_disc + 1e-8) - (1 - act_disc) * torch.log(1 - act_pred_disc + 1e-8))
    #     # if self._continuous_action_dims > 0:

    #     # Use mse loss for continuous actions
    #     bc_loss += torch.mean((act_pred_cont - act_cont)**2)

    #     return bc_loss
        
    def _compute_states_mean_std(self, states, eps: float = 1e-3):
        self._states_mean = states.mean(0)
        self._states_std = states.std(0) + eps

        # set them for the policy as well so that we use it when drawing actions
        self.policy._states_mean = self._states_mean
        self.policy._states_std = self._states_std

    def _norm_states(self, states):
        if self._states_mean is None or self._states_std is None:
            raise ValueError('States mean and std not computed yet. Call _compute_states_mean_std() on the dataset first.')
        return (states - self._states_mean) / self._states_std

    def _compute_actions_mean_std(self, actions, eps: float = 1e-3):
        self._actions_mean = actions.mean(0)
        self._actions_std = actions.std(0) + eps

        # set them for the policy as well so that we use it when drawing actions
        self.policy._actions_mean = self._actions_mean
        self.policy._actions_std = self._actions_std
    
    def _norm_actions(self, actions):
        if self._actions_mean is None or self._actions_std is None:
            raise ValueError('Actions mean and std not computed yet. Call _compute_actions_mean_std() on the dataset first.')
        return (actions - self._actions_mean) / self._actions_std
        
    def _post_load(self):
        # don't do optimizer stuff here. We will create a new one
        pass
        # self._actor_approximator = self.policy._approximator
        # self._update_optimizer_parameters(self._actor_approximator.model.network.parameters())
        # self._update_optimizer_parameters(self.policy._model.parameters())
