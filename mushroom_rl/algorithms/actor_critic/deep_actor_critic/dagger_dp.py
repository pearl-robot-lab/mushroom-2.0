import torch
import numpy as np

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.rl_utils.replay_memory import ReplayMemory
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core.dataset import Dataset
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.rl_utils.parameters import Parameter, to_parameter
from mushroom_rl.utils.torch import TorchUtils
from tqdm import tqdm, trange

# from torch.nn.functional import binary_cross_entropy_with_logits

class DAgger_DP(DeepAC):
    """
    "A reduction of imitation learning and structured prediction to no-regret online learning", JMLR 2011.
    DAgger algorithm that uses behavior cloning  with DiffusionPolicy built on top of the DeepAC class.
    Even though the algorithm is not an actor-critic method, it is implemented
    here to compare against other methods such as TD3+BC. Uses a replay memory to store expert actions.
    Normally uses ClippedGaussianPolicy
    Normally uses DiffusionPolicy (https://diffusion-policy.cs.columbia.edu/)

    """
    def __init__(self, mdp_info, policy_class, policy_params,
                 actor_params, actor_optimizer, critic_params=None,
                 batch_size=1, n_epochs_policy=1, patience=1, squash_actions=False,
                 discrete_action_dims=0, continuous_action_dims=0,
                 initial_replay_size=600, max_replay_size=10000, additional_replay=False,
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
            initial_replay_size (int, 600): the minimum number of samples before starting the learning with replay memory;
            max_replay_size (int, 10000): the maximum number of samples in the replay memory;
            additional_replay (bool, False): whether to use an additional replay memory for
            any additional expert action types (Eg. whole-body control actions);
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
        self._normalize_states = normalize_states # Unused for now
        self._states_mean = None
        self._states_std = None
        self._normalize_actions = normalize_actions  # Unused for now
        self._actions_mean = None
        self._actions_std = None
        self._discrete_action_dims = discrete_action_dims # unused for now
        self._continuous_action_dims = continuous_action_dims # unused for now

        # create an expert action replay memory buffer
        self._expert_replay_memory = ReplayMemory(mdp_info, self.info, initial_replay_size, max_replay_size)
        # (Optional) create another expert action replay memory buffer with a different actions (Eg. whole-body control actions)
        if additional_replay is True:
            self._additional_expert_replay_memory = ReplayMemory(mdp_info, self.info, initial_replay_size, max_replay_size)
        else:
            self._additional_expert_replay_memory = None

        self._fit_count = 0
        self._actor_last_loss = None # Store actor loss for logging

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
            _expert_replay_memory='mushroom',
            _additional_expert_replay_memory='mushroom',
            _fit_count='primitive',
            _actor_last_loss='primitive',
        )

    def load_dataset(self, dataset, debug=False):
        # For pre-training
        # Copy over dataset. Convert to torch tensors
        self.pre_train_dataset = dict()
        self.pre_train_dataset['obs'] = torch.as_tensor(dataset['obs'], dtype=torch.float32)
        self.pre_train_dataset['action'] = torch.as_tensor(dataset['action'], dtype=torch.float32)
        self.pre_train_dataset['last'] = torch.as_tensor(dataset['last'], dtype=torch.bool)

        ## rearrange data based on obs_horizon, action_pred_horizon etc. as per diffusion policy
        n_obs_steps = self.policy._n_obs_steps
        action_horizon = self.policy._horizon
        # rearrange into episodes
        episodes = []
        episode = {'obs': torch.empty((0, self.pre_train_dataset['obs'].shape[1])),
                   'action': torch.empty((0, self.pre_train_dataset['action'].shape[1]))}
        for i in range(len(self.pre_train_dataset['obs'])):
            episode['obs'] = torch.vstack((episode['obs'], self.pre_train_dataset['obs'][i].unsqueeze(0)))
            episode['action'] = torch.vstack((episode['action'], self.pre_train_dataset['action'][i].unsqueeze(0)))
            if self.pre_train_dataset['last'][i]:
                episodes.append(episode)
                episode = {'obs': torch.empty((0, self.pre_train_dataset['obs'].shape[1])),
                           'action': torch.empty((0, self.pre_train_dataset['action'].shape[1]))}
            if debug and i > 2000:
                print("[[Debugging so skipping time consuming data rearrangement]]")
                break
        # stack batches of size n_obs_steps for the obs and size horizon for the actions
        # Note: assumption is always that n_obs_steps < action_horizon
        # For example:
        # "observation.state": [-0.1, 0.0],
        # "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
        rearranged_dataset = {'obs': torch.empty((0, n_obs_steps, self.pre_train_dataset['obs'].shape[1])),
                              'action': torch.empty((0, action_horizon, self.pre_train_dataset['action'].shape[1]))}
        for idx, episode in enumerate(episodes):
            # stack obs
            obs_indices = torch.arange(len(episode['obs'])).unsqueeze(1) - torch.arange(n_obs_steps-1, -1, -1)
            # correct for indices out of range. Just pad with the first/last element
            obs_indices = torch.clip(obs_indices, 0, len(episode['obs'])-1)
            obs_stack = episode['obs'][obs_indices]
            rearranged_dataset['obs'] = torch.cat((rearranged_dataset['obs'], obs_stack), dim=0)
            # stack actions
            act_indices = torch.arange(len(episode['action'])).unsqueeze(1) - torch.arange(n_obs_steps-1, n_obs_steps-1-action_horizon, -1)
            # correct for indices out of range. Just pad with the first/last element
            act_indices = torch.clip(act_indices, 0, len(episode['action'])-1)
            act_stack = episode['action'][act_indices]
            rearranged_dataset['action'] = torch.cat((rearranged_dataset['action'], act_stack), dim=0)
            # TODO: make this function faster
            if debug and idx > 5:
                print("[[Debugging so skipping time consuming data rearrangement]]")
                break
        
        # move devices if needed
        rearranged_dataset['obs'] = rearranged_dataset['obs'].to(TorchUtils.get_device())
        rearranged_dataset['action'] = rearranged_dataset['action'].to(TorchUtils.get_device())
        
        self.pre_train_dataset = rearranged_dataset

        ## (Optional) Add this dataset to the replay memory:
        # # create mushroom dataset
        # if self._expert_replay_memory._max_size < len(dataset['obs']):
        #     print('[[Warning: pre-train dataset size exceeds max replay memory size! Sub-sampling!]]')
        #     rand_idx = np.random.choice(len(dataset['obs']), self._expert_replay_memory._max_size, replace=False)
        #     mushroom_dataset = Dataset.from_array(dataset['obs'][rand_idx], dataset['action'][rand_idx], dataset['reward'][rand_idx],
        #                       dataset['next_obs'][rand_idx], dataset['absorbing'][rand_idx], dataset['last'][rand_idx],
        #                       backend='torch')
        # else:
        #     mushroom_dataset = Dataset.from_array(dataset['obs'], dataset['action'], dataset['reward'],
        #                       dataset['next_obs'], dataset['absorbing'], dataset['last'],
        #                       backend='torch')
        # # copy it over to the replay buffer
        # # self._expert_replay_memory._initial_size = len(mushroom_dataset) # increase initial size (Needed?)
        # self._expert_replay_memory.add(mushroom_dataset)

        # no normalization for now
        # if self._normalize_states:
        #     self._compute_states_mean_std(self.pre_train_dataset['obs'])
        # if self._normalize_actions:
        #     self._compute_actions_mean_std(self.pre_train_dataset['action'])
    
    def fit(self, demo_dataset=None, n_epochs=None, pretrain_data=False, pre_train_maintain=False, pre_train_maintain_percent=0.0,
            use_high_level_actions=False, track_loss=True):
        # fit on the expert replay memory data for n_epochs
        if demo_dataset is None:
            demo_dataset = self.dataset
        if n_epochs is None:
            n_epochs = self._n_epochs_policy()
        if track_loss is True:
            acc_loss = []
        
        if pretrain_data is True and pre_train_maintain is False:
            # fit only the pre-train dataset
            pretrain_batch_size = self._batch_size()
            pre_train_batch_gen = minibatch_generator(pretrain_batch_size, self.pre_train_dataset['obs'], self.pre_train_dataset['action'])
            expert_replay_batch_size = 0
        elif pretrain_data is False and pre_train_maintain is False:
            # fit only on the expert replay memory
            pretrain_batch_size = 0
            expert_replay_batch_size = self._batch_size()
        elif pretrain_data is False and pre_train_maintain is True:
            # fit on both pre-train dataset and expert replay memory
            if not (0.0 < pre_train_maintain_percent < 1.0):
                raise ValueError("pre_train_maintain_percent must be between 0.0 and 1.0")
            pretrain_batch_size = int(self._batch_size() * pre_train_maintain_percent)
            pre_train_batch_gen = minibatch_generator(pretrain_batch_size, self.pre_train_dataset['obs'], self.pre_train_dataset['action'])
            expert_replay_batch_size = self._batch_size() - pretrain_batch_size
        else:
            raise ValueError("Invalid combination of pretrain_data and pre_train_maintain flags")
        
        epoch_count = 0
        for epoch_count in range(n_epochs):
            if expert_replay_batch_size > 0:
                # get batch from the expert replay memory
                if use_high_level_actions is True:
                    # get batch from the additional high-level action expert replay memory
                    obs, act, _, _, _, _ = self._additional_expert_replay_memory.get(expert_replay_batch_size)
                else:
                    # get batch from the regular expert replay memory
                    obs, act, _, _, _, _ = self._expert_replay_memory.get(expert_replay_batch_size)
            else:
                obs = np.zeros((0, self.mdp_info.observation_space.shape[0]))
                act = np.zeros((0, self.mdp_info.action_space.shape[0]))
            if pretrain_batch_size > 0:
                # get batch from pre-train dataset
                obs_pretrain, act_pretrain = next(pre_train_batch_gen)
                # stack the batches
                obs = np.vstack((obs, obs_pretrain))
                act = np.vstack((act, act_pretrain))
            
            # if self._normalize_states:
            #     state_fit = self._norm_states(obs)
            # else:
            state_fit = obs
            # if self._normalize_actions:
            #     act_fit = self._norm_actions(act)
            # else:
            act_fit = act
            loss = self._loss(state_fit, act_fit)
            self._optimize_actor_parameters(loss)
            if track_loss is True:
                # losses for logging
                acc_loss.append(loss.detach().cpu().numpy())

        if track_loss is True:
            # Store mean actor loss for logging
            self._actor_last_loss = np.mean(acc_loss)
    
    def _loss(self, state, act):
        # loss for behavior cloning
        
        # if isinstance(state, np.ndarray):
        #     state = torch.as_tensor(state, dtype=torch.float32)
        act = torch.as_tensor(act, dtype=torch.float32, device=TorchUtils.get_device())
        act_disc = act[:, :self._discrete_action_dims]
        act_cont = act[:, -self._continuous_action_dims:]

        act_pred = self._actor_approximator(state, **self._actor_predict_params)
        act_pred_disc = act_pred[:, :self._discrete_action_dims]
        act_pred_cont = act_pred[:, -self._continuous_action_dims:]
        if self._squash_actions:
            # Squash the continuous actions to [-1, 1] (Needed if RL policy squashes actions)
            act_pred_cont = torch.tanh(act_pred_cont)

        bc_loss = torch.zeros(1, device=TorchUtils.get_device())
        if self._discrete_action_dims > 0:
            # ensure targets are binary
            act_disc = (act_disc > 0.5).float()
            # treating discrete actions as logits. Use binary cross entropy loss
            act_pred_disc = torch.sigmoid(act_pred_disc)
            # bc_loss += binary_cross_entropy_with_logits(act_pred_disc, act_disc)
            bc_loss += torch.mean(-act_disc * torch.log(act_pred_disc + 1e-8) - (1 - act_disc) * torch.log(1 - act_pred_disc + 1e-8))
        if self._continuous_action_dims > 0:
            # Use mse loss for continuous actions
            bc_loss += torch.mean((act_pred_cont - act_cont)**2)

        return bc_loss
        
    # def _compute_states_mean_std(self, states: np.ndarray, eps: float = 1e-3):
    #     self._states_mean = states.mean(0)
    #     self._states_std = states.std(0) + eps

    #     # set them for the policy as well so that we use it when drawing actions
    #     self.policy._states_mean = self._states_mean
    #     self.policy._states_std = self._states_std

    # def _norm_states(self, states: np.ndarray):
    #     if self._states_mean is None or self._states_std is None:
    #         raise ValueError('States mean and std not computed yet. Call _compute_states_mean_std() on the dataset first.')
    #     return (states - self._states_mean) / self._states_std

    # def _compute_actions_mean_std(self, actions: np.ndarray, eps: float = 1e-3):
    #     self._actions_mean = actions.mean(0)
    #     self._actions_std = actions.std(0) + eps

    #     # set them for the policy as well so that we use it when drawing actions
    #     self.policy._actions_mean = self._actions_mean
    #     self.policy._actions_std = self._actions_std
    
    # def _norm_actions(self, actions: np.ndarray):
    #     if self._actions_mean is None or self._actions_std is None:
    #         raise ValueError('Actions mean and std not computed yet. Call _compute_actions_mean_std() on the dataset first.')
    #     return (actions - self._actions_mean) / self._actions_std
        
    def _post_load(self):
        self._actor_approximator = self.policy._approximator
        self._update_optimizer_parameters(self._actor_approximator.model.network.parameters())
