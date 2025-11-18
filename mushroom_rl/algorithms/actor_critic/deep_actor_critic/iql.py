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

class IQL(DeepAC):
    """
    IQL Offline-to-Online RL algorithm.
    "Offline Reinforcement Learning with Implicit Q-Learning".
    Kostrikov I. et al.. 2022.
    Reference implementation: https://github.com/corl-team/CORL.
    Modified to also supports hybrid policies (both discrete and continuous actions).

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
        self._actor_predict_params = dict() if actor_predict_params is None else actor_predict_params
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

        self._actor_approximator = Regressor(TorchApproximator, **actor_params)

        self._init_target(self._critic_approximator, self._target_critic_approximator)

        policy = policy_class(self._actor_approximator, **policy_params)

        policy_parameters = self._actor_approximator.model.network.parameters()

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

        self._batch_size = to_parameter(batch_size)
        self._tau = to_parameter(tau)
        self._fit_count = 0
        self._actor_last_loss = None # Store actor loss for logging
        self._value_last_loss = None # Store value loss for logging
        self._last_exp_adv = None # Store exp_adv for logging

        self._replay_memory = ReplayMemory(mdp_info, self.info, initial_replay_size, max_replay_size)

        self._squash_actions = squash_actions
        self._discrete_action_dims = discrete_action_dims
        self._continuous_action_dims = continuous_action_dims
        self._normalize_states = normalize_states
        self._states_mean = None
        self._states_std = None
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
            _actor_predict_params='pickle',
            _batch_size='mushroom',
            _tau='mushroom',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _value_func_approximator='mushroom',
            _value_func_optimizer='torch',
            _actor_approximator='mushroom',
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
            # if rewards, absorbings and last are not single dimensions, squeeze them
            for k in ['reward', 'absorbing', 'last']:
                dataset[k] = dataset[k].squeeze(1) if dataset[k].ndim > 1 else dataset[k]
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
        
        # copy over offline dataset to the replay buffer
        self._replay_memory._initial_size = len(self.offline_dataset) # set initial size to the size of the offline dataset
        if self._replay_memory._max_size < len(self.offline_dataset):
            print('[[Warning: Offline dataset size exceeds max replay memory size. Resizing replay memory to fit dataset.]]')
            self._replay_memory = ReplayMemory(self.mdp_info, self.info, len(self.offline_dataset), len(self.offline_dataset))
        self._replay_memory.add(self.offline_dataset)
    
    def offline_fit(self, n_epochs, fit_critic=True, fit_actor=True):
        if self.offline_dataset is None:
            raise ValueError('No offline dataset loaded!. Call load_dataset() first.')
        
        # fit on the dataset (for n_epochs)
        for epoch in trange(n_epochs):
            state, action, reward, next_state, absorbing, _ = self._replay_memory.get(self._batch_size())

            if self._normalize_states:
                state_fit = self._norm_states(state)
                next_state_fit = self._norm_states(next_state)
            else:
                state_fit = state
                next_state_fit = next_state

            self.iql_fit(state_fit, action, reward, next_state_fit, absorbing, fit_critic, fit_actor)
    
    def fit(self, dataset, fit_critic=True, fit_actor=True): # Online
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = self._replay_memory.get(self._batch_size())
            if self._normalize_states:
                state_fit = self._norm_states(state)
                next_state_fit = self._norm_states(next_state)
            else:
                state_fit = state
                next_state_fit = next_state
            
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
            assert fit_critic != fit_actor, 'fit_critic and fit_actor cannot be True at the same time for DDPG_BC'
            if fit_critic:
                with torch.no_grad():
                    next_v = self._value_func_approximator(next_state, **self._critic_predict_params)
                    next_v = next_v.cpu()
                # Get advantage & update value function (if fit_critic)
                adv = self._get_adv_and_update_v(state, action, fit_critic)
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
        q = reward + (~absorbing) * self.mdp_info.gamma * next_v

        # Fit critic
        self._critic_approximator.fit(state, action, q, **self._critic_fit_params)      

        # Update target critic
        self._update_target(self._critic_approximator, self._target_critic_approximator)

    def _update_actor_awr(self, adv, state, action):
        # compute advantage weighted BC loss
        exp_adv = torch.exp(self._iql_beta() * adv.detach()).clamp(max=self._max_clamp_adv)
        
        # target action from data:
        act = torch.as_tensor(action, dtype=torch.float32, device=TorchUtils.get_device())
        act_disc = act[:, :self._discrete_action_dims]
        act_cont = act[:, -self._continuous_action_dims:]

        act_pred = self._actor_approximator(state, **self._actor_predict_params)
        act_pred_disc = act_pred[:, :self._discrete_action_dims]
        act_pred_cont = act_pred[:, -self._continuous_action_dims:]
        if self._squash_actions:
            # Squash the continuous actions to [-1, 1] (Needed if RL policy squashes actions)
            act_pred_cont = torch.tanh(act_pred_cont)
        
        bc_loss = torch.zeros(act.shape[0], device=TorchUtils.get_device())
        if self._discrete_action_dims > 0:
            # ensure targets are binary
            act_disc = (act_disc > 0.5).float()
            # treating discrete actions as logits. Use binary cross entropy loss
            act_pred_disc = torch.sigmoid(act_pred_disc)
            # bc_loss += binary_cross_entropy_with_logits(act_pred_disc, act_disc)
            bc_loss += (-act_disc * torch.log(act_pred_disc + 1e-8) - (1 - act_disc) * torch.log(1 - act_pred_disc + 1e-8)).mean(1)
        if self._continuous_action_dims > 0:
            # Use mse loss for continuous actions
            bc_loss += torch.mean((act_pred_cont - act_cont)**2, dim=1)
        actor_loss = torch.mean(exp_adv * bc_loss)

        self._optimize_actor_parameters(actor_loss)
        
        if self._schedule_actor_lr:
            self._actor_lr_scheduler.step()

        self._actor_last_loss = actor_loss.detach().cpu().numpy() # Store actor loss for logging
        self._last_exp_adv = exp_adv.detach().mean().cpu().numpy() # Store exp_adv for logging

    def _update_actor_ddpg_plus_bc(self, state, action):
        # target action from data:
        act = torch.as_tensor(action, dtype=torch.float32, device=TorchUtils.get_device())
        act_disc = act[:, :self._discrete_action_dims]
        act_cont = act[:, -self._continuous_action_dims:]

        act_pred = self._actor_approximator(state, **self._actor_predict_params)
        act_pred_disc = act_pred[:, :self._discrete_action_dims]
        act_pred_cont = act_pred[:, -self._continuous_action_dims:]
        if self._squash_actions:
            # Squash the continuous actions to [-1, 1] (Needed if RL policy squashes actions)
            act_pred_cont = torch.tanh(act_pred_cont)
        
        # DDPG loss
        q = self._critic_approximator(state, act_pred, **self._critic_predict_params)
        q_loss = -q

        # BC loss
        bc_loss = torch.zeros(act.shape[0], device=TorchUtils.get_device())
        if self._discrete_action_dims > 0:
            # ensure targets are binary
            act_disc = (act_disc > 0.5).float()
            # treating discrete actions as logits. Use binary cross entropy loss
            act_pred_disc = torch.sigmoid(act_pred_disc)
            # bc_loss += binary_cross_entropy_with_logits(act_pred_disc, act_disc)
            bc_loss += (-act_disc * torch.log(act_pred_disc + 1e-8) - (1 - act_disc) * torch.log(1 - act_pred_disc + 1e-8)).mean(1)
        if self._continuous_action_dims > 0:
            # Use mse loss for continuous actions
            bc_loss += torch.mean((act_pred_cont - act_cont)**2, dim=1)
        bc_loss = bc_loss

        # Total loss
        actor_loss = torch.mean(q_loss + self._bc_weight_in_ddpg() * bc_loss)

        self._optimize_actor_parameters(actor_loss)
        
        if self._schedule_actor_lr:
            self._actor_lr_scheduler.step()

        self._actor_last_loss = actor_loss.detach().cpu().numpy() # Store actor loss for logging

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
