import torch
import numpy as np
# from mushroom_rl.algorithms.actor_critic.deep_actor_critic import TD3
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

class TD3_BC(DeepAC):
    """
    TD3 + BC Offline RL algorithm.
    Modified to add option of offline-to-online RL training.
    "A Minimalist Approach to Offline Reinforcement Learning".
    Fujimoto S. et al.. 2021.

    """
    def __init__(self, mdp_info, policy_class, policy_params,
                 actor_params, actor_optimizer, critic_params,
                 batch_size, initial_replay_size, max_replay_size, tau, policy_delay=2,
                 noise_std=.2, noise_clip=.5, squash_actions=False, normalize_states=False,
                 offline_alpha=0.25, online_alpha=0.0,
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
            batch_size ([int, Parameter]): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            tau ([float, Parameter]): value of coefficient for soft updates;
            policy_delay ([int, Parameter], 2): the number of updates of the critic after
                which an actor update is implemented;
            noise_std ([float, Parameter], .2): standard deviation of the noise used for
                policy smoothing;
            noise_clip ([float, Parameter], .5): maximum absolute value for policy smoothing
                noise;
            squash_actions (bool, False): whether to squash the actions to [-1, 1] with tanh;
            normalize_states (bool, False): whether to normalize states;
            offline_alpha ([float, Parameter], 0.25): weight of the BC loss when fitting on the offline dataset;
            online_alpha ([float, Parameter], 0.25): weight of the offline BC loss during online training;
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

        target_actor_params = deepcopy(actor_params)
        self._actor_approximator = Regressor(TorchApproximator, **actor_params)
        self._target_actor_approximator = Regressor(TorchApproximator, **target_actor_params)

        self._init_target(self._critic_approximator, self._target_critic_approximator)
        self._init_target(self._actor_approximator, self._target_actor_approximator)

        policy = policy_class(self._actor_approximator, **policy_params)

        policy_parameters = self._actor_approximator.model.network.parameters()

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

        self._batch_size = to_parameter(batch_size)
        self._tau = to_parameter(tau)
        self._policy_delay = to_parameter(policy_delay)
        self._fit_count = 0
        self._actor_last_loss = None # Store actor loss for logging

        self._replay_memory = ReplayMemory(mdp_info, self.info, initial_replay_size, max_replay_size)


        self._noise_std = to_parameter(noise_std)
        self._noise_clip = to_parameter(noise_clip)
        self._squash_actions = squash_actions
        self._normalize_states = normalize_states
        self._states_mean = None
        self._states_std = None



        self._offline_alpha = to_parameter(offline_alpha)
        self._online_alpha = to_parameter(online_alpha)
        self._bc_loss_fn = torch.nn.MSELoss()
        self.offline_dataset = None

        self._add_save_attr(
            _critic_fit_params='pickle',
            _critic_predict_params='pickle',
            _actor_predict_params='pickle',
            _batch_size='mushroom',
            _tau='mushroom',
            _policy_delay='mushroom',
            _fit_count='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _target_actor_approximator='mushroom',
            _noise_std='mushroom',
            _noise_clip='mushroom',
            _squash_actions='primitive',
            _normalize_states='primitive',
            _states_mean='primitive',
            _states_std='primitive',
            _offline_alpha='mushroom',
            _online_alpha='mushroom'
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
        
        # copy it over to the replay buffer
        self._replay_memory._initial_size = len(self.offline_dataset) # set initial size to the size of the offline dataset
        if self._replay_memory._max_size < len(self.offline_dataset):
            print('[[Warning: Offline dataset size exceeds max replay memory size. Resizing replay memory to fit dataset.]]')
            self._replay_memory = ReplayMemory(self.mdp_info, self.info, len(self.offline_dataset), len(self.offline_dataset))
        self._replay_memory.add(self.offline_dataset)

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.

        """
        a = self._target_actor_approximator(next_state, **self._actor_predict_params)
        if self._squash_actions:
            a = torch.tanh(a) # squash action to [-1, 1]
        a = a.detach().cpu().numpy() # TODO: Handle without casting to numpy

        low = self.mdp_info.action_space.low
        high = self.mdp_info.action_space.high
        eps = np.random.normal(scale=self._noise_std(), size=a.shape)
        eps_clipped = np.clip(eps, -self._noise_clip(), self._noise_clip.get_value())
        a_smoothed = np.clip(a + eps_clipped, low, high)

        q = self._target_critic_approximator.predict(next_state, a_smoothed,
                                                     prediction='min', **self._critic_predict_params)
        q = q.detach().cpu()
        q *= (~absorbing)

        return q
    
    def offline_fit(self, n_epochs):
        if self.offline_dataset is None:
            raise ValueError('No offline dataset loaded!. Call load_dataset() first.')
        
        # fit on the dataset (for n_epochs)
        for epoch in trange(n_epochs):
            obs, action, reward, next_obs, absorbing, _ = self._replay_memory.get(self._batch_size())
            if self._normalize_states:
                state = self._norm_states(obs)
                next_state = self._norm_states(next_obs)
            else:
                state = obs
                next_state = next_obs

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q, **self._critic_fit_params)

            if self._fit_count % self._policy_delay() == 0:
                loss = self._offline_loss(state, action)
                self._optimize_actor_parameters(loss)

            self._update_target(self._critic_approximator, self._target_critic_approximator)
            self._update_target(self._actor_approximator, self._target_actor_approximator)

            self._fit_count += 1
        
        self._actor_last_loss = loss.detach().cpu().numpy() # Store actor loss for logging
    
    def _offline_loss(self, state, action):
        # Offline loss
        td3_actor_loss, action_pred = self._td3_actor_loss(state, norm_q=True, return_pred_action=True)
        bc_loss = self._bc_loss(state, action, action_pred)

        return td3_actor_loss + self._offline_alpha() * bc_loss

    def fit(self, dataset): # Online
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = self._replay_memory.get(self._batch_size())

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q, **self._critic_fit_params)

            if self._fit_count % self._policy_delay() == 0:
                loss = self._online_loss(state)
                self._optimize_actor_parameters(loss)
                self._actor_last_loss = loss.detach().cpu().numpy() # Store actor loss for logging

            self._update_target(self._critic_approximator, self._target_critic_approximator)
            self._update_target(self._actor_approximator, self._target_actor_approximator)

            self._fit_count += 1
    
    def _online_loss(self, state):
        # Online loss
        td3_actor_loss = self._td3_actor_loss(state, norm_q=True, return_pred_action=False)
        if self._online_alpha() > 0:
            # Use both TD3 loss on the online data and BC loss on offline data
            # sample batch from offline dataset
            sample = self.offline_dataset.select_random_samples(self._batch_size())
            offline_state, offline_action, _, _, _, _ = sample.parse()
            bc_loss = self._bc_loss(offline_state, offline_action)

            return td3_actor_loss + self._online_alpha() * bc_loss
        else:
            # Use only TD3 loss on the online data
            return td3_actor_loss
    
    def _td3_actor_loss(self, state, norm_q=True, return_pred_action=False):
        # Online loss
        action = self._actor_approximator(state, **self._actor_predict_params)
        if self._squash_actions:
            action = torch.tanh(action) # squash action to [-1, 1]
        q = self._critic_approximator(state, action, idx=0, **self._critic_predict_params)
        loss = -q.mean()
        
        if norm_q:
            loss /= q.abs().mean().detach()
        
        if return_pred_action:
            return loss, action
        else:
            return loss
    
    def _bc_loss(self, obs, act, act_pred=None):
        # loss for behavior cloning
        act = torch.as_tensor(act, dtype=torch.float32, device=TorchUtils.get_device())
        if act_pred is None:
            # TODO: Handle hybrid policy
            act_pred = self._actor_approximator(obs, **self._actor_predict_params)
            if self._squash_actions:
                # Squash the actions to [-1, 1] (Needed if RL policy squashes actions)
                act_pred = torch.tanh(act_pred)

        return self._bc_loss_fn(act_pred, act) # normally mse loss
    
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
