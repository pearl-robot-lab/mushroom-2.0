import torch
import numpy as np

from .policy import ParametricPolicy


class OrnsteinUhlenbeckPolicy(ParametricPolicy):
    """
    Ornstein-Uhlenbeck process as implemented in:
    https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py.

    This policy is commonly used in the Deep Deterministic Policy Gradient algorithm.

    """
    def __init__(self, mu, sigma, theta, dt, x0=None):
        """
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean w.r.t. the state;
            sigma (torch.tensor): average magnitude of the random fluctations per square-root time;
            theta (float): rate of mean reversion;
            dt (float): time interval;
            x0 (torch.tensor, None): initial values of noise.

        """
        self._approximator = mu
        self._predict_params = dict()
        self._sigma = sigma
        self._theta = theta
        self._dt = dt
        self._x0 = x0
        self._x_prev = None

        self.reset()

        self._add_save_attr(
            _approximator='mushroom',
            _predict_params='pickle',
            _sigma='torch',
            _theta='primitive',
            _dt='primitive',
            _x0='torch'
        )

        super().__init__(self._approximator.output_shape)

    def __call__(self, state, action=None, policy_state=None):
        raise NotImplementedError

    def draw_action(self, state, policy_state):
        with torch.no_grad():
            mu = self._approximator.predict(state, **self._predict_params).cpu()
            sqrt_dt = np.sqrt(self._dt)

            x = policy_state - self._theta * policy_state * self._dt +\
                self._sigma * sqrt_dt * torch.randn(size=self._approximator.output_shape)

            return mu + x, x

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        return self._approximator.weights_size

    def reset(self):
        return self._x0 if self._x0 is not None else torch.zeros(self._approximator.output_shape)


class ClippedGaussianPolicy(ParametricPolicy):
    """
    Clipped Gaussian policy, as used in:

    "Addressing Function Approximation Error in Actor-Critic Methods".
    Fujimoto S. et al.. 2018.

    This is a non-differentiable policy for continuous action spaces.
    The policy samples an action in every state following a gaussian distribution, where the mean is computed in the
    state and the covariance matrix is fixed. The action is then clipped using the given action range.
    This policy is not a truncated Gaussian, as it simply clips the action if the value is bigger than the boundaries.
    Thus, the non-differentiability.

    """
    def __init__(self, mu, sigma, low, high, policy_state_shape=None, 
                 draw_random_act=False, draw_deterministic=False,
                 squash_actions=False, discrete_action_dims=0, continuous_action_dims=0,
                 normalize_states=False, normalize_actions=False):
        """
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean w.r.t. the state;
            sigma (torch.tensor): a square positive definite matrix representing the covariance matrix. The size of this
                matrix must be n x n, where n is the action dimensionality;
            low (torch.tensor): a vector containing the minimum action for each component;
            high (torch.tensor): a vector containing the maximum action for each component.
            draw_random_act (bool, False): if True, the policy will draw random actions.
            draw_deterministic (bool, False): if True, the policy will draw deterministic actions.
            squash_actions (bool, False): if True, the actions will be squashed to [-1, 1] with a tanh function.
            discrete_action_dims (int, 0): the number of discrete actions in the action space.
            continuous_action_dims (int, 0): the number of continuous actions in the action space.
            normalize_states (bool, False): if True, the states will be normalized before being passed to the regressor.
            normalize_actions (bool, False): if True, the actions will be normalized before being returned.

        """
        super().__init__(policy_state_shape)

        self._approximator = mu
        self._predict_params = dict()
        self._chol_sigma = torch.linalg.cholesky(sigma)
        self._low = torch.as_tensor(low)
        self._high = torch.as_tensor(high)
        self._draw_random_act = draw_random_act
        self._draw_deterministic = draw_deterministic
        self._squash_actions = squash_actions
        self._discrete_action_dims = discrete_action_dims
        self._continuous_action_dims = continuous_action_dims
        self._normalize_states = normalize_states
        self._states_mean = None # will be set by the agent class
        self._states_std = None # will be set by the agent class
        self._normalize_actions = normalize_actions
        self._actions_mean = None # will be set by the agent class
        self._actions_std = None # will be set by the agent class

        # debug
        self.debug_replay_states = None
        self.debug_replay_actions = None
        self.debug_replay_index = 0
        self.debug_action_diffs = []

        self._add_save_attr(
            _approximator='mushroom',
            _predict_params='pickle',
            _chol_sigma='torch',
            _low='torch',
            _high='torch',
            _draw_random_act='primitive',
            _draw_deterministic='primitive',
            _squash_actions='primitive',
            _discrete_action_dims='primitive',
            _continuous_action_dims='primitive',
            _normalize_states='primitive',
            _states_mean='primitive',
            _states_std='primitive',
            _normalize_actions='primitive',
            _actions_mean='primitive',
            _actions_std='primitive',
            debug_replay_states='primitive',
            debug_replay_actions='primitive',
            debug_replay_index='primitive',
            debug_action_diffs='primitive'
        )

    def __call__(self, state, action=None, policy_state=None):
        raise NotImplementedError

    def draw_action(self, state, policy_state=None):
        
        if self._draw_random_act is True:
            return self.draw_random_action()
        elif self._draw_deterministic is True:
            return self.draw_deterministic_action(state, policy_state)
        
        with torch.no_grad():
            if self._normalize_states:
                if self._states_mean is None:
                    raise ValueError('States mean is not set by the agent class')
                state_query = (state - self._states_mean) / self._states_std
            else:
                state_query = state
                
            mu = self._approximator.predict(state_query, **self._predict_params).cpu()
            # mu = np.reshape(self._approximator.predict(np.expand_dims(state_query, axis=0), **self._predict_params), -1)
            if self._squash_actions:
                # Squash the continuous actions to [-1, 1]
                mu[-self._continuous_action_dims:] = torch.tanh(mu[-self._continuous_action_dims:])

            if self._normalize_actions:
                if self._actions_mean is None:
                    raise ValueError('Actions mean is not set by the agent class')
                mu = mu * self._actions_std + self._actions_mean

            # sample continuous actions from distribution
            distribution = torch.distributions.MultivariateNormal(loc=mu[-self._continuous_action_dims:], scale_tril=self._chol_sigma,
                                                                  validate_args=False)
            action_raw = distribution.sample()

            if self._discrete_action_dims > 0:
                # discrete actions from network are logits, so sigmoid them
                action_disc = torch.sigmoid(mu[:self._discrete_action_dims])
                action = torch.cat((action_disc, action_raw), dim=0)
                # print("action: ", torch.round(action*100)/100)
            else:
                action = action_raw

            return torch.clip(action, self._low, self._high), None

    
    def draw_random_action(self):
        return torch.rand(self._low.shape) * (self._high - self._low) + self._low, None

    def draw_deterministic_action(self, state, policy_state=None):
        # fix if loading is not on the correct device
        if self._low.device != state.device:
            self._low = self._low.to(state.device)
            self._high = self._high.to(state.device)
            self._chol_sigma = self._chol_sigma.to(state.device)
        with torch.no_grad():
            ## Debug for distribution shift...
            if self.debug_replay_states is not None:
            # self._high = self._high.to(state.device)
            # self._low = self._low.to(state.device)
                # Use state from replay buffer to induce the same observation distribution
                # to test the actions from the network
                state_query = torch.tensor(self.debug_replay_states[self.debug_replay_index])
                if self.debug_replay_actions is None:
                    self.debug_replay_index += 1
                if self._normalize_states:
                    if self._states_mean is None:
                        raise ValueError('States mean is not set by the agent class')
                    state_query = (state_query - self._states_mean) / self._states_std
            ## Debug end
            else:
                if self._normalize_states:
                    if self._states_mean is None:
                        raise ValueError('States mean is not set by the agent class')
                    state_query = (state - self._states_mean) / self._states_std
                else:
                    state_query = state
            
            mu = self._approximator.predict(state_query, **self._predict_params).cpu()
            # mu = np.reshape(self._approximator.predict(np.expand_dims(state_query, axis=0), **self._predict_params), -1)
            
            if self._squash_actions:
                # Squash the continuous actions to [-1, 1]
                mu[-self._continuous_action_dims:] = torch.tanh(mu[-self._continuous_action_dims:])

            if self._normalize_actions:
                if self._actions_mean is None:
                    raise ValueError('Actions mean is not set by the agent class')
                mu = mu * self._actions_std + self._actions_mean
            
            action_raw = mu[-self._continuous_action_dims:]

            if self._discrete_action_dims > 0:
                # discrete actions from network are logits, so sigmoid them
                action_disc = torch.sigmoid(mu[:self._discrete_action_dims])
                action = torch.cat((action_disc, action_raw), dim=0)
                # print("action: ", torch.round(action*100)/100)
            else:
                action = action_raw

            ## Debug for distribution shift...
            if self.debug_replay_actions is not None:
                next_replay_action = torch.tensor(self.debug_replay_actions[self.debug_replay_index])
                self.debug_replay_index += 1

                # Check if the network action is the same as the replay action
                action_diff = torch.abs(action - next_replay_action)
                self.debug_action_diffs.append(action_diff)

                # Optional: Take the replay action instead of the network action to induce the same observation distribution
                action = next_replay_action
            ## Debug end

            action_clipped = torch.clip(action, self._low, self._high)
            
            return action_clipped, None
    
    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        return self._approximator.weights_size
