import torch
import numpy as np
from mushroom_rl.algorithms.actor_critic.deep_actor_critic.iql_dp import IQL_DP
from mushroom_rl.utils.torch import TorchUtils


class IDQL(IQL_DP):
    """
    IDQL (Implicit Diffusion Q-Learning) test-time action selection.
    
    This class extends IQL_DP to provide IDQL's categorical action selection at test time:
    - Samples N actions from a pretrained diffusion policy
    - Computes Q values for all N actions using the critic
    - Selects the action with the highest Q value (argmax)
    
    No training is performed - this is purely a test-time method.
    """
    
    def __init__(self, *args, idql_n_samples=64, **kwargs):
        """
        Initialize IDQL with configurable parameters.
        
        Args:
            *args: Arguments passed to IQL_DP.__init__
            idql_n_samples (int, 64): Number of action samples to generate per state
            **kwargs: Additional arguments passed to IQL_DP.__init__
        """
        super().__init__(*args, **kwargs)
        self._idql_n_samples = idql_n_samples
        
        # Override policy's draw_action and draw_deterministic_action to use IDQL
        self._override_policy_draw_methods()
    
    def _override_policy_draw_methods(self):
        """Override policy's draw_action and draw_deterministic_action to use IDQL."""
        def _convert_state_to_numpy(state):
            """Convert state to numpy array."""
            if isinstance(state, torch.Tensor):
                state_np = state.cpu().numpy()
            else:
                state_np = np.array(state)
            return state_np.flatten() if state_np.ndim > 1 else state_np
        
        def idql_wrapper(state, policy_state=None):
            """Wrapper that converts state and calls IDQL action selection."""
            state_np = _convert_state_to_numpy(state)
            action = self.draw_action_idql(state_np)
            return torch.as_tensor(action, dtype=torch.float32), None
        
        # Replace both methods with IDQL wrapper
        self.policy.draw_action = idql_wrapper
        self.policy.draw_deterministic_action = idql_wrapper
    
    def draw_action_idql(self, state, n_samples=None):
        """
        IDQL test-time action selection: sample N actions and select best based on Q values.
        
        Args:
            state: Current state observation (numpy array)
            n_samples (int, None): Number of action samples. Defaults to self._idql_n_samples
        
        Returns:
            Selected action (numpy array)
        """
        n_samples = n_samples or self._idql_n_samples
        device = TorchUtils.get_device()
        
        # Normalize state if needed
        if self._normalize_states:
            if self._states_mean is None or self._states_std is None:
                raise ValueError('States mean and std not computed yet.')
            state = self._norm_states(state)
        
        # Prepare state tensor
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        n_obs_steps = self.policy._n_obs_steps
        obs_dim = self.mdp_info.observation_space.shape[0]
        action_horizon = self.policy._horizon
        act_dim = self.mdp_info.action_space.shape[0]
        
        # Reshape state for diffusion policy
        if state_tensor.shape[1] == obs_dim:
            state_repeated = state_tensor.repeat(1, n_obs_steps).view(1, n_obs_steps, obs_dim)
        elif state_tensor.shape[1] == n_obs_steps * obs_dim:
            state_repeated = state_tensor.view(1, n_obs_steps, obs_dim)
        else:
            raise ValueError(f'Unexpected state shape: {state_tensor.shape}')
        
        # Repeat for N samples
        state_batch = state_repeated.repeat(n_samples, 1, 1)
        
        # Sample N actions from pretrained diffusion policy
        with torch.no_grad():
            actions = self.policy._model.generate_actions({'observation.state': state_batch})
            actions_flat = actions.view(n_samples, action_horizon * act_dim)
            
            # Compute Q values and select best
            q_values = self._critic_approximator.predict(
                state_batch.view(n_samples, -1),
                actions_flat,
                prediction='min',
                **self._critic_predict_params
            )
            
            best_idx = torch.argmax(q_values)
            return actions[best_idx][0].cpu().numpy()

