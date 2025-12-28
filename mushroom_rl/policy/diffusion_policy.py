import torch
import numpy as np
from pathlib import Path
from .policy import ParametricPolicy
from collections import deque
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from mushroom_rl.utils.torch import TorchUtils

from safetensors.torch import save_model as save_model_as_safetensor
from safetensors.torch import load_model as load_model_as_safetensor

# from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
# from lerobot.common.policies.normalize import Normalize, Unnormalize
# from lerobot.common.policies.pretrained import PreTrainedPolicy
# from lerobot.common.policies.utils import populate_queues

def populate_queues(queues, batch):
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


class DiffusionPolicy(ParametricPolicy):
    """
    Diffusion-based policy, as used in:

    "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion".
    Chi et al.. 2023.
    Code reference: https://github.com/huggingface/lerobot

    """
    def __init__(self, policy_params, policy_state_shape=None, 
                 draw_random_act=False, draw_deterministic=False,
                 squash_actions=False, normalize_states=False, normalize_actions=False):
        """
        Constructor.

        Args:
            policy_params (dict): config parameters of the diffusion policy.
            draw_random_act (bool, False): if True, the policy will draw random actions.
            draw_deterministic (bool, False): if True, the policy will draw deterministic actions.
            squash_actions (bool, False): if True, the actions will be squashed to [-1, 1] with a tanh function.
            normalize_states (bool, False): if True, the states will be normalized before being passed to the regressor.
            normalize_actions (bool, False): if True, the actions will be normalized before being returned.

        """
        super().__init__(policy_state_shape) # TODO: Needed?

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None
        
        self._n_obs_steps = policy_params['n_obs_steps']
        self._horizon = policy_params['horizon']
        self._n_action_steps = policy_params['n_action_steps']
        self._temp_ensemble = policy_params['temp_ensemble']
        if self._temp_ensemble is True:
            # full_episode_actions dict for temporal ensembling
            self._full_episode_actions = {}
            self._episode_step = 0
            assert (self._horizon-(self._n_obs_steps-1)) % self._n_action_steps == 0 , \
                  "Config error: For temporal ensembling, the horizon minus the number of observation steps must be divisible by the number of action steps"
            self._num_ensembles = int( (self._horizon-(self._n_obs_steps-1)) / self._n_action_steps )
            ensem_weights = torch.exp(- 0.5 * torch.arange(self._num_ensembles, device=TorchUtils.get_device()) )
            self._ensem_weights = ensem_weights / torch.sum(ensem_weights)

        self._image_features = policy_params['image_features']
        self._env_state_feature = policy_params['env_state_feature']
        model_config = policy_params # just use the same
        self._model = policy_params['model_class'](model_config)
        # move model to correct device
        self._model.to(TorchUtils.get_device())
        
        self.reset()

        # self._approximator = mu
        # self._predict_params = dict()
        # self._chol_sigma = torch.linalg.cholesky(sigma)
        self._low = torch.as_tensor(policy_params['low'])
        self._high = torch.as_tensor(policy_params['high'])
        self._draw_random_act = draw_random_act # unused for now
        self._draw_deterministic = draw_deterministic # unused for now
        self._squash_actions = policy_params['squash_actions']
        self._normalize_states = policy_params['normalize_states']
        self._states_mean = None # will be set by the agent class
        self._states_std = None # will be set by the agent class
        self._normalize_actions = policy_params['normalize_actions']
        self._actions_mean = None # will be set by the agent class
        self._actions_std = None # will be set by the agent class

        # debug
        self.debug_replay_states = None
        self.debug_replay_actions = None
        self.debug_replay_index = 0
        self.debug_action_diffs = []

        self._add_save_attr(
            # _model='mushroom', # loaded separately with safetensors
            _n_obs_steps='primitive',
            _horizon='primitive',
            _n_action_steps='primitive',
            _temp_ensemble='primitive',
            _image_features='primitive',
            _env_state_feature='primitive',
            _low='torch',
            _high='torch',
            _draw_random_act='primitive',
            _draw_deterministic='primitive',
            _squash_actions='primitive',
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
            # _predict_params='pickle', # deprecated
            # _chol_sigma='torch', # deprecated
        )

    def __call__(self, state, action=None, policy_state=None):
        raise NotImplementedError

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self._n_obs_steps),
            "action": deque(maxlen=self._n_action_steps),
        }
        if self._image_features:
            raise NotImplementedError("Image features are not implemented yet")
            self._queues["observation.images"] = deque(maxlen=self._n_obs_steps)
        if self._env_state_feature:
            raise NotImplementedError("Environment state feature is not implemented yet")
            self._queues["observation.environment_state"] = deque(maxlen=self._n_obs_steps)
        if self._temp_ensemble is True:
            # clear full_episode_actions dict for the next episode
            self._full_episode_actions = {}
            self._episode_step = 0

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        # batch = self.normalize_inputs(batch) # Assuming inputs are normalized already if needed
        if self._image_features:
            raise NotImplementedError("Image features are not implemented yet")
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self._image_features], dim=-4
            )
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self._model.generate_actions(batch)

            # TODO(rcadene): make above methods return output dictionary?
            # actions = self.unnormalize_outputs({"action": actions})["action"]
            # TODO: check if action normalization/unnormalization is needed

            # Extract `n_action_steps` steps worth of actions (from the current observation).
            start = self._n_obs_steps - 1
            end = start + self._n_action_steps
            actions_queued = actions[:, start:end].clone()

            if self._temp_ensemble is True:
                # Add current actions to the full_episode_actions dict
                curr_step = self._episode_step
                for act in actions[0, start:]:
                    if curr_step in self._full_episode_actions:
                        self._full_episode_actions[curr_step] = torch.vstack((act.unsqueeze(0), self._full_episode_actions[curr_step])) # NOTE: Make sure you put new actions on top
                    else:
                        self._full_episode_actions[curr_step] = act.unsqueeze(0)
                    curr_step += 1
                
                # get weighted ensembled actions for n_action_steps
                for step in range(self._episode_step, self._episode_step+self._n_action_steps):
                    if len(self._full_episode_actions[step]) < self._num_ensembles:
                        # don't emsemble if not enough actions are available
                        break
                    else:
                        actions_queued[0, step-self._episode_step] = torch.sum(self._full_episode_actions[step] * self._ensem_weights.unsqueeze(1), dim=0)

                # set self._episode_step to the next expected step in this call
                self._episode_step += self._n_action_steps

            self._queues["action"].extend(actions_queued.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def forward(self, batch: dict[str, Tensor], squash_actions: bool) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        # batch = self.normalize_inputs(batch) # Assuming inputs are normalized already if needed
        if self._image_features:
            raise NotImplementedError("Image features are not implemented yet")
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self._image_features], dim=-4
            )
        # batch = self.normalize_targets(batch)
        # TODO: check if action normalization/unnormalization is needed
        act_pred, loss = self._model.compute_loss(batch, squash_actions)
        return {"act_pred": act_pred, "loss": loss}

    def draw_action(self, state, policy_state=None):
        
        if state.type == 'numpy':
            state = torch.tensor(state)
        
        if self._draw_random_act is True:
            return self.draw_random_action()
        elif self._draw_deterministic is True:
            return self.draw_deterministic_action(state, policy_state)
        
        with torch.no_grad():
            if self._normalize_states:
                if self._states_mean is None:
                    raise ValueError('States mean is not set by the agent class')
                if self._states_mean.device != state.device:
                    self._states_mean = self._states_mean.to(state.device)
                    self._states_std = self._states_std.to(state.device)
                state_query = (state - self._states_mean) / self._states_std
            else:
                state_query = state
                
            input_batch = { # batch_size = 1
                'observation.state': state_query.unsqueeze(0).to(TorchUtils.get_device())
                }
            mu = self.select_action(input_batch).cpu()[0]
            # mu = self._model.predict(state_query, **self._predict_params).cpu()
            # mu = np.reshape(self._approximator.predict(np.expand_dims(state_query, axis=0), **self._predict_params), -1)
            if self._squash_actions:
                # Squash the continuous actions to [-1, 1]
                mu = torch.tanh(mu)

            if self._normalize_actions:
                if self._actions_mean is None:
                    raise ValueError('Actions mean is not set by the agent class')
                if self._actions_mean.device != mu.device:
                    self._actions_mean = self._actions_mean.to(mu.device)
                    self._actions_std = self._actions_std.to(mu.device)
                mu = mu * self._actions_std + self._actions_mean

            # sample continuous actions from distribution
            distribution = torch.distributions.MultivariateNormal(loc=mu, scale_tril=self._chol_sigma,
                                                                  validate_args=False)
            action_raw = distribution.sample()

            # if self._discrete_action_dims > 0:
            #     # discrete actions from network are logits, so sigmoid them
            #     action_disc = torch.sigmoid(mu[:self._discrete_action_dims])
            #     action = torch.cat((action_disc, action_raw), dim=0)
            #     # print("action: ", torch.round(action*100)/100)
            # else:
            action = action_raw

            return torch.clip(action, self._low, self._high), None
    
    def draw_random_action(self):
        return torch.rand(self._low.shape) * (self._high - self._low) + self._low, None

    def draw_deterministic_action(self, state, policy_state=None):
        # fix if loading is not on the correct device
        if self._low.device != state.device:
            self._low = self._low.to(state.device)
            self._high = self._high.to(state.device)
            # self._chol_sigma = self._chol_sigma.to(state.device)
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
                    if self._states_mean.device != state_query.device:
                        self._states_mean = self._states_mean.to(state_query.device)
                        self._states_std = self._states_std.to(state_query.device)
                    state_query = (state_query - self._states_mean) / self._states_std
            ## Debug end
            else:
                if self._normalize_states:
                    if self._states_mean is None:
                        raise ValueError('States mean is not set by the agent class')
                    if self._states_mean.device != state.device:
                        self._states_mean = self._states_mean.to(state.device)
                        self._states_std = self._states_std.to(state.device)
                    state_query = (state - self._states_mean) / self._states_std
                else:
                    state_query = state
            
            input_batch = { # batch_size = 1
                'observation.state': state_query.unsqueeze(0).to(TorchUtils.get_device())
                }
            mu = self.select_action(input_batch).cpu()[0]
            # mu = self._model.predict(state_query, **self._predict_params).cpu()
            
            if self._squash_actions:
                # Squash the continuous actions to [-1, 1]
                mu = torch.tanh(mu)

            if self._normalize_actions:
                if self._actions_mean is None:
                    raise ValueError('Actions mean is not set by the agent class')
                mu = mu * self._actions_std + self._actions_mean
            
            action_raw = mu

            # if self._discrete_action_dims > 0:
            #     # discrete actions from network are logits, so sigmoid them
            #     action_disc = torch.sigmoid(mu[:self._discrete_action_dims])
            #     action = torch.cat((action_disc, action_raw), dim=0)
            #     # print("action: ", torch.round(action*100)/100)
            # else:
            action = action_raw

            ## Debug for distribution shift...
            if self.debug_replay_actions is not None:
                next_replay_action = torch.tensor(self.debug_replay_actions[self.debug_replay_index])
                self.debug_replay_index += 1

                # Check if the network action is the same as the replay action
                action_diff = torch.mean(torch.abs(action - next_replay_action))
                self.debug_action_diffs.append(action_diff)

                # Optional: Take the replay action instead of the network action to induce the same observation distribution
                action = next_replay_action
            ## Debug end

            action_clipped = torch.clip(action, self._low, self._high)
            
            return action_clipped, None
    
    def save_model(self, save_path: Path) -> None:
        save_model_as_safetensor(self._model, str(save_path))

    def load_model(self, model_config: dict, model_path: Path) -> None:
        self._model = model_config['model_class'](model_config)
        print(f"Loading safetensors from local directory {model_path}")
        load_model_as_safetensor(self._model, model_path, strict=True, device=TorchUtils.get_device())
        # explicitly move model to correct device # TODO: Check why we need this even though device has been set?
        self._model.to(TorchUtils.get_device())
    
    def set_weights(self, weights):
        self._model.set_weights(weights)

    def get_weights(self):
        return self._model.get_weights()

    @property
    def weights_size(self):
        return self._model.weights_size