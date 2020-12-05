from ray.rllib.policy.policy import Policy
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_torch
from gym.spaces import Discrete, MultiDiscrete, Tuple

torch, nn = try_import_torch()


class LearnableSignalerPolicy(Policy):
    def __init__(self, obs_space, act_space, config):
        super(LearnableSignalerPolicy, self).__init__(obs_space, act_space, config)
        self.framework = "torch"
        self.exploration = self._create_exploration()
        self.n_signals = act_space.spaces[0].n if isinstance(act_space, Tuple) else act_space.n
        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))
        self.model = ModelCatalog.get_model_v2(
            MultiDiscrete([2]),
            Discrete(self.n_signals),
            self.n_signals,
            config['sig_model'],
            framework="torch",
            name="SignalerNet",
            default_model=FullyConnectedNetwork
        )

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        explore=None,
                        timestep=None,
                        **kwargs):
        obs_batch = torch.as_tensor(obs_batch, dtype=torch.float, device=self.device)
        action_logits, _ = self.model({"obs": obs_batch})
        from torch.distributions import Categorical
        dist = Categorical(logits=action_logits)
        actions = dist.sample().long().cpu().numpy()
        return [[a] for a in actions], [], {}

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {
            "model": self._cpu_dict(self.model.state_dict()),
        }

    def set_weights(self, weights):
        self.model.load_state_dict(self._device_dict(weights["model"]))

    def _device_dict(self, state_dict):
        return {
            k: torch.as_tensor(v, device=self.device)
            for k, v in state_dict.items()
        }

    @staticmethod
    def _cpu_dict(state_dict):
        return {k: v.cpu().detach().numpy() for k, v in state_dict.items()}


class LearnableMultiSignalerPolicy(Policy):
    def __init__(self, obs_space, act_space, config):
        super(LearnableMultiSignalerPolicy, self).__init__(obs_space, act_space, config)
        self.framework = "torch"
        self.exploration = self._create_exploration()
        # self.n_signals = act_space.spaces[0].n if isinstance(act_space, Tuple) else act_space.n
        self.signal_shape = act_space.spaces[0].shape if isinstance(act_space, Tuple) else act_space.shape
        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))
        self.model = ModelCatalog.get_model_v2(
            MultiDiscrete([2]),
            Discrete(self.n_signals),
            self.n_signals,
            config['sig_model'],
            framework="torch",
            name="SignalerNet",
            default_model=FullyConnectedNetwork
        )

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        explore=None,
                        timestep=None,
                        **kwargs):
        obs_batch = torch.as_tensor(obs_batch, dtype=torch.float, device=self.device)
        action_logits, _ = self.model({"obs": obs_batch})
        from torch.distributions import Categorical
        dist = Categorical(logits=action_logits)
        actions = dist.sample().long().cpu().numpy()
        return [[a] for a in actions], [], {}

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {
            "model": self._cpu_dict(self.model.state_dict()),
        }

    def set_weights(self, weights):
        self.model.load_state_dict(self._device_dict(weights["model"]))

    def _device_dict(self, state_dict):
        return {
            k: torch.as_tensor(v, device=self.device)
            for k, v in state_dict.items()
        }

    @staticmethod
    def _cpu_dict(state_dict):
        return {k: v.cpu().detach().numpy() for k, v in state_dict.items()}