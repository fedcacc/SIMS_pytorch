import numpy as np

from utils.utils import MODE
from agents.nfsp.avg_torch_policy import AVGPolicy
from gym.spaces import Tuple
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import override
from agents.nfsp.simple_q_torch_policy import SimpleQTorchPolicyMSE


class NFSPPolicy(Policy):
    """
    Policy class for Neural Fictitious Self Play described in https://arxiv.org/pdf/1603.01121.pdf.
    At the beginning of each episode the policy samples a sub-policy to use for the whole episode:
    with probability p=(anticipatory_param) it plays according to a best response strategy modeled as a DQN policy, and
    with probability p=(1-anticipatory_param) it plays according to an average strategy modeled as an AVG policy.

    """
    def __init__(self, observation_space, action_space, config):
        super(NFSPPolicy, self).__init__(observation_space, action_space, config)
        self.anticipatory_param = config["anticipatory_param"]
        avg_config = {**config, **config['avg_policy']}
        dqn_config = {**config, **config["dqn_policy"]}

        avg_config["model"]["fcnet_hiddens"] = config["model_struc"]
        dqn_config["model"]["fcnet_hiddens"] = config["model_struc"]
        # avg_config["lr"] = config["lr"]
        # dqn_config["lr"] = config["lr"]


        if isinstance(action_space, Tuple):
            assert len(action_space.spaces) == 1
            self.action_space = action_space.spaces[0]


        # Define DQN:
        self.dqn = SimpleQTorchPolicyMSE(observation_space, action_space, dqn_config)

        # Define AVG policy
        self.avg = AVGPolicy(observation_space, action_space, avg_config)
        self.framework = "torch"
        if not hasattr(self, "exploration"):
            self.exploration = self._create_exploration()

    @override(Policy)
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None,
                        info_batch=None, episodes=None, explore=None, timestep=None, **kwargs):

        if episodes is not None:
            modes = [ep.user_data["modes"][self].value for ep in episodes]
        else:
            modes = [MODE.average_strategy.value for o in obs_batch]
        assert len(modes) == len(obs_batch), "Error when checking execution modes for the batch" + obs_batch

        # TODO: implement for multiple observations
        # assert len(obs_batch) != 1, "This function is not yet supported for more than one observations at a time!"

        if modes[0] == MODE.best_response.value:
            action, rnn, info = self.dqn.compute_actions(obs_batch, state_batches, prev_action_batch, prev_reward_batch,
                                                         info_batch, episodes, explore, timestep, **kwargs)

        elif modes[0] == MODE.average_strategy.value:
            action, rnn, info = self.avg.compute_actions(obs_batch, state_batches, prev_reward_batch, prev_reward_batch,
                                                         info_batch, episodes, explore, timestep, **kwargs)
            rnn = state_batches
            # add fictitious q_value info for the interface
            info["q_values"] = -np.ones(shape=(len(obs_batch), self.action_space.n))

        else:
            raise ValueError("Wrong mode chosen: {}".format(modes[0]))

        info["action_dist_inputs"] = -np.ones(shape=(len(obs_batch), self.action_space.n))
        info["mode"] = np.array(modes)
        return action, rnn, info


    @override(Policy)
    def learn_on_batch(self, samples):
        # Depending on the samples structure either train average network or dqn
        is_dqn_batch = samples.REWARDS in samples.data.keys()
        if is_dqn_batch:
            return self.dqn.learn_on_batch(samples)
        else:
            return self.avg.learn_on_batch(samples)

    @override(Policy)
    def get_weights(self):
        return {
            "DQN": self.dqn.get_weights(),
            "AVG": self.avg.get_weights()
        }

    @override(Policy)
    def set_weights(self, weights):
        self.dqn.set_weights(weights["DQN"])
        self.avg.set_weights(weights["AVG"])

    def update_target(self):
        self.dqn.update_target()

    def sample_mode(self):
        return np.random.choice([MODE.best_response, MODE.average_strategy],
                                p=[self.anticipatory_param, 1-self.anticipatory_param])

    @override(Policy)
    def get_initial_state(self):
        return [s.cpu().numpy() for s in self.dqn.model.get_initial_state()]



