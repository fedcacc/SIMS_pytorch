from ray.rllib.execution.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from ray.rllib.execution.replay_ops import SimpleReplayBuffer


class MultiAgentReplayBuffer:
    """
    Class for multi-agent replay buffer. Soon to be deprecated  TODO: update
    """
    def __init__(self, size, policies, prioritized=False, alpha=1):
        self.buffers = {}
        self.steps = {}
        self.policies = policies
        for policy_id in policies.keys():
            if prioritized:
                self.buffers[policy_id] = PrioritizedReplayBuffer(size, alpha)
            else:
                self.buffers[policy_id] = ReplayBuffer(size)
            self.steps[policy_id] = 0


class MultiAgentSimpleReplayBuffer:
    """
    Class for multi-agent replay buffer.
    """
    def __init__(self, size, policies):
        self.buffers = {}
        self.steps = {}
        self.policies = policies
        for policy_id in policies.keys():
            self.buffers[policy_id] = SimpleReplayBuffer(size)
            self.steps[policy_id] = 0


