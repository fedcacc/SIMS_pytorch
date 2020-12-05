import numpy as np
import random
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.compression import unpack_if_needed
from ray.rllib.execution.replay_ops import SimpleReplayBuffer


@DeveloperAPI
class ReservoirBuffer:
    @DeveloperAPI
    def __init__(self, size):
        """Create Reservoir buffer.
        Inspired to ray.rllib.optimizers.replay_buffer.ReplayBuffer
        Parameters
        ----------
        size: int
          Max number of transitions to store in the buffer. When the buffer
          overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._hit_count = np.zeros(int(size))
        self._num_added = 0
        self._num_sampled = 0
        self._episodes_registry = {}

    def __len__(self):
        return len(self._storage)

    @DeveloperAPI
    def add(self, obs_t, action, eps_id=None):
        data = (obs_t, action)
        self._num_added += 1

        if len(self._storage) < self._maxsize:
            self._storage.append(data)

        else:
            idx = np.random.randint(0, self._num_added + 1)
            if idx < self._maxsize:
                self._storage[idx] = data

    def _encode_sample(self, idxes):
        obses_t, actions = [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action = data
            obses_t.append(np.array(unpack_if_needed(obs_t), copy=False))
            actions.append(np.array(action, copy=False))

            self._hit_count[i] += 1
        return np.array(obses_t), np.array(actions)

    @DeveloperAPI
    def sample_idxes(self, batch_size):
        return np.random.randint(0, len(self._storage), batch_size)

    @DeveloperAPI
    def sample_with_idxes(self, idxes):
        self._num_sampled += len(idxes)
        return self._encode_sample(idxes)

    @DeveloperAPI
    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
          batch of observations
        act_batch: np.array
          batch of actions executed given obs_batch
        """
        idxes = [
            random.randint(0,
                           len(self._storage) - 1) for _ in range(batch_size)
        ]
        self._num_sampled += batch_size
        return self._encode_sample(idxes)

    @DeveloperAPI
    def stats(self, debug=False):
        data = {
            "added_count": self._num_added,
            "sampled_count": self._num_sampled,
            "num_entries": len(self._storage),
        }
        return data


class MultiAgentReservoirBuffer:
    """
    Class for multi-agent reservoir buffer.
    parameters
    """
    def __init__(self, size, policies):
        self.buffers = {}
        self.steps = {}
        self.fake = []
        self.policies = policies
        for policy_id in policies.keys():
            self.buffers[policy_id] = ReservoirBuffer(size)
            self.steps[policy_id] = 0


@DeveloperAPI
class SimpleReservoirBuffer:
    @DeveloperAPI
    def __init__(self, size):
        """Create Reservoir buffer.
        Inspired to ray.rllib.optimizers.replay_buffer.ReplayBuffer
        Parameters
        ----------
        size: int
          Max number of transitions to store in the buffer. When the buffer
          overflows the old memories are dropped.
        """
        self._storage = []
        self._eps_dict = {}
        self._maxsize = size
        self._next_idx = 0
        self._hit_count = np.zeros(int(size))
        self._num_added = 0
        self._num_sampled = 0
        self._episodes_registry = {}

    def __len__(self):
        return len(self._storage)

    @DeveloperAPI
    def add(self, episode_sample, eps):
        if eps in self._episodes_registry:
            idx = self._episodes_registry[eps]
            ss = self._storage[idx][0]
            out_batch = []
            for k in range(len(ss)):
                if episode_sample[k] is None:
                    if ss[k] is None:
                        out_batch.append(None)
                    else:
                        out_batch.append(ss[k])
                else:
                    if ss[k] is None:
                        out_batch.append(episode_sample[k].compress(bulk=True))
                    else:
                        ss[k].decompress_if_needed()
                        episode_sample[k].decompress_if_needed()
                        b = ss[k].concat(episode_sample[k])
                        b.compress(bulk=True)
                        out_batch.append(b)
            self._storage[idx] = (out_batch, eps)

        else:
            self._num_added += 1
            if len(self._storage) < self._maxsize:
                self._storage.append((episode_sample, eps))
                self._episodes_registry[eps] = len(self._storage) - 1

            else:
                idx = np.random.randint(0, self._num_added + 1)
                if idx < self._maxsize:
                    episode_out = self._storage[idx][1]
                    self._episodes_registry.pop(episode_out)
                    self._storage[idx] = (episode_sample, eps)
                    self._episodes_registry[eps] = idx

    def _encode_sample(self, idxes):
        obses_t, actions = [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action = data
            obses_t.append(np.array(unpack_if_needed(obs_t), copy=False))
            actions.append(np.array(action, copy=False))

            self._hit_count[i] += 1
        return np.array(obses_t), np.array(actions)

    @DeveloperAPI
    def sample_idxes(self, batch_size):
        return np.random.randint(0, len(self._storage), batch_size)

    @DeveloperAPI
    def sample_with_idxes(self, idxes):
        self._num_sampled += len(idxes)
        return self._encode_sample(idxes)

    @DeveloperAPI
    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
          batch of observations
        act_batch: np.array
          batch of actions executed given obs_batch
        """
        observations = []
        actions = []
        in_episode_samples = batch_size
        episode_size = int(batch_size/in_episode_samples)
        for i in range(episode_size):
            episode_idx = random.randint(0, len(self._storage)-1)
            while any([x==None for x in self._storage[episode_idx][0]]):
                episode_idx = random.randint(0, len(self._storage)-1)
            episode_sample = self._storage[episode_idx][0]
            for _ in range(in_episode_samples):
                obs = []
                ac = []
                for policy_batch in episode_sample:
                    policy_batch.decompress_if_needed()
                    num_samples = policy_batch.count
                    sample_idx = np.random.randint(0, num_samples)
                    o = unpack_if_needed(policy_batch['obs'][sample_idx])
                    a = policy_batch['actions'][sample_idx]
                    # o = unpack_if_needed(policy[0])
                    # traj_ids = np.random.randint(0, len(o))
                    obs.append(o)
                    ac.append(a)
                observations.append(obs)
                actions.append(ac)

        return np.array(observations, copy=False).reshape(batch_size, -1),\
               np.array(actions, copy=False).reshape(batch_size, -1)


class MultiAgentSimpleReservoirBuffer:
    """
    Class for multi-agent reservoir buffer.
    parameters
    """
    def __init__(self, size, policies):
        self.buffers = {}
        self.steps = {}
        self.policies = policies
        for policy_id in policies.keys():
            self.buffers[policy_id] = SimpleReservoirBuffer(size)
            self.steps[policy_id] = 0

