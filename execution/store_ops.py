from optimizers.replay_buffer import MultiAgentReplayBuffer, MultiAgentSimpleReplayBuffer
from optimizers.reservoir_buffer import MultiAgentReservoirBuffer
from utils.utils import MODE
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.compression import pack_if_needed
import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch


class StoreToBuffers:
    """
    Callable that stores data into either into a replay buffer or into a reservoir buffer, depending
    on the strategy used to obtain them.
    """

    def __init__(self, replay_buffers, reservoir_buffers, policies_to_train):
        assert isinstance(replay_buffers, MultiAgentSimpleReplayBuffer)
        assert isinstance(reservoir_buffers, MultiAgentReservoirBuffer)
        self.replay_buffers = replay_buffers
        self.reservoir_buffers = reservoir_buffers
        self.policies_to_train = policies_to_train

    def __call__(self, batch: SampleBatchType):
        x = 0
        for policy_id, s in batch.policy_batches.items():
            if policy_id in self.policies_to_train:
                for row in s.rows():
                    flag = row["mode"] == MODE.best_response.value
                    if flag:
                        # Transition must be inserted in the reservoir buffer
                        self.reservoir_buffers.buffers[policy_id].add(
                            pack_if_needed(row["obs"]),
                            row["actions"])
                        self.replay_buffers.steps[policy_id] += 1

                    bb = SampleBatch({
                        'obs': row["obs"].reshape(1, -1),
                        'actions': row['actions'].reshape(1,-1),
                        'rewards': row['rewards'].reshape(1,-1),
                        'new_obs': row['new_obs'].reshape(1,-1),
                        'dones': np.array([row['dones']]),
                        "eps_id": np.array([row['eps_id']]),
                        'unroll_id': np.array([row['unroll_id']]),
                        'agent_index': np.array([row['agent_index']])
                    })
                    bb.compress(bulk=True)
                    self.replay_buffers.buffers[policy_id].add_batch(bb)
                    self.reservoir_buffers.steps[policy_id] += 1

        return batch


class StoreToReplayBuffer:
    def __init__(self, replay_buffer: MultiAgentReplayBuffer):
        assert isinstance(replay_buffer, MultiAgentReplayBuffer)
        self.replay_buffers = replay_buffer

    def __call__(self, batch: SampleBatchType):
        for policy_id, s in batch.policy_batches.items():
            for row in s.rows():
                self.replay_buffers.buffers[policy_id].add(
                    pack_if_needed(row["obs"]),
                    row["actions"],
                    row["rewards"],
                    pack_if_needed(row["new_obs"]),
                    row["dones"],
                    weight=None)
        return batch


class StoreToSimpleReplayBuffer:
    def __init__(self, replay_buffer: MultiAgentSimpleReplayBuffer):
        assert isinstance(replay_buffer, MultiAgentSimpleReplayBuffer)
        self.replay_buffers = replay_buffer

    def __call__(self, batch: SampleBatchType):
        for policy_id, s in batch.policy_batches.items():
            for row in s.rows():
                b = {}
                for k, v in row.items():
                    if not isinstance(v, np.ndarray):
                       b[k] = np.array([v])
                    else:
                        b[k] = v.reshape(1, -1)
                b = SampleBatch(b)
                b.compress(bulk=True)
                self.replay_buffers.buffers[policy_id].add_batch(b)
                self.replay_buffers.steps[policy_id] += 1
        return batch


class StoreToBuffersEpisodeWise:
    """
    Callable that stores data into either into a replay buffer or into a reservoir buffer, depending
    on the strategy used to obtain them.
    """

    def __init__(self, replay_buffers, reservoir_buffers, policies_to_train):
        assert isinstance(replay_buffers, MultiAgentSimpleReplayBuffer)
        assert isinstance(reservoir_buffers, MultiAgentReservoirBuffer)
        self.replay_buffers = replay_buffers
        self.reservoir_buffers = reservoir_buffers
        self.policies_to_train = policies_to_train

    def __call__(self, batch: SampleBatchType):
        x = 0
        for policy_id, s in batch.policy_batches.items():
            if policy_id in self.policies_to_train:
                for row in s.rows():
                    if row["mode"] == MODE.best_response.value:
                        # Transition must be inserted in the reservoir buffer
                        self.reservoir_buffers.buffers[policy_id].add(
                            pack_if_needed(row["obs"]),
                            row["actions"])
                        self.replay_buffers.steps[policy_id] += 1

                episode_ids = np.unique(s['eps_id'])
                for ep_id in episode_ids:
                    sample_ids = np.where(s["eps_id"] == ep_id)
                    bb = SampleBatch({
                        'obs': s["obs"][sample_ids],
                        'actions': s['actions'][sample_ids],
                        'rewards': s['rewards'][sample_ids],
                        'new_obs': s['new_obs'][sample_ids],
                        'dones': np.array(s['dones'][sample_ids]),
                        "eps_id": np.array(s['eps_id'][sample_ids]),
                        'unroll_id': np.array(s['unroll_id'][sample_ids]),
                        'agent_index': np.array(s['agent_index'][sample_ids])
                    })
                    bb.compress(bulk=True)
                    self.replay_buffers.buffers[policy_id].add_batch(bb)
                    self.reservoir_buffers.steps[policy_id] += bb.count


        return batch

