import numpy as np
from optimizers.replay_buffer import MultiAgentReplayBuffer, MultiAgentSimpleReplayBuffer
from ray.util.iter_metrics import SharedMetrics
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.util.iter import LocalIterator, _NextValueNotReady


def LocalReplayMultiagent(replay_buffers, train_batch_size, min_size_to_learn, learn_every, learn_every_res=None,
                          prioritized=False, beta=1):
    """Replay experiences from a MultiAgentReplayBuffer instance.
    Soon to be deprecated # TODO: update
    Arguments:
        replay_buffers (MultiAgentReplayBuffer): Buffers to replay experiences from.
        train_batch_size (int): Batch size of fetches from the buffer.
        min_size_to_learn (int): Minimum buffer length to start learning.
        learn_every (int): Number of steps between any learning iteration.
        learn_every_res (int): Number of steps between two learning iteration of the avg network
        prioritized (bool): DEPRECATED
        beta (float): DEPRECATED
:
    """
    assert isinstance(replay_buffers, MultiAgentReplayBuffer)


    def gen_replay(timeout):
        while True:
            samples = {}
            idxes = None
            for policy_id, replay_buffer in replay_buffers.buffers.items():
                policy_multiplier = 2 if policy_id == 'policy_team' else 1
                if len(replay_buffer) >= min_size_to_learn*policy_multiplier and \
                     replay_buffers.steps[policy_id] >= learn_every:

                    idxes = replay_buffer.sample_idxes(train_batch_size)
                    replay_buffers.steps[policy_id] = 0

                    if prioritized:
                        (obses_t, actions, rewards, obses_tp1, dones, w, ind) \
                            = replay_buffer.sample_with_idxes(idxes, beta)

                    else:
                        (obses_t, actions, rewards, obses_tp1, dones) = replay_buffer.sample_with_idxes(idxes)
                    weights = np.ones_like(rewards)
                    batch_indexes = -np.ones_like(rewards)
                    samples[policy_id] = SampleBatch({
                        "obs": obses_t,
                        "actions": actions,
                        "rewards": rewards,
                        "new_obs": obses_tp1,
                        "dones": dones,
                        "weights": weights,
                        "batch_indexes": batch_indexes
                    })

            if samples == {}:
                yield _NextValueNotReady()
            else:
                yield MultiAgentBatch(samples, train_batch_size)

    return LocalIterator(gen_replay, SharedMetrics())


def SimpleLocalReplayMultiagent(replay_buffers, train_batch_size, min_size_to_learn, learn_every):
    """Replay experiences from a MultiAgentReplayBuffer instance.
    Arguments:
        replay_buffers (MultiAgentSimpleReplayBuffer): Buffers to replay experiences from.
        train_batch_size (int): Batch size of fetches from the buffer.
        min_size_to_learn (int): Minimum buffer length to start learning.
        learn_every (int): Number of steps between any learning iteration.
    """
    assert isinstance(replay_buffers, MultiAgentSimpleReplayBuffer)


    def gen_replay(timeout):
        while True:
            samples = {}
            for policy_id, replay_buffer in replay_buffers.buffers.items():
                if len(replay_buffer.replay_batches) >= min_size_to_learn and \
                     replay_buffers.steps[policy_id] >= learn_every:

                    batch = None
                    for x in range(train_batch_size):
                        if batch is None:
                            batch = replay_buffer.replay().decompress_if_needed()
                        else:
                            batch = batch.concat(replay_buffer.replay().decompress_if_needed())

                    replay_buffers.steps[policy_id] = 0
                    samples[policy_id] = batch

            if samples == {}:
                yield _NextValueNotReady()
            else:
                yield MultiAgentBatch(samples, train_batch_size)

    return LocalIterator(gen_replay, SharedMetrics())