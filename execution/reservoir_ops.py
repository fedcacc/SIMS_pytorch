import os
from optimizers.reservoir_buffer import MultiAgentReservoirBuffer, MultiAgentSimpleReservoirBuffer
from ray.util.iter_metrics import SharedMetrics
from ray.util.iter import LocalIterator, _NextValueNotReady
from ray.rllib.execution.common import _get_shared_metrics, STEPS_SAMPLED_COUNTER
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from utils.visualization.visual import _dump_buffer_content

def LocalReservoirMultiagent(reservoir_buffers, train_batch_size, min_size_to_learn, learn_every):
    """
    Get experiences from multi-agent reservoir buffer
    Arguments:
        reservoir_buffers (MultiAgentReservoirBuffer): Buffers to replay experiences from.
        train_batch_size (int): Batch size of fetches from the buffer.
        min_size_to_learn (int): Minimum buffer length to start learning.
        learn_every (int): Number of steps between any learning iteration.
    """
    assert isinstance(reservoir_buffers, MultiAgentReservoirBuffer)

    def gen_replay(timeout):
        while True:
            samples = {}
            idxes = None

            for policy_id, reservoir_buffer in reservoir_buffers.buffers.items():
                if len(reservoir_buffer) >= min_size_to_learn and \
                        reservoir_buffers.steps[policy_id] >= learn_every:

                    idxes = reservoir_buffer.sample_idxes(train_batch_size)
                    (obses_t, actions) = reservoir_buffer.sample_with_idxes(idxes)
                    samples[policy_id] = SampleBatch({
                        "obs": obses_t,
                        "actions": actions,
                    })

                    reservoir_buffers.steps[policy_id] = 0

            if samples == {}:
                yield _NextValueNotReady()
            else:
                yield MultiAgentBatch(samples, train_batch_size)

    return LocalIterator(gen_replay, SharedMetrics())


def SimpleLocalReservoirMultiagent(reservoir_buffers, train_batch_size, min_size_to_learn, learn_every):
    """
    Get experiences from multi-agent reservoir buffer
    Arguments:
        reservoir_buffers (MultiAgentSimpleReservoirBuffer): Buffers to replay experiences from.
        train_batch_size (int): Batch size of fetches from the buffer.
        min_size_to_learn (int): Minimum buffer length to start learning.
        learn_every (int): Number of steps between any learning iteration.
    """
    assert isinstance(reservoir_buffers, MultiAgentSimpleReservoirBuffer)

    def gen_replay(timeout):
        while True:
            samples = {}
            idxes = None

            for policy_id, reservoir_buffer in reservoir_buffers.buffers.items():
                if len(reservoir_buffer) >= min_size_to_learn and \
                        reservoir_buffers.steps[policy_id] >= learn_every:
                    # idxes = reservoir_buffer.sample_idxes(train_batch_size)
                    (obses_t, actions) = reservoir_buffer.sample(train_batch_size)
                    samples[policy_id] = SampleBatch({
                        "obs": obses_t,
                        "actions": actions,
                    })

                    reservoir_buffers.steps[policy_id] = 0

            if samples == {}:
                yield _NextValueNotReady()
            else:
                yield MultiAgentBatch(samples, train_batch_size)

    return LocalIterator(gen_replay, SharedMetrics())


class DumpBufferContent:
    def __init__(self, workers, f_path, buffer, do_every=1e6):
        self.f_path = f_path
        self.buffer = buffer
        self.do_every = do_every
        self.workers = workers
        self.prev_update = 0

    def __call__(self, _):
        metrics = _get_shared_metrics()
        cur_ts = metrics.counters[STEPS_SAMPLED_COUNTER]
        if cur_ts - self.prev_update > self.do_every:
            self.prev_update = cur_ts
            filepath = os.path.join(self.f_path, "{}_steps".format(cur_ts))
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            _dump_buffer_content(self.buffer, filepath)