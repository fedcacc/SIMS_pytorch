from optimizers.reservoir_buffer import MultiAgentReservoirBuffer
from optimizers.replay_buffer import MultiAgentSimpleReplayBuffer
from execution.store_ops import StoreToBuffers
from execution.reservoir_ops import LocalReservoirMultiagent
from execution.replay_ops import SimpleLocalReplayMultiagent
from agents.nfsp.nfsp_torch_policy import NFSPPolicy
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.train_ops import TrainOneStep, UpdateTargetNetwork
from agents import with_general_config
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from execution.metric_ops import LowMemoryMetricsReporting


NFSP_CONFIG = with_general_config({
    # Policy configs (NFSP)
    "anticipatory_param": 0.5,

    # Average policy specific configs (NFSP)
    "avg_policy": {
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": 'relu',
            "max_seq_len": 200,
            "conv_filters": [
                [16, [3, 3], 2],
                [32, [3, 3], 6],
            ],
            "lstm_cell_size": 64,
        },
        "lr": 1e-3,
        "exploration_config": {
            "type": "StochasticSampling"
        }
    },

    # DQN policy specific configs (NFSP)
    "dqn_policy": {
        "use_pytorch": True,
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": 'relu',
            "max_seq_len": 200,
            "conv_filters": [
                [16, [3, 3], 2],
                [32, [3, 3], 6],
            ],
            "lstm_cell_size": 64,
        },
        "exploration_config": {
            "type": "utils.exploration.epsilon_greedy.EpsilonGreedy",
            "initial_epsilon": 0.2,
            "final_epsilon": 0.01,
            "epsilon_timesteps": int(7e5)
        },
        "target_network_update_freq": 10000,
        "lr": 1e-3
        ,
    },

    "recurrent_dqn": False,

})


def execution_plan_nfsp(workers, config):
    # 1. define buffers
    replay_size = config["replay_buffer_size"]
    reservoir_size = config["reservoir_buffer_size"]
    replay_buffers = MultiAgentSimpleReplayBuffer(replay_size,
                                            config["multiagent"]["policies"])
    reservoir_buffers = MultiAgentReservoirBuffer(reservoir_size,
                                                  config["multiagent"]["policies"])
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # 2. define store operations
    store_op = rollouts.for_each(StoreToBuffers(replay_buffers, reservoir_buffers,
                                                config['multiagent']['policies_to_train'])) # Sampling

    # 3. define replay/reservoir operations
    replay_op = SimpleLocalReplayMultiagent(replay_buffers, config["replay_train_batch_size"],
                                      config["replay_min_size_to_learn"],
                                      config["replay_train_every"]) \
        .for_each(TrainOneStep(workers))\
        .for_each(UpdateTargetNetwork(workers, config['dqn_policy']["target_network_update_freq"]))

    reservoir_op = LocalReservoirMultiagent(reservoir_buffers, config["reservoir_train_batch_size"],
                                            config["reservoir_min_size_to_learn"],
                                            config["reservoir_train_every"])\
        .for_each(TrainOneStep(workers))

    # 4. define main train loop
    train_op = Concurrently([replay_op, reservoir_op, store_op], mode="round_robin")
    return LowMemoryMetricsReporting(train_op, workers, config)


NFSPTrainer = build_trainer(
    name='NFSPTrainer',
    default_policy=NFSPPolicy,
    default_config=NFSP_CONFIG,
    execution_plan=execution_plan_nfsp
)