from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from execution.metric_ops import LowMemoryMetricsReporting
from ray.rllib.execution.train_ops import UpdateTargetNetwork
from optimizers.reservoir_buffer import MultiAgentReservoirBuffer
from optimizers.replay_buffer import MultiAgentSimpleReplayBuffer
from execution.replay_ops import SimpleLocalReplayMultiagent
from execution.store_ops import StoreToBuffers, StoreToBuffersEpisodeWise
from execution.reservoir_ops import LocalReservoirMultiagent
from agents.sims.sims_policy import SIMSPolicy
from agents.sims.ops import StoreJointOptimized, StoreJointOptimizedSinglePolicy
from agents import with_general_config
from agents.nfsp.nfsp_trainer import NFSP_CONFIG
import torch

SIMS_CONFIG = with_general_config({
    # optimisers config
    "model_optimiser": {
      "type": torch.optim.Adam,
      "lr": 1e-4
    },
    "sig_model_optimiser": {
        "type": torch.optim.Adam,
        "lr": 1e-4
    },

    # SIMS train batch size
    "sims_train_batch_size": 128,

    # number of signals used at training time
    "n_train_signals": 2,
    # beta parameter for SIMS loss function
    "beta": 0.1,
    # gamma parameter for SIMS loss function
    "factor_ent": 0.,

    # lr scheduling options TODO (fede): still experimental, check and include different options for scheduling
    'lr_scheduling': {
        'initial_lr': 1e-3,
        'shrink_factor': 0.1,
        'shrink_every_iters': 5e3
    },

    # beta scheduling options TODO (fede): still experimental, check and include different options for scheduling
    'beta_scheduling': {
        'initial_beta': 0.01,
        'shrink_factor': 10,
        'shrink_every_iters': 5e3,
        'final_beta': 0.1
    },

})


def execution_plan_sims_infsp(workers, config):
    # 1. Define buffers
    replay_size = config["replay_buffer_size"]
    reservoir_size = config["reservoir_buffer_size"]

    replay_buffers = MultiAgentSimpleReplayBuffer(replay_size,
                                                  config["multiagent"]["policies"])
    reservoir_buffers = MultiAgentReservoirBuffer(reservoir_size,
                                                  config["multiagent"]["policies"])
    reservoir_buf_team = MultiAgentReservoirBuffer(reservoir_size,
                                                   {"policy_team": ()})
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # 2. Store operation (sampling and insertion in buffers)
    store_class = StoreToBuffersEpisodeWise if config["recurrent_dqn"] else StoreToBuffers
    store_op = rollouts.for_each(store_class(replay_buffers, reservoir_buffers,
                                             config['multiagent']['policies_to_train']))\
        .for_each(StoreJointOptimizedSinglePolicy(reservoir_buf_team))

    # 3. Train operations: sample from reservoir/replay buffer and train
    replay_train_batch = config["replay_num_episodes"] if config["recurrent_dqn"] \
        else config["replay_train_batch_size"]
    replay_min_size_to_learn = config["replay_min_episodes_to_learn"] if config["recurrent_dqn"] \
        else config["replay_min_size_to_learn"]
    replay_op = SimpleLocalReplayMultiagent(replay_buffers, replay_train_batch,
                                            replay_min_size_to_learn,
                                            config["replay_train_every"]) \
        .for_each(TrainOneStep(workers))\
        .for_each(UpdateTargetNetwork(workers, config['dqn_policy']["target_network_update_freq"]))

    reservoir_op = LocalReservoirMultiagent(reservoir_buffers, config["reservoir_train_batch_size"],
                                            config["reservoir_min_size_to_learn"],
                                            config["reservoir_train_every"])\
        .for_each(TrainOneStep(workers))

    reservoir_op_team = LocalReservoirMultiagent(reservoir_buf_team, config["sims_train_batch_size"],
                                                 config["reservoir_min_size_to_learn_sims"],
                                                 config["reservoir_train_every_sims"]) \
        .for_each(TrainOneStep(workers))

    # 4. Training loop definition
    train_op = Concurrently([replay_op, store_op, reservoir_op, reservoir_op_team], mode="round_robin")
    return LowMemoryMetricsReporting(train_op, workers, config)


SIMSiNFSPTrainer = build_trainer(
    name="SignalTrainer",
    default_policy=SIMSPolicy,
    default_config={**NFSP_CONFIG, **SIMS_CONFIG},
    execution_plan=execution_plan_sims_infsp
)
