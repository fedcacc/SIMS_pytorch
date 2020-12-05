from ray.rllib.agents import with_common_config

GENERAL_CONFIGS = with_common_config({
    # Buffers
    "replay_buffer_size": int(2e4),
    "reservoir_buffer_size": int(1e5),

    # Train specs
    "replay_train_batch_size": 128,
    "reservoir_train_batch_size": 128,
    "replay_train_every": 10,
    "reservoir_train_every": 66,
    "reservoir_train_every_sims": 300,
    "replay_min_size_to_learn": 10,
    "reservoir_min_size_to_learn": 1000,
    "reservoir_min_size_to_learn_sims": 1000,
    "replay_num_episodes": 10,
    "replay_min_episodes_to_learn": 100,

    # Models
    "model": {
        "lstm_cell_size": 128,
        "max_seq_len": 999999,
        "fcnet_activation": 'relu',
        "fcnet_hiddens": [128, 128, 128]
    },

    "sig_model": {
        "fcnet_hiddens": [64],
        "fcnet_activation": 'relu',
        "max_seq_len": 20
    },

    # Generals
    "framework": "torch",
    "use_exec_api": True,

    # keys in obs space to be used at inference time
    "test_obs_keys": ["obs"],

    # keys in obs space to be used at training time
    "train_obs_keys": ["obs"],

    # Train statistics flag
    "log_stats": False,
    "stats_fn": None,

    # Wandb integration configs
    "logger_config": {
        "wandb": {
            "project": None,
            "api_key_file": "/home/coordination/wandb_api",
            "log_config": True
        }
    },

    # Param to support cases in which training environment is different than execution environment (e.g. SIMS)
    "train_obs_space": None,

    # Param for hyperparameter tuning
    "model_struc": None,  # TODO (fede): remove once tuning is done

    # Debugging purposes, log probabilities for specific actions when training
    "relevant_obs": None
})


def with_general_config(configs_new):
    conf = GENERAL_CONFIGS.copy()
    conf.update(configs_new)
    return conf
