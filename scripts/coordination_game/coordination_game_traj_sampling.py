from scripts.coordination_game.default_coord_config import DEFAULT_CONFIG
from omegaconf import OmegaConf
import ray
from ray import tune
import numpy as np
from ray.tune.registry import register_env
from agents.nfsp.nfsp_trainer import NFSPPolicy, NFSPTrainer
from envs.coord.coordination_env_perfect_info_v2 import CoordinationEnvPerfectInfo
from integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS
from envs.coord.nfsp_eval_learning.eval_fn import eval_probs


def seed_fn(seed):
    pass

def on_episode_start(info):
    """ Sample one play mode (either BR or AVG) for each NFSP agent """
    info["episode"].user_data["modes"] = {}
    team_mode = info['policy']['policy_t1'].sample_mode()
    for policy_id, policy in info["policy"].items():
        if isinstance(policy, NFSPPolicy):
            if policy_id == 'policy_opp':
                info["episode"].user_data["modes"][policy] = policy.sample_mode()
            else:
                info["episode"].user_data["modes"][policy] = team_mode


def select_policy(agent_id):
    assert agent_id in {"t1", "t2", "opponent", "team", "signaler"}
    if agent_id == "t1":
        return "policy_t1"
    elif agent_id == "t2":
        return "policy_t1"
    elif agent_id == "team":
        return "policy_team"
    elif agent_id == "signaler":
        return "policy_signaler"
    return "policy_opp"


def run(args, cl_args):
    nfsp_env_cls = CoordinationEnvPerfectInfo
    # NFSP environment
    sample_env = nfsp_env_cls(args.env_config)
    obs_space_defender = sample_env.observation_space
    obs_space_attacker = sample_env.observation_space
    action_space_defender = sample_env.action_space
    action_space_attacker = sample_env.action_space

    register_env("nfsp_env", lambda _: sample_env)
    horz = args.env_config.horizon

    train_config = {
        "use_exec_api": True,
        "anticipatory_param": tune.grid_search([0.3, 0.5]),
        "seed": tune.sample_from(lambda x: np.random.randint(10000)),
        "rollout_fragment_length": 100*horz,
        "timesteps_per_iteration": 100*horz,
        "env": "nfsp_env",
        "num_workers": 1,
        "num_envs_per_worker": 1,
        "train_batch_size": args.train_batch_size,
        "multiagent": {
            "policies": {
                "policy_opp": (NFSPPolicy, obs_space_attacker, action_space_attacker, {}),
                "policy_t1": (NFSPPolicy, obs_space_defender, action_space_defender, {}),
                "policy_t2": (NFSPPolicy, obs_space_defender, action_space_defender, {})
            },
            "policies_to_train": ["policy_t1", "policy_opp"],
            "policy_mapping_fn": select_policy,

        },
        "stats_fn": eval_probs,
        "callbacks": {
            "on_episode_start": on_episode_start,
            # "on_episode_step": on_episode_step,
            # "on_episode_end": on_episode_end,
            # "on_sample_end": on_sample_end,
            # "on_train_result": on_train_results,
            # "on_postprocess_traj": on_postprocess_traj,
        },
        "replay_train_every": 10,
        "reservoir_train_every": 66,
        "reservoir_train_every_sims": 66,
        "batch_mode": "complete_episodes",
        "lr": 1e-2,
        "model_struc": tune.grid_search([
            [64, 64],
            [128, 128]
        ]),
        "clip_actions": False,
        "replay_buffer_size": int(2e4),
        "reservoir_buffer_size": int(5e5),
        "framework": "torch",
        "evaluation_interval": 50,
        "evaluation_num_episodes": 100,
        "evaluation_config": {
            'anticipatory_param': 0.
        },
        "log_stats": True,
        "logger_config": {
            "wandb": {
                "project": "infsp_coord_game",
                "api_key_file": "/home/coordination/wandb_api",
                "log_config": False
            }
        },
    }

    ray.init(log_to_driver=False, local_mode=False)

    # Build loggers
    DEFAULT_DIR = "~/ray_results/coord_traj_sampling_tests"
    tune.run(NFSPTrainer,
             config=train_config,
             local_dir=DEFAULT_DIR,
             stop={"timesteps_total": 3e6},
             num_samples=cl_args.num_samples,
             loggers=DEFAULT_LOGGERS + (WandbLogger,))

    ray.shutdown()


if __name__ == "__main__":
    cl_args = OmegaConf.from_cli()
    input_args = OmegaConf.create({**DEFAULT_CONFIG, **dict(cl_args)})
    dict_args = {
        'env_config': {
            'horizon': cl_args.horizon,
            'actions_payoff': [(0, 100.), (1, 50.)],
        },
        'train_batch_size': 128,
    }
    dict_args = OmegaConf.create(dict_args)
    run(dict_args, input_args)