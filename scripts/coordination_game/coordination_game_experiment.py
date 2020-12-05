from scripts.patrolling.default_patrolling_config import DEFAULT_CONFIG
from omegaconf import OmegaConf
import ray
from ray import tune
import numpy as np
from ray.tune.registry import register_env
from gym.spaces import Tuple
from agents.sims.sims_policy import SIMSPolicy
from agents.signal.signaler import LearnableSignalerPolicy
from gym.spaces import Discrete, Dict
from agents.sims.sims_trainer import SIMSiNFSPTrainer
from experiments_paper.wolfpack_simple.eval_function import eval_function_sims
from envs.coord.coord_wrapper_class import CoordWrapperClass
from envs.coord.coordination_signaler_v2 import CoordinationSignalerImperfectInfo
from agents.nfsp.nfsp_torch_policy import NFSPPolicy
from envs.coord.coordination_env_perfect_info_v2 import CoordinationEnvPerfectInfo
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS


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
    elif agent_id == "t1":
        return "policy_t2"
    elif agent_id == "team":
        return "policy_team"
    elif agent_id == "signaler":
        return "policy_signaler"
    return "policy_opp"


def run(args, cl_args):
    obs_space_fake = Dict({
        'obs': Discrete(1)
    })
    n_signals = args.discrete_env_config.n_signals
    nfsp_env_cls = CoordinationEnvPerfectInfo
    sims_env_cls = CoordinationSignalerImperfectInfo

    # NFSP environment
    sample_env = nfsp_env_cls(args.env_config)
    obs_space_defender = sample_env.observation_space
    obs_space_attacker = sample_env.observation_space
    action_space_defender = sample_env.action_space
    action_space_attacker = sample_env.action_space

    register_env("nfsp_env", lambda _: sample_env)

    # SIMS environment
    signaled_sample_env = sims_env_cls(args.discrete_env_config)
    obs_space_signaled = signaled_sample_env.observation_space_signaled
    team_obs_space = Tuple([obs_space_signaled, obs_space_signaled])
    team_act_space = Tuple([action_space_defender, action_space_defender])
    groups = {
        "team": ["t1", "t2"],
        "signaler": ["signaler"],
        "opponent": ["opponent"]
    }
    grouped_env_eval = sims_env_cls(args.discrete_env_config).with_agent_groups(groups)
    grouped_env_eval.seed = seed_fn
    args.discrete_env_config.groups = groups
    args.env_config.groups = {
        "t1": ["t1"],
        "t2": ["t2"],
        "opponent": ["opponent"]
    }
    register_env("team_env", lambda _: grouped_env_eval)
    train_config = {
        "n_train_signals": n_signals,
        "use_exec_api": True,
        "seed": tune.sample_from(lambda x: np.random.randint(10000)),
        "env": CoordWrapperClass,
        "env_config": args.env_config,
        "rollout_fragment_length": args.env_config.horizon,
        "timesteps_per_iteration": 1,
        "batch_mode": "complete_episodes",
        "num_workers": 1,
        "num_envs_per_worker": 1,
        "train_batch_size": args.train_batch_size,
        "multiagent": {
            "policies": {
                "policy_team": (SIMSPolicy, team_obs_space, team_act_space,
                                {"train_obs_space": Tuple([obs_space_defender] * 2)}),
                "policy_opp": (NFSPPolicy, Tuple([obs_space_attacker]), Tuple([action_space_attacker]), {}),
                "policy_signaler": (LearnableSignalerPolicy, Tuple([obs_space_fake]), Tuple([Discrete(n_signals)]), {}),
                "policy_t1": (NFSPPolicy, Tuple([obs_space_defender]), Tuple([action_space_defender]), {
                    "test_obs_keys": ["obs"],
                    "train_obs_keys": ["obs"],
                }),
            },
            "policies_to_train": ["policy_t1", "policy_opp", "policy_team"],
            "policy_mapping_fn": select_policy,

        },
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
        "lr": 1e-3,
        "beta": 0.1,
        "clip_actions": False,
        "replay_buffer_size": int(2e4),
        "reservoir_buffer_size": int(1e5),
        "framework": "torch",
        "recurrent_dqn": False,
        "evaluation_interval": 100,
        "evaluation_num_episodes": 100,
        "evaluation_config": {
            'env_config': args.discrete_env_config,
            'anticipatory_param': 0.
        },
        "custom_eval_function": eval_function_sims,
        "log_stats": True,
        "logger_config": {
            "wandb": {
                "project": "sims_coordgame",
                "api_key_file": "/home/coordination/wandb_api",
                "log_config": False
            }
        },
    }

    ray.init(log_to_driver=False, local_mode=False)

    # Build loggers
    DEFAULT_DIR = "~/ray_results/coord_game"
    tune.run(SIMSiNFSPTrainer,
             config=train_config,
             local_dir=DEFAULT_DIR,
             stop={"timesteps_total": 3e6},
             checkpoint_at_end=True,
             num_samples=cl_args.num_samples,
             loggers=DEFAULT_LOGGERS + (WandbLogger,),
             )

    ray.shutdown()


if __name__ == "__main__":
    cl_args = OmegaConf.from_cli()
    input_args = OmegaConf.create({**DEFAULT_CONFIG, **dict(cl_args)})

    dict_args = {
        'env_config': {
            'horizon': 2,
            'relaxed': True,
            'actions_payoff': [(0, 100.), (1, 50.)],
        },
        'discrete_env_config': {
            'horizon': 2,
            'actions_payoff': [(0, 100.), (1, 50.)],
            'n_signals': 2
        },
        'train_batch_size': 128,
    }
    dict_args = OmegaConf.create(dict_args)
    run(dict_args, input_args)