from ray.rllib.env.multi_agent_env import MultiAgentEnv
from envs.coord.coordination_signaler_v2 import CoordinationSignalerImperfectInfo
from envs.coord.coordination_env_perfect_info_v2 import CoordinationEnvPerfectInfo


class CoordWrapperClass(MultiAgentEnv):
    """Wrapper environment for PatrollingEnv"""
    def __init__(self, args):
        from omegaconf import OmegaConf
        def seed_fn(seed):
            pass

        args = OmegaConf.create(args)
        if args.relaxed:
            if args.groups:
                env = CoordinationEnvPerfectInfo(args).with_agent_groups(
                    args.groups
                )
            else:
                env = CoordinationEnvPerfectInfo(args)
        else:
            if args.groups:
                env = CoordinationSignalerImperfectInfo(args).with_agent_groups(
                    args.groups
                )
            else:
                env = CoordinationSignalerImperfectInfo(args)
            env.seed = seed_fn

        from inspect import getmembers
        for name, val in getmembers(env):
            try:
                self.__setattr__(name, val)
            except AttributeError:
                pass
