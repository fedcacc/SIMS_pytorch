from omegaconf import ListConfig, DictConfig
from omegaconf import OmegaConf
import enum
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.utils import try_import_torch
import numpy as np
torch, nn = try_import_torch()

MODE = enum.Enum("mode", "average_strategy best_response")

def get_maybe_missing_args(args, key, default=None):
    if OmegaConf.is_missing(args, key):
        return default
    else:
        return args.get(key)


def is_list(obj):
    return isinstance(obj, (list, tuple, ListConfig))


def is_dict(obj):
    return isinstance(obj, (dict, DictConfig))


def unpack_train_observations(policy, obs, device=None):
    """
        Unpack observation from training time: observation space is in policy.real_train_obs_space.
        Returns obs, mask, state, signal (None if not present)
        """
    obs_space = policy.real_train_obs_space
    if hasattr(obs_space, "original_space"):
        obs_space = obs_space.original_space
    obs_keys = policy.train_obs_keys
    num_actions = policy.n_actions
    return _unpack_general(obs_space, obs_keys, num_actions, obs, device)


def unpack_observations(policy, obs, device=None):
    """
    Unpack observation from test time: observation space is in policy.real_test_obs_space.
    Returns obs, mask, state, signal (None if not present)
    """
    obs_space = policy.real_test_obs_space
    if hasattr(obs_space, "original_space"):
        obs_space = obs_space.original_space
    obs_keys = policy.test_obs_keys
    num_actions = policy.n_actions
    return _unpack_general(obs_space, obs_keys, num_actions, obs, device)


def _unpack_general(obs_space, obs_keys, num_actions, obs, device=None):
    """Unpack observation in :obs: given arguments"""
    if isinstance(obs, dict):
        obs = obs['obs']

    if not isinstance(obs, torch.Tensor):
        obs = torch.as_tensor(obs, dtype=torch.float, device=device)

    unpacked = _unpack_obs(obs,
                           obs_space,
                           tensorlib=np)

    # Observation
    if isinstance(unpacked, list):
        n_agents = len(obs_space.spaces)
        # obs
        obs = torch.stack([torch.cat([u[k].reshape(len(obs), -1) for k in obs_keys
                                      if k != "signal" and k != "action_mask"], dim=-1)
                           for u in unpacked], dim=1).reshape(len(obs), n_agents, -1)
        # mask
        default_mask = torch.as_tensor(
            np.ones(shape=(obs.size(0), num_actions)), dtype=torch.float, device=obs.device
        )
        mask = torch.stack([u.get("action_mask", default_mask).reshape(len(obs), -1) for u in unpacked], dim=1)
        # state
        if unpacked[0].get("state", None) is not None:
            state = torch.stack([u.get("state").reshape(len(obs), -1) for u in unpacked], dim=1)\
                .reshape(len(obs), n_agents, -1)
        else:
            state = None

        # signals
        if unpacked[0].get("signal", None)is not None:
            signal = torch.stack([u.get("signal").reshape(len(obs), -1) for u in unpacked], dim=1)\
                .reshape(len(obs), n_agents, -1)
        else:
            signal = None

    else:
        obs = torch.cat([unpacked[k].reshape(len(obs), -1) for k in obs_keys
                         if k != "signal" and k != "action_mask"], dim=-1)

        # Action mask
        default_mask = torch.as_tensor(
            np.ones(shape=(obs.size(0), num_actions)), dtype=torch.float, device=obs.device
        )
        mask = unpacked.get("action_mask", default_mask)

        # State
        state = unpacked.get("state", None)

        # Signal
        signal = unpacked.get("signal", None)

    return obs, mask, state, signal
