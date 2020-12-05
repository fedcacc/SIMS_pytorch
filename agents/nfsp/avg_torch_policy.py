from ray.rllib.agents import with_common_config
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils import try_import_torch
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from gym.spaces import Dict, Tuple, Discrete, Box, flatten_space
from ray.rllib.utils.error import UnsupportedSpaceException
from utils.utils import unpack_observations, unpack_train_observations
import numpy as np

torch, nn = try_import_torch()
DEFAULT_CONFIG = with_common_config({})


def make_model_and_action_dist(policy, obs_space, action_space, config):
    """create model neural network"""
    policy.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))
    policy.log_stats = config["log_stats"]  # flag to log statistics
    if policy.log_stats:
        policy.stats_dict = {}
        policy.stats_fn = config["stats_fn"]

    # Keys of the observation space that must be used at train and test time ('signal' and 'mask' will be excluded
    # from the actual obs space)
    policy.train_obs_keys = config["train_obs_keys"]
    policy.test_obs_keys = config["test_obs_keys"]

    # Check whether policy observation space is inside a Tuple space
    policy.requires_tupling = False
    if isinstance(action_space, Tuple) and len(action_space.spaces) == 1:
        policy.action_space = action_space.spaces[0]
        action_space = action_space.spaces[0]
        policy.requires_tupling = True
    if not isinstance(action_space, Discrete):
        raise UnsupportedSpaceException(
            "Action space {} is not supported for DQN.".format(action_space))

    # Get real observation space
    if isinstance(obs_space, Box):
        assert hasattr(obs_space, "original_space"), "Invalid observation space"
        obs_space = obs_space.original_space
        if isinstance(obs_space, Tuple):
            obs_space = obs_space.spaces[0]
    assert isinstance(obs_space, Dict), "Invalid observation space"
    policy.has_action_mask = "action_mask" in obs_space.spaces
    assert all([k in obs_space.spaces for k in policy.train_obs_keys]), "Invalid train keys specification"
    assert all([k in obs_space.spaces for k in policy.test_obs_keys]), "Invalid test keys specification"

    # Get observation space used for training
    if config["train_obs_space"] is None:
        train_obs_space = obs_space
    else:
        train_obs_space = config["train_obs_space"]
        if isinstance(train_obs_space, Box):
            assert hasattr(train_obs_space, "original_space"), "Invalid observation space"
            train_obs_space = train_obs_space.original_space
            if isinstance(train_obs_space, Tuple):
                train_obs_space = train_obs_space.spaces[0]

    # Obs spaces used for training and testing
    sp = Dict({
        k: obs_space.spaces[k]
        for k in policy.test_obs_keys
    })

    policy.real_test_obs_space = flatten_space(sp)
    policy.real_test_obs_space.original_space = sp
    model_space = Dict({
        k: obs_space.spaces[k]
        for k in policy.test_obs_keys if k != "signal" and k != "action_mask"
    })


    sp = Dict({
        k: train_obs_space.spaces[k]
        for k in policy.train_obs_keys
    })
    policy.real_train_obs_space = flatten_space(sp)
    policy.real_train_obs_space.original_space = sp
    policy.n_actions = action_space.n
    def update_target():
        pass

    policy.update_target = update_target
    model = FullyConnectedNetwork(flatten_space(model_space), action_space, action_space.n, name="FcNet",
                                 model_config=config['model']).to(policy.device)
    return model, ModelCatalog.get_action_dist(action_space, config, framework='torch')


def optimizer_fn(policy, config):
    return torch.optim.Adam(policy.model.parameters(), lr=config['lr'])


def build_loss(policy, model, dist_class, train_batch):
    obs, mask, _, _ = unpack_train_observations(policy, train_batch[SampleBatch.CUR_OBS], policy.device)

    actions = train_batch[SampleBatch.ACTIONS]
    if policy.requires_tupling:
        actions = actions.squeeze(1)
    input_dict = {"obs": obs}
    logits, _ = model(input_dict)
    loss_fn = nn.CrossEntropyLoss()
    # logits[mask==0.] = -np.inf
    loss = loss_fn(logits, actions)
    if policy.log_stats:
        if policy.stats_fn is not None:
            policy.stats_fn(policy, obs, mask)
    return loss


def action_sampler_fn(policy, model, obs, state=None, explore=None, timestep=None):
    policy.exploration.before_compute_actions(
        explore=explore, timestep=timestep)
    obs, action_mask, _, _ = unpack_observations(policy, obs, policy.device)
    unmasked_dist_input, state = model({'obs': obs})

    masked_dist_input = unmasked_dist_input.clone()
    masked_dist_input[action_mask == 0.0] = -float('inf')
    dist_class = policy.dist_class[0]
    action_dist = dist_class(masked_dist_input, model)

    actions, logp = \
        policy.exploration.get_exploration_action(
            action_distribution=action_dist,
            timestep=timestep,
            explore=explore
        )
    if policy.requires_tupling:
        actions = actions.unsqueeze(1).tolist()
        logp = logp.unsqueeze(1)

    return actions, logp, state


def extra_grad_process(policy, opt, loss):
    if policy.log_stats:
        return {**{"classification_loss": loss.item()}, **policy.stats_dict}
    else:
        return {"classification_loss": loss.item()}


AVGPolicy = build_torch_policy(
    name="AVGTorchPolicy",
    loss_fn=build_loss,
    action_sampler_fn=action_sampler_fn,
    make_model_and_action_dist=make_model_and_action_dist,
    get_default_config=lambda: DEFAULT_CONFIG,
    extra_grad_process_fn=extra_grad_process,
    optimizer_fn=optimizer_fn,
)

