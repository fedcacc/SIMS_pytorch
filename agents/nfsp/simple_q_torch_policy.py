import logging
import numpy as np
from gym.spaces import Discrete, Dict, Tuple, Box, flatten_space
from torch.nn import MSELoss
from utils.utils import unpack_observations, unpack_train_observations
from ray.rllib.agents.dqn.simple_q_torch_policy import SimpleQTorchPolicy
from ray.rllib.agents.dqn.simple_q_tf_policy import compute_q_values
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.qmix.model import RNNModel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.rnn_sequencing import chop_into_sequences



torch, nn = try_import_torch()

F = None
if nn:
    F = nn.functional
logger = logging.getLogger(__name__)

Q_SCOPE = "q_func"
Q_TARGET_SCOPE = "target_q_func"


def build_q_models(policy, obs_space, action_space, config):
    policy.log_stats = config["log_stats"]
    if policy.log_stats:
        policy.stats_dict = {}
        policy.stats_fn = config["stats_fn"]

    if not isinstance(action_space, Discrete):
        raise UnsupportedSpaceException(
            "Action space {} is not supported for DQN.".format(action_space))
    policy.device = (torch.device("cuda")
                     if torch.cuda.is_available() else torch.device("cpu"))
    default_model = RNNModel if config["recurrent_dqn"] else FullyConnectedNetwork
    policy.q_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=action_space.n,
        model_config=config["model"],
        framework=config["framework"],
        default_model=default_model,
        name=Q_SCOPE).to(policy.device)

    policy.target_q_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=action_space.n,
        model_config=config["model"],
        framework=config["framework"],
        default_model=default_model,
        name=Q_TARGET_SCOPE).to(policy.device)

    policy.q_func_vars = policy.q_model.variables()
    policy.target_q_func_vars = policy.target_q_model.variables()

    return policy.q_model


def build_q_model_and_distribution_comp(policy, obs_space, action_space, config):
    # Keys of the observation space that must be used at train and test time
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

    sp = Dict({
        k: train_obs_space.spaces[k]
        for k in policy.train_obs_keys
    })
    policy.real_train_obs_space = flatten_space(sp)
    policy.real_train_obs_space.original_space = sp
    policy.n_actions = action_space.n

    model_space = Dict({
        k: obs_space.spaces[k]
        for k in policy.test_obs_keys if k != "action_mask" and k != "signal"
    })
    return build_q_models(policy, flatten_space(model_space), action_space, config), \
           TorchCategorical


def build_q_losses(policy, model, dist_class, train_batch):
    if policy.config["recurrent_dqn"] is True:
        return build_q_losses_recurrent(policy, model, dist_class, train_batch)
    else:
        return build_q_losses_normal(policy, model, dist_class, train_batch)


def compute_sequence_q_values(policy, model, obs, explore, is_training):
    B = obs.size(0)
    T = obs.size(1)

    out = []
    h = [s.expand([B, -1]) for s in model.get_initial_state()]
    for t in range(T):
        o = obs[:, t, :]
        obs_dict = {'obs': o, "is_training": is_training}
        q, h = model(obs_dict, h)
        out.append(q)

    return torch.stack(out, dim=1)


def build_q_losses_recurrent(policy, model, dist_class, samples):
    # Observations
    obs_batch, action_mask, _, _ = unpack_train_observations(policy, samples[SampleBatch.CUR_OBS], policy.device)
    next_obs_batch, next_action_mask, _, _ = unpack_train_observations(policy, samples[SampleBatch.NEXT_OBS],
                                                                       policy.device)
    rewards = samples[SampleBatch.REWARDS]

    # Obtain sequences
    input_list = [
        rewards, action_mask, next_action_mask,
        samples[SampleBatch.ACTIONS], samples[SampleBatch.DONES],
        obs_batch, next_obs_batch
    ]
    output_list, _, seq_lens = \
        chop_into_sequences(
            episode_ids=samples[SampleBatch.EPS_ID],
            unroll_ids=samples[SampleBatch.UNROLL_ID],
            agent_indices=samples[SampleBatch.AGENT_INDEX],
            feature_columns=input_list,
            state_columns=[],  # RNN states not used here
            max_seq_len=policy.config["model"]["max_seq_len"],
            dynamic_max=True)
    (rew, action_mask, next_action_mask, act, dones, obs,
     next_obs) = output_list

    B, T = len(seq_lens), max(seq_lens)

    def to_batches(arr, dtype):
        new_shape = [B, T] + list(arr.shape[1:])
        return torch.as_tensor(
            np.reshape(arr, new_shape), dtype=dtype, device=policy.device)

    rewards = to_batches(rew, torch.float)
    actions = to_batches(act, torch.long)
    obs = to_batches(obs, torch.float).reshape(
        [B, T, -1])
    action_mask = to_batches(action_mask, torch.float)
    next_obs = to_batches(next_obs, torch.float).reshape(
        [B, T, -1])
    next_action_mask = to_batches(next_action_mask, torch.float)
    terminated = to_batches(dones, torch.float).unsqueeze(2).expand(
        B, T, 1)
    filled = np.reshape(
        np.tile(np.arange(T, dtype=np.float32), B),
        [B, T]) < np.expand_dims(seq_lens, 1)
    mask = torch.as_tensor(
        filled, dtype=torch.float, device=policy.device).unsqueeze(2).expand(
        B, T, 1)

    q_t = compute_sequence_q_values(policy,
                                    policy.q_model,
                                    obs,
                                    explore=False,
                                    is_training=True)
    chosen_action_qvals = torch.gather(
        q_t, dim=-1, index=actions.unsqueeze(-1))

    q_tp1 = compute_sequence_q_values(policy,
                                      policy.target_q_model,
                                      next_obs,
                                      explore=False,
                                      is_training=True)
    ignore_action_tp1 = (next_action_mask == 0) & (mask == 1)
    q_tp1[ignore_action_tp1] = -np.inf
    target_max_qvals = q_tp1.max(dim=-1)[0]
    targets = rewards.squeeze(-1) + policy.config["gamma"] * (1 - terminated.squeeze(-1)) * target_max_qvals
    td_error = (chosen_action_qvals.squeeze(-1) - targets.detach())
    mask = mask.squeeze(-1).expand_as(td_error)
    masked_td_error = td_error * mask
    loss = (masked_td_error ** 2).sum() / mask.sum()
    policy.td_error = td_error
    return loss

def build_q_losses_normal(policy, model, dist_class, train_batch):
    # q network evaluation
    obs, mask, _, _ = unpack_train_observations(policy, train_batch[SampleBatch.CUR_OBS], policy.device)
    obs_tp1, mask_tp1, _, _ = unpack_train_observations(policy, train_batch[SampleBatch.NEXT_OBS], policy.device)
    q_t = compute_q_values(
        policy,
        policy.q_model,
        obs,
        explore=False,
        is_training=True)

    # target q network evalution
    q_tp1 = compute_q_values(
        policy,
        policy.target_q_model,
        obs_tp1,
        explore=False,
        is_training=True)

    if isinstance(policy.action_space, Tuple) and len(policy.action_space.spaces) == 1:
        ac_space = policy.action_space.spaces[0]
    else:
        ac_space = policy.action_space

    # q scores for actions which we know were selected in the given state.
    one_hot_selection = F.one_hot(train_batch[SampleBatch.ACTIONS],
                                  ac_space.n)
    if policy.requires_tupling or len(one_hot_selection.shape) == 3:
        one_hot_selection = one_hot_selection.squeeze(1)
    x = q_t*one_hot_selection
    q_t_selected = torch.sum(x, 1)

    # compute estimate of best possible value starting from state at t + 1
    dones = train_batch[SampleBatch.DONES].float()
    ignore_action_tp1 = (mask_tp1 == 0)
    q_tp1[ignore_action_tp1] = -np.inf

    q_tp1_best_one_hot_selection = F.one_hot(
        torch.argmax(q_tp1, 1), ac_space.n)
    x = q_tp1 * q_tp1_best_one_hot_selection
    x[torch.isnan(x)] = 0.
    q_tp1_best = torch.sum(x, 1)
    q_tp1_best_masked = (1.0 - dones) * q_tp1_best

    # compute RHS of bellman equation
    q_t_selected_target = (train_batch[SampleBatch.REWARDS].reshape(-1) +
                           policy.config["gamma"] * q_tp1_best_masked)

    # Compute the error (Square/Huber).
    td_error = q_t_selected - q_t_selected_target.detach()
    policy.td_error = td_error
    loss_fn = MSELoss()
    loss = loss_fn(q_t_selected, q_t_selected_target)

    if policy.log_stats:
        if policy.stats_fn is not None and False:
            policy.stats_fn(policy, obs, mask)
    return loss


def action_sampler_fn(policy, model, obs, state=None, explore=None, timestep=None):
    policy.exploration.before_compute_actions(
        explore=explore, timestep=timestep
    )
    obs, action_mask, _, _ = unpack_observations(policy, obs, policy.device)
    state = policy.get_initial_state() if state == [] else state
    state = [torch.as_tensor(s, dtype=torch.float, device=policy.device) for s in state]
    unmasked_dist_input, state = model({'obs': obs}, state)

    masked_dist_input = unmasked_dist_input.clone()
    masked_dist_input[action_mask == 0.0] = -float('inf')
    dist_class = policy.dist_class
    action_dist = dist_class(masked_dist_input, model)
    state = [s.cpu().numpy() for s in state]
    actions, logp = \
        policy.exploration.get_exploration_action(
            action_distribution=action_dist,
            timestep=timestep,
            explore=explore)

    policy.q_values = unmasked_dist_input
    if policy.requires_tupling:
        actions = actions.unsqueeze(1).tolist()
        logp = logp.unsqueeze(1)

    return actions, logp, state


def extra_grad_process(policy, opt, loss):
    if policy.log_stats:
        return {"td_error_loss": loss.item(), **policy.stats_dict}
    else:
        return {"td_error_loss": loss.item()}


def optimizer_fn(policy, config):
    return torch.optim.Adam(policy.model.parameters(), lr=config['lr'])


SimpleQTorchPolicyMSE = SimpleQTorchPolicy.with_updates(
    name="SimpleQPolicyMSE",
    loss_fn=build_q_losses,
    action_sampler_fn=action_sampler_fn,
    make_model_and_action_dist=build_q_model_and_distribution_comp,
    extra_grad_process_fn=extra_grad_process,
    optimizer_fn=optimizer_fn
)