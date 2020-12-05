from ray.rllib.policy.policy import Policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.annotations import override
import logging
from utils.utils import unpack_train_observations, unpack_observations
from models.rnn_multiagent_model import _get_size
import numpy as np
from ray.rllib.models.catalog import ModelCatalog
from models.fc_multiagent_model import  MultiAgentFullyConnectedNetwork
from gym.spaces import Dict, Tuple, Discrete, MultiDiscrete, flatten_space
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.metrics import LEARNER_STATS_KEY
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

torch, nn = try_import_torch(error=True)

logger = logging.getLogger(__name__)

ENV_STATE = "state"


class Signaler():
    def draw(self, n):
        return np.identity(n)


class SIMSPolicy(Policy):
    """"
    Signal Mediated Strategies (SIMS) Policy
    TODO: detailed description
    """
    def __init__(self, obs_space, action_space, config):
        """ Only Dict observation spaces are allowed"""
        super().__init__(obs_space, action_space, config)

        # General configs
        self.framework = "torch"
        self.n_agents = len(obs_space.original_space.spaces)
        assert self.n_agents == 2, "At this moment only two-team agents are supported {} is not a valid number"\
            .format(self.n_agents)  # TODO (fede) increase the number of team agents to arbitrary
        self.n_actions = action_space.spaces[0].n
        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))
        self.beta = config["beta"]
        self.gamma = self.beta * config['factor_ent']
        self.n_train_signals = config['n_train_signals']


        # Flag that regulates whether to log statistics
        self.log_stats = config["log_stats"]
        self.eval_fn = config["stats_fn"]

        # Keys from the observation space that must be used at training and test time
        self.train_obs_keys = config["train_obs_keys"]
        self.test_obs_keys = config["test_obs_keys"]

        # Get and validate real observation space
        # (Assumed uniform observation and action spaces for the players)
        agent_obs_space = obs_space.original_space.spaces[0]
        assert isinstance(agent_obs_space, Dict), "Invalid observation space"
        assert "signal" in agent_obs_space.spaces, "Observation space must contain field 'signal'" + \
                                                   str(agent_obs_space.spaces)

        self.real_test_obs_space = flatten_space(Tuple([agent_obs_space] * self.n_agents))
        self.real_test_obs_space.original_space = Tuple([agent_obs_space] * self.n_agents)
        self.test_obs_size = _get_size(self.real_test_obs_space)
        self.signal_size = _get_size(agent_obs_space.spaces["signal"])
        if "action_mask" in agent_obs_space.spaces:
            mask_shape = tuple(agent_obs_space.spaces["action_mask"].shape)
            assert mask_shape == (self.n_actions,), "Invalid shape for action mask"

        # Get and validate train observation space
        # (Assumed uniform observation and action spaces for the players)
        if config["train_obs_space"] is None:
            train_obs_space = agent_obs_space
        else:
            train_obs_space = config["train_obs_space"]
            if isinstance(train_obs_space, Tuple):
                train_obs_space = train_obs_space.spaces[0]

        self.real_train_obs_space = flatten_space(Tuple([train_obs_space]*self.n_agents))
        self.real_train_obs_space.original_space = Tuple([train_obs_space]*self.n_agents)
        agent_obs_space_signaled = Tuple([
            Dict({**{
                k: agent_obs_space.spaces[k]
                for k in self.test_obs_keys if k != "signal" and k != "action_mask"
            }, **{
                "signal": MultiDiscrete([2]*self.n_train_signals)
            }})
        ]*self.n_agents)

        # training signaler
        self.signaler = Signaler()

        # Models
        self.model = ModelCatalog.get_model_v2(
            agent_obs_space_signaled,
            action_space,
            self.n_actions,
            config["model"],
            framework="torch",
            name="SignaledFCNet",
            default_model=MultiAgentFullyConnectedNetwork
        )

        self.signaler_model = ModelCatalog.get_model_v2(
            MultiDiscrete([2]),
            Discrete(self.n_train_signals),
            self.n_train_signals,
            config['sig_model'],
            framework="torch",
            name="SignalerNet",
            default_model=FullyConnectedNetwork
        )

        # exploration
        self.exploration = self._create_exploration()

        # Setup the optimizer and loss TODO (fede): add custom choice possibility for optimiser
        self.model_optimiser = config["model_optimiser"]["type"](
            self.model.parameters(),
            lr=config["model_optimiser"]["lr"]
        )
        self.signaler_optimiser = config["sig_model_optimiser"]["type"](
            self.signaler_model.parameters(),
            lr=config["sig_model_optimiser"]["lr"]
        )

        # lr/beta scheduling algorithm (experimental)
        # self._curr_ts = 0
        # self._prev_lr_update = 0
        # self._prev_beta_update = 0

        self.classification_loss = nn.CrossEntropyLoss()

        def EntropyLoss(dist, reduce=True):
            S = nn.Softmax(dim=-1)
            LS = nn.LogSoftmax(dim=-1)
            b = S(dist) * LS(dist)
            b = torch.sum(b, 1)
            if reduce:
                b = torch.mean(b)
            return b

        self.entropy_loss = EntropyLoss

    @override(Policy)
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None,
                        info_batch=None, episodes=None, explore=None, timestep=None, **kwargs):

        obs_batch, action_mask, _, signal_batch = unpack_observations(self, obs_batch, self.device)
        B, n_agents = obs_batch.size(0), obs_batch.size(1)

        with torch.no_grad():
            # Compute policies
            policies, _ = self.model({'obs': torch.cat((obs_batch, signal_batch), dim=2)})
            policies = policies.reshape(B, n_agents, -1)

            # Mask out invalid actions
            masked_policies = policies.clone()
            masked_policies[action_mask==0.0] = -float('inf')

            # Compute actions
            from torch.distributions import Categorical
            dist = Categorical(logits=masked_policies)
            probs = dist.probs
            actions = dist.sample().long().cpu().numpy()

            sampled_probs = probs[:, np.arange(n_agents), actions].squeeze(1).cpu().numpy()
            pr = []
            for ag in range(n_agents):
                pr += [s for s in sampled_probs[:,ag]]

        return tuple(actions.transpose([1, 0])), [], {}

    @override(Policy)
    def compute_log_likelihoods(self, actions, obs_batch, state_batches=None, prev_action_batch=None,
                                prev_reward_batch=None):
        obs_batch, action_mask,sig_batch = self._unpack_observation(obs_batch)
        return np.zeros(obs_batch.size()[0])

    @override(Policy)
    def learn_on_batch(self, samples):
        # Update learning rates
        # lr = self._get_lr()
        # for p in self.model_optimiser.param_groups:
        #    p["lr"] = lr
        # for p in self.signaler_optimiser.param_groups:
        #    p["lr"] = lr

        # Update beta param
        # self.beta = self._get_beta()
        # Zero optimizers
        self.model_optimiser.zero_grad()
        self.signaler_optimiser.zero_grad()

        # Get data from batch
        obs_batch, mask, _, _ = unpack_train_observations(self, samples[SampleBatch.CUR_OBS], self.device)
        actions = torch.as_tensor(
            np.array([self._encode_joint_action(a) for a in samples[SampleBatch.ACTIONS]]),
            dtype=torch.long, device=self.device)
        BATCH = obs_batch.size(0)
        N_PLAYERS = self.n_agents
        FULL_BATCH = BATCH*self.n_train_signals

        # Compute nets inputs by mixing obs with signals
        obs_signaled = torch.repeat_interleave(obs_batch, self.n_train_signals, 0)\
            .reshape(BATCH*self.n_train_signals, self.n_agents, -1)
        signals = torch.as_tensor(
            np.tile(self.signaler.draw(self.n_train_signals), BATCH).T.reshape(FULL_BATCH, -1),
            dtype=torch.float, device=self.device)
        signals = torch.stack([signals]*self.n_agents, dim=1)
        full_obs = torch.cat([obs_signaled, signals], dim=2)

        # Compute losses (entropy and average difference)
        # 1. compute joint distributions for each signal
        players_actions, _ = self.model({'obs': full_obs})
        players_actions = players_actions.reshape(FULL_BATCH, N_PLAYERS, -1)
        mask = torch.stack([mask] * self.n_train_signals, dim=1) \
            .reshape(FULL_BATCH, N_PLAYERS, -1)
        # players_actions[mask == 0.] = -float('inf')

        pl1_probs = nn.Softmax(1)(players_actions[:, 0, :])
        pl2_probs = nn.Softmax(1)(players_actions[:, 1, :])
        joint_probs = torch.bmm(pl1_probs.unsqueeze(2), pl2_probs.unsqueeze(1))\
            .reshape(BATCH, self.n_train_signals, -1)
        # 2. marginalise with signal probabilities (sum 1e-8 at the end to avoid nans)
        sig_probs_in = torch.as_tensor([[1.]]*BATCH, dtype=torch.float, device=self.device)
        sig_logits, _ = self.signaler_model({'obs': sig_probs_in})
        sig_probs = nn.Softmax(1)(sig_logits).unsqueeze(-1)
        joint_averaged = torch.log(torch.sum(joint_probs*sig_probs, 1)+1e-8)

        # Compute losses
        pl1_entropy = self.entropy_loss(players_actions[:, 0], reduce=False)\
            .reshape(BATCH, self.n_train_signals, -1)
        pl2_entropy = self.entropy_loss(players_actions[:, 1], reduce=False)\
             .reshape(BATCH, self.n_train_signals, -1)

        entropy = -torch.mean(torch.sum(pl1_entropy*sig_probs.detach(), 1)) \
                  - torch.mean(torch.sum(pl2_entropy*sig_probs.detach(), 1))

        # entropy = -(self.entropy_loss(players_actions[:,0])) - (self.entropy_loss(players_actions[:,1]))

        cl_loss = self.classification_loss(joint_averaged, actions.squeeze())
        sig_entropy = -(self.entropy_loss(sig_logits))
        loss = cl_loss + self.beta*entropy + self.gamma*sig_entropy

        # Do one optimiser step
        loss.backward()

        with torch.no_grad():
            sig_model_params = list(self.signaler_model.parameters())
            sig_grad_norm = torch.norm(torch.as_tensor([
                torch.norm(p.grad) for p in sig_model_params if p.grad is not None]))

            model_params = list(self.model.parameters())
            model_grad_norm = torch.norm(torch.as_tensor([
                torch.norm(p.grad) for p in model_params if p.grad is not None]))

        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.signaler_model.parameters(), 0.5)

        self.model_optimiser.step()
        self.signaler_optimiser.step()

        # Base statistics
        stats = {
            "classification_loss": cl_loss.item(),
            "strat_entropy_loss": entropy.item(),
            "signal_entropy_loss": sig_entropy.item(),
            "sig_model_grad_norm": sig_grad_norm.item(),
            "model_grad_norm": model_grad_norm.item()
        }
        # Signal probabilities
        stats.update({
            "sig={}_prob".format(s): float(sig_probs[0, s].squeeze().cpu().detach().numpy())
            for s in range(self.n_train_signals)
        })

        if self.log_stats:
            # Statistics: compute player marginal losses to log them
            with torch.no_grad():
                pl1_probs_eval = pl1_probs.reshape(BATCH, self.n_train_signals, -1)
                pl2_probs_eval = pl2_probs.reshape(BATCH, self.n_train_signals, -1)
                pl1_averaged_eval = torch.log(torch.sum(pl1_probs_eval * sig_probs, 1) + 1e-8)
                pl2_averaged_eval = torch.log(torch.sum(pl2_probs_eval * sig_probs, 1) + 1e-8)
                actions_single = torch.as_tensor(np.array(samples[SampleBatch.ACTIONS]),
                                                 dtype=torch.long, device=self.device)
                loss_pl1 = self.classification_loss(pl1_averaged_eval, actions_single[:, 0].squeeze())
                loss_pl2 = self.classification_loss(pl2_averaged_eval, actions_single[:, 1].squeeze())
            stats.update({
                "pl1_class_loss": loss_pl1.item(),
                "pl2_class_loss": loss_pl2.item()
            })
        if self.eval_fn is not None:
            dd = self.eval_fn(self, obs_batch, mask)
        else:
            dd = {}

        return {LEARNER_STATS_KEY: {**stats, **dd}}

    def update_target(self):
        pass

    def _get_lr(self):
        # (Experimental) for lr scheduling
        curr_lr = self.model_optimiser.param_groups[0]['lr']
        scheduling_configs = self.config['lr_scheduling']
        if self._curr_ts - self._prev_lr_update >= scheduling_configs['shrink_every_iters']:
            self._prev_lr_update = self._curr_ts
            lr = curr_lr*scheduling_configs['shrink_factor']
        else:
            lr = curr_lr
        return lr

    def _get_beta(self):
        # (Experimental) for beta scheduling
        curr_beta = self.beta
        scheduling_configs = self.config["beta_scheduling"]
        if curr_beta == scheduling_configs['final_beta']:
            return curr_beta
        if self._curr_ts - self._prev_beta_update >= scheduling_configs["shrink_every_iters"]:
            self._prev_beta_update = self._curr_ts
            beta = curr_beta*scheduling_configs["shrink_factor"]
        else:
            beta = curr_beta
        return beta

    def set_timestep(self, ts):
        # for lr/beta scheduling
        self._curr_ts = ts

    @override(Policy)
    def get_weights(self):
        return {
            "model": self._cpu_dict(self.model.state_dict()),
            "signaler": self._cpu_dict(self.signaler_model.state_dict())
        }

    @override(Policy)
    def set_weights(self, weights):
        self.model.load_state_dict(self._device_dict(weights["model"]))
        self.signaler_model.load_state_dict(self._device_dict(weights["signaler"]))

    def _device_dict(self, state_dict):
        return {
            k: torch.as_tensor(v, device=self.device)
            for k, v in state_dict.items()
        }

    @staticmethod
    def _cpu_dict(state_dict):
        return {k: v.cpu().detach().numpy() for k, v in state_dict.items()}

    def _encode_joint_action(self, action):
        num_pl = len(action)
        assert num_pl == self.n_agents
        n = 0
        for i in range(num_pl):
            n += (self.n_actions**i) * action[num_pl-1-i]
        return n
