from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from models.rnn_multiagent_model import _get_size
from gym.spaces import Dict, Box
from gym.spaces.utils import flatten_space
import numpy as np

torch, nn = try_import_torch()


class MultiAgentFullyConnectedNetwork(TorchModelV2, nn.Module):
    """
        Model for multi-agent FCN synchronous execution
        obs spaces and action spaces are tuples
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_sizes = _get_size(obs_space.spaces[0])
        self.n_players = len(obs_space.spaces)
        self.n_actions = action_space.spaces[0].n

        os = []
        for pl in range(self.n_players):
            os.append(flatten_space(obs_space.spaces[pl]))

        self.pl_models = {
            pl: FullyConnectedNetwork(os[pl], action_space.spaces[pl],
                                      action_space.spaces[pl].n, model_config, name)
            for pl in range(self.n_players)
        }


        # Set models as attributes to obtain parameters
        for pl in range(self.n_players):
            setattr(self, "model_{}".format(pl), self.pl_models[pl])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Expects a batch of the shape [B, n_agents, obs_size]
        B, n_agents = input_dict['obs'].size(0), input_dict['obs'].size(1)
        inputs = [input_dict['obs'][:, pl, :].float() for pl in range(self.n_players)]
        outputs = [self.pl_models[pl]({'obs': inputs[pl]})[0] for pl in range(self.n_players)]

        return torch.stack(outputs, dim=1).reshape(B*n_agents, -1), []


class MultiAgentFullyConnectedNetworkEmbedding(TorchModelV2, nn.Module):
    """
        Model for multi-agent FCN synchronous execution with embedding layers before processing
        obs spaces and action spaces are tuples.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_sizes = _get_size(obs_space.spaces[0])
        self.n_players = len(obs_space.spaces)
        self.n_actions = action_space.spaces[0].n

        os = []
        ms = []
        intermediate_space = Box(low=0, high=2, shape=(8,), dtype=np.float32)
        for pl in range(self.n_players):
            os.append(flatten_space(obs_space.spaces[pl].spaces['obs']))
            mid_space = flatten_space(
                Dict({
                    'obs': intermediate_space,
                    'signal': obs_space.spaces[pl]['signal']
                })
            )
            ms.append(mid_space)

        assert self.n_players <= 2, "Not yet supported for more than 2 players"  # TODO: make it support n_players > 2
        embed_config = {
           "fcnet_hiddens": [128, 128],
           "fcnet_activation": 'relu',
           "max_seq_len": 20
        }
        self.embed_pl_models = {
            pl: FullyConnectedNetwork(os[pl], intermediate_space,
                                      8, embed_config, "{}_embeding".format(pl))
            for pl in range(self.n_players)
        }

        self.pl_models = {
            pl: FullyConnectedNetwork(ms[pl], action_space.spaces[pl],
                                      action_space.spaces[pl].n, model_config, name)
            for pl in range(self.n_players)
        }

        # Set models as attributes to obtain parameters
        for pl in range(self.n_players):
            setattr(self, "embed_model_{}".format(pl), self.embed_pl_models[pl])
            setattr(self, "model_{}".format(pl), self.pl_models[pl])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Expects a batch of the shape [B, n_agents, obs_size]
        B, n_agents = input_dict['obs'].size(0), input_dict['obs'].size(1)
        assert input_dict['obs'].size(0) == input_dict['signal'].size(0)
        embed_inputs = [input_dict['obs'][:, pl, :].float() for pl in range(self.n_players)]
        embed_outputs = [self.embed_pl_models[pl]({'obs': embed_inputs[pl]})[0] for pl in range(self.n_players)]
        signal_input = [input_dict['signal'][:, pl, :].float() for pl in range(self.n_players)]
        mixed_input = [torch.cat([embed_outputs[pl], signal_input[pl]], dim=1) for pl in range(self.n_players)]

        outputs =[self.pl_models[pl]({'obs': mixed_input[pl]})[0] for pl in range(self.n_players)]
        return torch.stack(outputs, dim=1).reshape(B*n_agents, -1), []


        
        

