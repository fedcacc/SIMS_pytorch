from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
from ray.rllib.agents.qmix.model import RNNModel

torch, nn = try_import_torch()


class RNNMultiagentModel(TorchModelV2, nn.Module):
    """
        Model for multi-agent RNN synchronous execution
        obs spaces and action spaces are tuples
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.obs_sizes = _get_size(obs_space.spaces[0])
        self.rnn_hidden_dim = model_config["lstm_cell_size"]

        self.n_players = len(obs_space.spaces)
        self.pl_models = {
            pl: RNNModel(obs_space.spaces[pl], action_space.spaces[pl],  # TODO: make sure that arguments are properly passed
                         num_outputs, model_config, name)
            for pl in range(self.n_players)
        }

        # Set model as attributes to obtain parameters
        for pl in self.pl_models:
            setattr(self, "model_{}".format(pl), self.pl_models[pl])


    @override(ModelV2)
    def get_initial_state(self):
        states = [self.pl_models[pl].fc1.weight.new(1, self.rnn_hidden_dim).zero_().squeeze(0)
                  for pl in range(self.n_players)]
        return [torch.stack(states)]

    @override(ModelV2)
    def forward(self, input_dict, hidden_state, seq_lens):
        B, n_agents = input_dict['obs'].size(0), input_dict['obs'].size(1)
        inputs = [input_dict['obs'][:, pl, :].float() for pl in range(self.n_players)]
        x = [nn.functional.relu(self.pl_models[pl].fc1(inputs[pl]))
             for pl in range(self.n_players)]

        h_in = [hidden_state[0][:, pl, :].reshape(-1, self.rnn_hidden_dim)
                for pl in range(self.n_players)]

        h = [self.pl_models[pl].rnn(x[pl], h_in[pl]) for pl in range(self.n_players)]
        q = [self.pl_models[pl].fc2(h[pl]) for pl in range(self.n_players)]

        # Need reshape or raises error (not fixable, must change the interface of ModelV2)
        h_out = torch.stack(h, dim=1).reshape(B*n_agents, -1)
        q_out = torch.stack(q, dim=1).reshape(B*n_agents, -1)
        return q_out, [h_out]


def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size

