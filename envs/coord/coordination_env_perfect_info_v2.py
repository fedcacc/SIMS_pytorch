"""
    Coordination game with one opponent, and two team members.
    The game goes as follows:
    2) The opponent and the first team member independently choose an action.
    3) Second team member observes first team member's action and plays accordingly
    4) First and second team players play alternatively observing all moves of the other player
    4) All players observe payoffs

    Payoff structure of the game

    * Team members' payoffs: positive if all players play the same move, zero otherwise

    * The opponent's payoff is -(team's payoff).

    Observations:

    * Team members: a dict with record of moves for every team player (encoding is done through a matrix of
                              boolean values of the shape (n_of_total_turns, n_of_actions).

    * Opponent: a singleton.


"""

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from utils.utils import is_dict, is_list
from gym.spaces import Discrete, Dict, MultiDiscrete, Space
import numpy as np


# -- DEFAULTS

# actions of the opponent and of the team members
ACTIONS_PAYOFF = [(0, 90), (1, 30)]

# the signaler plays, then the first team player plays with the opponent and then the second plays
HORIZON = 2
# ---------------------------------------------------------


class BoolBox(Space):
    """Matrix of boolean values with defined shape"""

    def __init__(self, shape):
        self.shape = shape

        super(BoolBox, self).__init__(self.shape)

    def sample(self):
        """ Return a random sample from the space"""
        return np.random.randint(low=0, high=2, size=self.shape)

    def null_value(self):
        """ Return the space null value"""
        null = np.zeros(self.shape)
        return null

    def contains(self, x):
        return x.shape == self.shape


class CoordinationEnvPerfectInfo(MultiAgentEnv):
    def __init__(self, env_config=dict()):
        self.opponent = "opponent"
        self.team1 = "t1"
        self.team2 = "t2"

        self.team = "team"

        # -- build the environment with custom config
        assert is_dict(env_config), \
            "env_config should be a dict or None"
        self._horizon = env_config.get("horizon", HORIZON)
        self._actions_payoff = env_config.get("actions_payoff", ACTIONS_PAYOFF)

        assert is_list(self._actions_payoff), \
            'actions_payoff should be a list of tuples = {(action_0, payoff_0) ...}'
        assert len(self._actions_payoff) >= 2, \
            'actions_payoff should have len >= 2'
        assert all([is_list(el) and len(el) == 2
                    for el in self._actions_payoff]), \
            'actions_items should be a list of tuples with len 2, %s' % self._actions_payoff

        self._actions_payoff = {el[0]: el[1] for el in self._actions_payoff}

        # compute environment params
        self._n_actions = len(self._actions_payoff)

        self.observation_space = Dict({
            'obs': MultiDiscrete([2]*self._n_actions*self._horizon)
        })
        self.action_space = Discrete(self._n_actions)

        # initialize game state
        self.num_moves = 0
        self.prev_moves = {
            self.team: np.zeros(shape=(self._horizon,)),
            self.opponent: np.zeros(shape=(self._horizon,))
        }

    def reset(self):
        self.num_moves = 0
        self.prev_moves = {
            self.team: np.zeros(shape=(self._horizon,)),
            self.opponent: np.zeros(shape=(self._horizon,))
        }
        return {
            self.opponent: {
                'obs': self._one_hot(self.prev_moves[self.opponent])},
            self.team1: {
                'obs': self._one_hot(self.prev_moves[self.team])}
        }

    def seed(self, seed):
        pass

    def _compute_rew(self):
        # check if all actions are equal
        moves_set = set(self.prev_moves[self.team]).union(set([self.prev_moves[self.opponent][0]]))

        success = len(moves_set) == 1


        if not success:
            payoff = 0.
        else:
            payoff = self._actions_payoff[int(moves_set.pop()) - 1]

        return {
            self.opponent: -payoff,
            self.team1: payoff,
            self.team2: payoff,
        }

    def step(self, action_dict):
        if self.num_moves == 0:
            # Initial turn
            assert self.opponent in action_dict and self.team1 in action_dict \
                   and self.team2 not in action_dict, "Invalid Move"

            self.prev_moves[self.team][self.num_moves] = action_dict[self.team1] +1
            self.prev_moves[self.opponent][self.num_moves] = action_dict[self.opponent] +1

            obs = {self.team2: {'obs': self._one_hot(self.prev_moves[self.team])}}
            rew = {self.team2: 0}

        elif (self.num_moves %2) == 1:
            assert self.team2 in action_dict and self.team1 not in action_dict \
                   and self.opponent not in action_dict, "Invalid Move"

            self.prev_moves[self.team][self.num_moves] = action_dict[self.team2] +1

            obs = {self.team1: {'obs': self._one_hot(self.prev_moves[self.team])}}
            rew = {self.team1: 0}

        else:
            assert self.team1 in action_dict and self.team2 not in action_dict \
                   and self.opponent not in action_dict, "Invalid Move"

            self.prev_moves[self.team][self.num_moves] = action_dict[self.team1] +1
            obs = {self.team2: {'obs': self._one_hot(self.prev_moves[self.team])}}
            rew = {self.team2: 0}

        self.num_moves += 1
        done_flag = self.num_moves >= self._horizon
        done = {
            "__all__": done_flag
        }
        if done_flag:
            obs = {
                self.team1: {'obs': self._one_hot(self.prev_moves[self.team])},
                self.team2: {'obs': self._one_hot(self.prev_moves[self.team])},
                self.opponent:  {'obs': self._one_hot(self.prev_moves[self.opponent])}
            }
            rew = self._compute_rew()

        return obs, rew, done, {}

    def _one_hot(self, moves_arr):
        r = np.zeros((self._n_actions*self._horizon,))
        for h in range(self._horizon):
            if moves_arr[h] != 0:
                r[2*h + int(moves_arr[h]) - 1] = 1.
        return r


