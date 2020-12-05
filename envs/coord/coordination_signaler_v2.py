"""
    Code for Example 1 of https://arxiv.org/pdf/1912.07712.pdf

    Coordination game with one signaler, one opponent, and two team members.
    The game goes as follows:
    1) the signaler randomly picks a signal. This is observed only by the two team members.
    2) The opponent and the two team members independently choose an action.
    3) Players observe the episode payoffs

    Payoff structure of the game

    * Team mebers' payoffs:
        Opponent action A:
            - Team's payoff matrix
                    A     B
                A | 90 | 0 |
                B |  0  | 0 |

        Opponent action B:
            - Team's payoff matrix
                    A     B
                A |  0 | 0   |
                B |  0 | 30  |

    * The opponent's payoff is -(team's payoff).
    * The signaler payoff is always 0.

"""

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete, Dict, MultiDiscrete
from utils.utils import is_dict, is_list
import numpy as np


# -- DEFAULTS

# actions of the opponent and of the team members
ACTIONS_PAYOFF = [(0, 90), (1, 30)]

NUM_SIGNALS = 3


# the signaler plays, and then the others
HORIZON = 2
# ---------------------------------------------------------


class CoordinationSignalerImperfectInfo(MultiAgentEnv):
    def __init__(self, env_config=dict()):
        self.signaler = "signaler"
        self.opponent = "opponent"
        self.team1 = "t1"
        self.team2 = "t2"
        self.team = "team"

        # -- build the environment with custom config
        assert is_dict(env_config), \
            "env_config should be a dict or None"
        self._horizon = env_config.get("horizon", HORIZON)
        self._actions_payoff = env_config.get("actions_payoff", ACTIONS_PAYOFF)
        self._num_signals = env_config.get("n_signals", NUM_SIGNALS)
        self.signal_space = Discrete(self._num_signals)
        self.num_moves = 0
        self._n_actions = len(self._actions_payoff)
        self.signal = None

        assert is_list(self._actions_payoff), \
            'actions_payoff should be a list of tuples = {(action_0, payoff_0) ...}'
        assert len(self._actions_payoff) >= 2, \
            'actions_payoff should have len >= 2'
        assert all([is_list(el) and len(el) == 2
                    for el in self._actions_payoff]), \
            'actions_items should be a list of tuples with len 2, %s' % self._actions_payoff

        self._actions_payoff = {el[0]: el[1] for el in self._actions_payoff}

        self.observation_space_signaled = Dict({
            'obs': MultiDiscrete([2]*self._n_actions*self._horizon),
            'signal': MultiDiscrete([2]*self._num_signals)
        })

        self.observation_space_unsignaled = MultiDiscrete([2]*self._n_actions*self._horizon)


        self.action_space = Discrete(self._n_actions)

        self.prev_moves = {
            self.team: np.zeros(shape=(self._horizon,)),
            self.opponent: np.zeros(shape=(self._horizon,))
        }

    def reset(self):
        self.num_moves = 0
        self.signal = None
        self.prev_moves = {
            self.team: np.zeros(shape=(self._horizon,)),
            self.opponent: np.zeros(shape=(self._horizon,))
        }
        return {
            self.signaler: {'obs': 0},
        }

    def _compute_rew(self, action_dict):
        # check if all actions are equal
        done_flag = self.num_moves >= self._horizon
        moves_set = set(self.prev_moves[self.team]).union(set([self.prev_moves[self.opponent][0]]))
        success = len(moves_set) == 1

        if not done_flag:
            rew = {
                self.team1: 0,
                self.team2: 0
            }
        else:
            if not success:
                payoff = 0
            else:
                payoff = self._actions_payoff[action_dict[self.team1]]
            rew = {
                self.opponent: -payoff,
                self.team1: payoff/2,
                self.team2: payoff/2
            }

        return done_flag, rew

    def step(self, action_dict):
        sig = np.zeros(self._num_signals)
        if self.signaler in action_dict:
            self.signal = action_dict[self.signaler]
            assert self.signal in self.signal_space
            sig[self.signal] = 1

            obs = {
                self.opponent: {'obs': self._one_hot(self.prev_moves[self.opponent])},
                self.team1: {
                    "obs": self._one_hot(self.prev_moves[self.team]),
                    "signal": sig
                },
                self.team2: {
                    'obs': self._one_hot(self.prev_moves[self.team]),
                    'signal': sig
                }
            }
            done = False
            rew = {
                self.opponent: 0,
                self.team1: 0,
                self.team2: 0,
            }
        else:
            sig[self.signal] = 1
            if self.opponent in action_dict:
                self.prev_moves[self.opponent][self.num_moves] = action_dict[self.opponent]+1
            self.prev_moves[self.team][self.num_moves] = action_dict[self.team1]+1
            self.prev_moves[self.team][self.num_moves+1] = action_dict[self.team2]+1
            obs = {
                self.team1: {
                    "obs": self._one_hot(self.prev_moves[self.team]),
                    "signal": sig
                },
                self.team2: {
                    'obs': self._one_hot(self.prev_moves[self.team]),
                    'signal': sig
                }
            }
            self.num_moves += 2
            done, rew = self._compute_rew(action_dict)
            if done:
                obs.update({
                    self.opponent: {'obs': self._one_hot(self.prev_moves[self.opponent])},
                })

        done = {
            "__all__": done,
        }

        return obs, rew, done, {}

    def _one_hot(self, moves_arr):
        r = np.zeros((self._n_actions*self._horizon,))
        for h in range(self._horizon):
            if moves_arr[h] != 0:
                r[2*h + int(moves_arr[h])-1] = 1.
        return r