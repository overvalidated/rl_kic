"""Catch reinforcement learning environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel


# Actions
NOOP = 0
LEFT = 1
RIGHT = 2

_Point = collections.namedtuple("Point", ["x", "y"])
_NUM_PLAYERS = 1
_N_PERSONS = 12
_N_SHIFTS=62

_GAME_TYPE = pyspiel.GameType(
    short_name="python_schedule",
    long_name="Python Schedule Game",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={}
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=10,
    max_chance_outcomes=0,
    num_players=1,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_N_PERSONS*_N_SHIFTS
)

class Environment:
    """A catch reinforcement learning environment.
    The implementation considers illegal actions: trying to move the paddle in the
    wall direction when next to a wall will incur in an invalid action and an
    error will be purposely raised.
    """

    def __init__(self, n_persons, n_shifts, oso_monthly, oso_daily, seed=None):
        self.n_persons = n_persons
        self.n_shifts = n_shifts 
        self.oso_monthly = oso_monthly # (n_persons, 1)
        self.oso_daily = oso_daily # (1, n_shifts)
        self._should_reset = True
        self._num_actions = 10

        # Discount returned at non-initial steps.
        self._discounts = [1] * self.num_players

    def reset(self):
        self.field = np.zeros((self.n_persons, self.n_shifts, 10), dtype=np.float32)
        self.shift = 0
        self.person = 0
        self._should_reset = False

        """Resets the environment."""
        observations = {
            "info_state": [self._get_observation()],
            "legal_actions": list(range(self._num_actions)),
            "current_player": 0,
        }

        return rl_environment.TimeStep(
            observations=observations,
            rewards=None,
            discounts=None,
            step_type=rl_environment.StepType.FIRST)

    def step(self, actions):
        """Updates the environment according to `actions` and returns a `TimeStep`.
        Args:
        actions: A singleton list with an integer, or an integer, representing the
            action the agent took.
        Returns:
        A `rl_environment.TimeStep` namedtuple containing:
            observation: singleton list of dicts containing player observations,
                each corresponding to `observation_spec()`.
            reward: singleton list containing the reward at this timestep, or None
                if step_type is `rl_environment.StepType.FIRST`.
            discount: singleton list containing the discount in the range [0, 1], or
                None if step_type is `rl_environment.StepType.FIRST`.
            step_type: A `rl_environment.StepType` value.
        """
        if self._should_reset:
            return self.reset()

        if isinstance(actions, list):
            action = actions[0]
        elif isinstance(actions, int):
            action = actions
        else:
            raise ValueError("Action not supported.", actions)
    
        terminal_cond = False

        # using this to calculate total hours by dot product
        hours_array = np.array([[0] + list(range(4, 13))])

        # add to field
        if action in range(10):
            if action != 0:
                self.field[self.person, self.shift, action-1] = 1 

            # check 2 shifts
            if self.shift >= 1:
                terminal_cond = (self.field[self.person, self.shift-1, 1:]>0).sum() == 1
                if terminal_cond:
                    print('SECOND SHIFT')

            # check 1001
            if (not terminal_cond) and self.shift >= 4:
                terminal_cond = np.array_equal(self.field[self.person, self.shift-4:self.shift, 1:].sum(axis=1).ravel(),
                                               np.array([1, 0, 0, 1]))
                if terminal_cond:
                    print('1001')

            # check 6 days
            if (not terminal_cond) and self.shift >= 10: #01 23 45 67 89 1011 12 13 14:
                terminal_cond = self.field[self.person, self.shift-10:self.shift, 1:].sum() >= 6
                if terminal_cond:
                    print('6')
                
            # check 42 hours
            if (not terminal_cond) and self.shift >= 14:
                terminal_cond = self.field[self.person, self.shift-14:self.shift].dot(hours_array) <= 42
                if terminal_cond:
                    print('42')
            
            # go to next person/shift, check if enough people are on shift
            if not terminal_cond:
                self.person += 1
                if self.person // self.n_persons == 1:
                    # check personnel number
                    terminal_cond = terminal_cond or self.field[:, self.shift, 1:].sum() < 3
                    if terminal_cond:
                        print('PERSONNEL N')
                    self.shift += 1
                self.person %= self.n_persons
        else:
            raise ValueError("unrecognized action ", action)

        # checking conditions, setting rewards
        if terminal_cond:
            done = True
            reward = -1.0
        elif self.shift == self.n_shifts:
            done = True
            ### TODO ДОБАВИТЬ РАСЧЕТ НАГРАДЫ
            reward = 1.0
        else:
            done = False
            reward = 0.0

        # Return observation
        step_type = (
            rl_environment.StepType.LAST if done else rl_environment.StepType.MID
        )
        self._should_reset = step_type == rl_environment.StepType.LAST

        observations = {
            "info_state": [self._get_observation()],
            "legal_actions": [range(10)],
            "current_player": 0,
        }

        return rl_environment.TimeStep(
            observations=observations,
            rewards=[reward],
            discounts=self._discounts,
            step_type=step_type)

    def _get_observation(self):
        field = np.zeros(self.field.shape[:2] + (11,))
        field[:, :, :9] = self.field[:, :, 1:]
        field[:, :, 9] = np.repeat(self.oso_daily, self.n_persons, axis=0)
        field[:, :, 10] = np.repeat(self.oso_monthly, self.n_shifts, axis=1)
        return field

    def observation_spec(self):
        """Defines the observation provided by the environment.
        Each dict member will contain its expected structure and shape.
        Returns:
        A specification dict describing the observation fields and shapes.
        """
        return dict(
            info_state=tuple([self.n_persons * self.n_shifts * 11]),
            legal_actions=(10, ),
            current_player=(),
        )

    def action_spec(self):
        """Defines action specifications.
        Specifications include action boundaries and their data type.
        Returns:
        A specification dict containing action properties.
        """
        return dict(num_actions=self._num_actions, min=0, max=9, dtype=int)

    @property
    def num_players(self):
        return 1

    @property
    def is_turn_based(self):
        return False


class TicTacToeGame(pyspiel.Game):
  """A Python version of the Tic-Tac-Toe game."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return TicTacToeState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return BoardObserver(params)
    else:
      return IIGObserverForPublicInfoGame(iig_obs_type, params)


class TicTacToeState(pyspiel.State):
  """A python version of the Tic-Tac-Toe state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._cur_player = 0
    self._player0_score = 0.0
    self._is_terminal = False
    self.board = np.zeros((_N_PERSONS, _N_SHIFTS, 10))

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every perfect-information sequential-move game.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    return [a for a in range(_NUM_CELLS) if self.board[_coord(a)] == "."]

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    self.board[_coord(action)] = "x" if self._cur_player == 0 else "o"
    if _line_exists(self.board):
      self._is_terminal = True
      self._player0_score = 1.0 if self._cur_player == 0 else -1.0
    elif all(self.board.ravel() != "."):
      self._is_terminal = True
    else:
      self._cur_player = 1 - self._cur_player

  def _action_to_string(self, player, action):
    """Action -> string."""
    row, col = _coord(action)
    return "{}({},{})".format("x" if player == 0 else "o", row, col)

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return [self._player0_score, -self._player0_score]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return _board_to_string(self.board)


class BoardObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    # The observation should contain a 1-D tensor in `self.tensor` and a
    # dictionary of views onto the tensor, which may be of any shape.
    # Here the observation is indexed `(cell state, row, column)`.
    shape = (1 + _NUM_PLAYERS, _NUM_ROWS, _NUM_COLS)
    self.tensor = np.zeros(np.prod(shape), np.float32)
    self.dict = {"observation": np.reshape(self.tensor, shape)}

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    del player
    # We update the observation via the shaped tensor since indexing is more
    # convenient than with the 1-D tensor. Both are views onto the same memory.
    obs = self.dict["observation"]
    obs.fill(0)
    for row in range(_NUM_ROWS):
      for col in range(_NUM_COLS):
        cell_state = ".ox".index(state.board[row, col])
        obs[cell_state, row, col] = 1

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    return _board_to_string(state.board)


# Helper functions for game details.


def _line_value(line):
  """Checks a possible line, returning the winning symbol if any."""
  if all(line == "x") or all(line == "o"):
    return line[0]


def _line_exists(board):
  """Checks if a line exists, returns "x" or "o" if so, and None otherwise."""
  return (_line_value(board[0]) or _line_value(board[1]) or
          _line_value(board[2]) or _line_value(board[:, 0]) or
          _line_value(board[:, 1]) or _line_value(board[:, 2]) or
          _line_value(board.diagonal()) or
          _line_value(np.fliplr(board).diagonal()))


def _coord(move):
  """Returns (row, col) from an action id."""
  return (move // _NUM_COLS, move % _NUM_COLS)


def _board_to_string(board):
  """Returns a string representation of the board."""
  return "\n".join("".join(row) for row in board)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, TicTacToeGame)