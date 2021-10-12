"""Catch reinforcement learning environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from open_spiel.python import rl_environment

# Actions
NOOP = 0
LEFT = 1
RIGHT = 2

_Point = collections.namedtuple("Point", ["x", "y"])


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
                # terminal_cond = (self.field[self.person, self.shift-1, 1:]>0).sum() == 1
                if terminal_cond:
                    print('SECOND SHIFT')

            # check 1001
            if (not terminal_cond) and self.shift >= 4:
                # terminal_cond = np.array_equal(self.field[self.person, self.shift-4:self.shift, 1:].sum(axis=1).ravel(),
                                            #    np.array([1, 0, 0, 1]))
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