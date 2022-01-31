import gym
from gym import spaces

import numpy as np

class ScheduleEnv(gym.Env):
    def __init__(self, ):
        super().__init__()
        self.action_space = spaces.Box(low=0.0, high=13.0, shape=(1,))
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(1,))

    def reset(self):
        # creating field and generating features based on it
        # we will be predicting only based on feature of the past
        # since we predict on features of the past it doesn't matter in which order we predict
        # it randomly selects whom to

    def step(self, action):
        pass

    def render(self):
        pass
