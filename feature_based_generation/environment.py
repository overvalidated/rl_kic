
import gym
from gym import spaces

class ScheduleEnv(gym.Env):
    def __init__(self, ):
        super().__init__()
        self.action_space = spaces.Box(low=0.0, high=13.0, shape=(1,))
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(1,))

    def reset(self, ):
        pass

    def step(self, action):
        pass

    def render(self):
        pass


if __name__ == "__main__":
    # testing environment