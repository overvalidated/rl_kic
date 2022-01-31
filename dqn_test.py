import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from open_spiel.python.algorithms.dqn import DQN
from environment import Env
import random
import numpy as np

from open_spiel.python.rl_environment import TimeStep

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

torch.manual_seed(42)

if __name__ == "__main__":
    dqn = DQN()
    for play_ in range(100000):
        state_ = Env.initial_state()
        while Env.is_done_state(state_, 0):
            state_dqn = TimeStep(observations=[Env.get_obs_for_states([])], rewards=[0], )
    dqn.step()
