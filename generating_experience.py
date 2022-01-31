
import numpy as np
import argparse

import torch
import env
import gym
from gym import spaces
import pickle as pkl

from imitation.data.types import Trajectory
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)

# two ways to tackle problems with training for more than one person
# 1) add reward predicting for 
# 2) transfer weights from one-person model with slicing over incompatible tensors.

# target 

N_PERSONS = 8
INTERMEDIATE_SIZE = 128

# wrapper for cython environment
class ScheduleGym(gym.Env):
    def __init__(self, n_persons=N_PERSONS, n_shifts=42):
        super().__init__()
        self.training_iterations = 0
        self.action_space = spaces.Box(low=0.0, high=14, shape=(n_persons, ))
        # self.action_space = spaces.MultiDiscrete([15, ] * n_persons)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n_persons * 10 + n_shifts, ))
        self.env_state = env.new_state(
            n_persons=n_persons,
            n_shifts=n_shifts
        )

    def reset(self):
        self.env_state = env.new_state(
            n_persons=self.env_state['n_persons'],
            n_shifts=self.env_state['n_shifts']
        )
        hours = [0, 4, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13]

        self.target_hours = np.random.choice([56, 58, 60, 62, 64], size=(42, ))
        self.target_hours[1::2] = 0
        # self.target_hours[10::14] = self.target_hours[10::14] // 2
        # self.target_hours[12::14] = self.target_hours[12::14] // 2

        pre_shifts = 16
        # pre_shifts = np.random.randint(16, 20)
        # if pre_shifts % 2 == 1:
        #     pre_shifts += np.random.random() > 0.2


        self.env_state = env.prepare_env(**self.env_state, shift_prep=42)
        # self.target_hours[:16] = np.sum(self.env_state['hours'][:, :16], axis=0)
        self.target_hours = np.sum(self.env_state['hours'], axis=0)

        list_observation = []
        list_actions = []

        upcoming_shifts = np.array(self.target_hours.ravel())
        observation = np.array(env.get_observation(**self.env_state, shift_pos=pre_shifts))[:-self.env_state['n_persons']]
        upcoming_shifts[:-pre_shifts] = upcoming_shifts[pre_shifts:]
        upcoming_shifts[-pre_shifts:] = 0
        observation = np.concatenate([observation, upcoming_shifts, 
            np.sum(env.get_observation(**self.env_state, shift_pos=self.env_state['shift'])[:-N_PERSONS].reshape(N_PERSONS, -1), axis=1)])
        list_observation += [observation]

        for _ in range(42 - pre_shifts):
            actions = [hours.index(hour) for hour in np.array(self.env_state['hours'][:, pre_shifts])]
            list_actions += [actions]

            pre_shifts += 1

            upcoming_shifts = np.array(self.target_hours.ravel())
            observation = np.array(env.get_observation(**self.env_state, shift_pos=pre_shifts))[:-self.env_state['n_persons']]
            upcoming_shifts[:-pre_shifts] = upcoming_shifts[pre_shifts:]
            upcoming_shifts[-pre_shifts:] = 0
            observation = np.concatenate([observation, upcoming_shifts, 
                np.sum(env.get_observation(**self.env_state, shift_pos=self.env_state['shift'])[:-N_PERSONS].reshape(N_PERSONS, -1), axis=1)])
            list_observation += [observation]

        return (
            np.array(list_observation),
            np.array(list_actions)
        )

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--number", default=1000000, type=int)
    return argparser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    env_ = ScheduleGym()
    list_tr = []
    
    for i in tqdm(range(args.number)):
        res = env_.reset()
        list_tr += [Trajectory(res[0], res[1], None, terminal=True)]
        
    with open('traces.pkl', 'wb') as f:
        pkl.dump(list_tr, f)