
# from copy import deepcopy as copy
import sys

import numpy as np
from copy import copy as real_copy#, deepcopy
import env
# import time

def copy(schedule):
    return dict(
        field=real_copy(np.array(schedule['field'])),
        hours=real_copy(np.array(schedule['hours'])),
        possible_moves=real_copy(np.array(schedule['possible_moves'])),
        worked_days=real_copy(np.array(schedule['worked_days'])),
        person=schedule['person'],
        shift=schedule['shift'],
        n_persons=schedule['n_persons'],
        n_shifts=schedule['n_shifts']
    )

class Env:
    n_actions = 10
    n_persons = 4
    n_shifts = 62
    daily_oso = np.random.random(size=(1,n_shifts)) * 300 * 6.2 / 60
    initial_state_ = None

    @staticmethod
    def initial_state():
        # if CythonScheduleStaticEnvironment.initial_state
        if Env.initial_state_ == None:
            return env.new_state(Env.n_persons, Env.n_shifts)
        else:
            return Env.initial_state_

    @staticmethod
    def is_done_state(state_, state_idx):
        # state = copy(state_)
        # hours_array = np.array([[0] + list(range(4, 13))]).reshape(-1, 1)
        return env.is_terminal(**state_) #check_terminal(state)# or term_cond

    @staticmethod
    def is_done_state_strict(state_):
        return env.is_terminal_strict(**state_)

    @staticmethod
    def next_state(state_, action):
        state = copy(state_)
        # # print(action)
        # if state['shift'] < Env.n_shifts:
        #     state['field'][state['person'], state['shift'], action] = 1
        #     state['person'] += 1
        #     state['shift'] += state['person'] // Env.n_persons
        #     state['person'] %= Env.n_persons
        return env.next_state(**state, action=action)

    #reward for
    @staticmethod
    def get_return(state_, state_idx):
        # state = copy(state_)
        # if state_idx == 60:
        #     str_state = '\n'.join([''.join(map(str, i)) for i in state['field'][:, :, 1:].sum(axis=2).tolist()])
        #     print(str_state)
        #     sys.exit()
        # return state_idx / (4 * 62) #state['field'][:, :, :].sum() / 100 - 1
        return env.get_return(**state_)

    @staticmethod
    def get_obs_for_states(states_):
        obs_ = []
        for state in states_:
            obs_ += [env.get_observation(**state).reshape(1, -1)]
        return np.concatenate(obs_, axis=0)
