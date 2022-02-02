# wrapper for cython environment
import gym
from gym import spaces
import env
import numpy as np
from copy import copy

N_PERSONS = 16


class ScheduleGym(gym.Env):
    def __init__(self, n_persons=N_PERSONS, n_shifts=42):
        super().__init__()
        self.training_iterations = 0
        self.action_space = spaces.Box(low=0.0, high=14, shape=(n_persons,))
        # self.action_space = spaces.MultiDiscrete([15, ] * n_persons)
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(n_persons * n_shifts,))
        self.env_state = env.new_state(
            n_persons=n_persons,
            n_shifts=n_shifts
        )
        self.target_hours = 4 * 8 + np.random.choice([-1, 0, 1], size=(
            self.env_state['n_shifts'],))  # * np.random.choice([20,21,22,23,24,25], size=(self.env_state['n_shifts'], ))
        self.target_hours[1::2] = 0
        # self.target_hours[10::14] = self.target_hours[10::14] // 2
        # self.target_hours[12::14] = self.target_hours[12::14] // 2
        self.env_state = env.prepare_env(**self.env_state, shift_prep=16)
        self.orig_state = copy(self.env_state)
        self.target_hours[:16] = np.sum(
            self.env_state['hours'][:, :16], axis=0)
        self.target_hours[16:32] = self.target_hours[:16]
        self.target_hours[32:42] = self.target_hours[:10]
        self.acc_reward = 0
        self.recent_acc_rewards = [0]
        #  action=np.random.choice(np.arange(10)[env.get_possible_moves(**self.env_state)]))

    def reset(self):
        self.env_state = env.new_state(
            n_persons=self.env_state['n_persons'],
            n_shifts=self.env_state['n_shifts']
        )
        self.acc_reward = 0
        self.target_hours = 4 * 8 + np.random.choice([-1, 0, 1], size=(
            self.env_state['n_shifts'],))  # * np.random.choice([20,21,22,23,24,25], size=(self.env_state['n_shifts'], ))
        self.target_hours[1::2] = 0
        # self.target_hours[10::14] = self.target_hours[10::14] // 2
        # self.target_hours[12::14] = self.target_hours[12::14] // 2
        self.env_state = env.prepare_env(**self.env_state, shift_prep=16)
        self.target_hours[:16] = np.sum(
            self.env_state['hours'][:, :16], axis=0)
        self.target_hours[16:32] = self.target_hours[:16]
        self.target_hours[32:42] = self.target_hours[:10]
        observation = np.array(env.get_observation(**self.env_state, shift_pos=self.env_state['shift']))[
            :-self.env_state['n_persons']]
        upcoming_shifts = np.array(self.target_hours.reshape(-1, ))
        upcoming_shifts[:-self.env_state['shift']
                        ] = upcoming_shifts[self.env_state['shift']:]
        upcoming_shifts[-self.env_state['shift']:] = 0

        observation = np.concatenate([
            observation,
            upcoming_shifts,
            np.sum(
                env.get_observation(**self.env_state, shift_pos=self.env_state['shift'])[:-N_PERSONS].reshape(N_PERSONS,
                                                                                                              -1),
                axis=1)])

        # print(observation.shape)
        # observation = np.concatenate([observation, prev_shifts])
        # observation = np.concatenate([observation, np.array([self.target_hours[self.env_state['shift']]])])
        self.training_iterations += 1
        return np.array(self.env_state['hours']).ravel()

    def step(self, action):
        done = False
        violation_done = 0
        reward = 0

        # hours = [0, 4, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13]
        # def get_rounding(num):
        #     if 0.25 <= num - int(num) < 0.75:
        #         return float(int(num)) + 0.5
        #     else:
        #         return float(round(num))

        workers = 0 if self.env_state['shift'] % 2 == 0 else self.env_state['n_persons']
        if self.target_hours[self.env_state['shift']] == 0:
            workers += self.env_state['n_persons']
        # for i in range(len(action)):
        #     if action[i] < 1.0:
        #         action[i] = 0
        #     elif 1.0 <= action[i] < 3.75 or 4.25 <= action[i] < 6.75:
        #         action[i] = 0
        #         # violation_done += 1
        #     else:
        #         action[i] = get_rounding(action[i])
        #     action[i] = int(hours.index(action[i]))
        # action = action.astype(np.int32)

        for i in range(self.env_state['n_persons']):
            possible_moves = np.array(
                env.get_possible_moves(**self.env_state)).ravel()
            if possible_moves[int(action[i])] == 0:
                done = True
            self.env_state, reward_ = env.next_state(
                **self.env_state, action=action[i])
            workers += 1 if int(action[i]) != 0 else 0
            # if reward_ >= 0:
            #     reward_ = np.exp(min(1.23, 0.23 + (self.training_iterations * 8 / (10e6))) * -reward_)
            # elif reward_ < 0:
            #     reward_ = np.exp(min(0.6148, 0.2148 + 0.4 * (self.training_iterations * 8 / (10e6))) * reward_)
            # reward *= reward_

            violation_done += possible_moves[int(action[i])] == 0
            done = done or env.is_terminal(**self.env_state)

        done = done or workers < 1
        hours = np.sum(np.array(self.env_state['hours'])[
                       :, self.env_state['shift'] - 1])

        if workers >= 1:
            a_coef = 0.2  # min(0.3 + 10e6)), 1.3)
            reward = 1 if self.target_hours[self.env_state['shift'] - 1] > 0 else 0.1
            # if self.target_hours[self.env_state['shift'] - 1] > 0 and abs(hours / self.target_hours[self.env_state['shift'] - 1] - 1) <= 0.1:
            #     reward = 1
            # else:

            reward *= np.exp(-a_coef * np.abs(hours -
                             self.target_hours[self.env_state['shift'] - 1]))
            for i in range(self.env_state['n_persons']):
                reward *= np.exp(-0.03 * max(
                    np.sum(self.env_state['hours'][i, self.env_state['shift'] - 15:self.env_state['shift']]) - 44, 0))
            # reward *= 0.7 ** violation_done
            # real_dist = np.array(self.env_state['hours']).sum(axis=1).ravel() / np.array(self.env_state['hours']).sum()
            # awaited = np.ones((4, )) / 4
            # reward *= max(1-kl_div(real_dist, awaited).sum(), 1e-2)
        else:
            reward = -min(violation_done, 1) / (i + 1) - 0.5 * \
                (workers < 1)  # if np.random.random() < 0.3 else 0
            done = True
            # reward =
        # 3 дневных смены - не может быть ночи
        # reward = reward if self.env_state['shift'] % 2 == 1 else min(reward, 0)

        observation = np.array(env.get_observation(**self.env_state, shift_pos=self.env_state['shift']))[
            :-self.env_state['n_persons']]
        upcoming_shifts = np.array(self.target_hours.reshape(-1, ))
        upcoming_shifts[:-self.env_state['shift']
                        ] = upcoming_shifts[self.env_state['shift']:]
        upcoming_shifts[-self.env_state['shift']:] = 0

        observation = np.concatenate([
            observation,
            upcoming_shifts,
            np.sum(
                env.get_observation(**self.env_state,
                                    shift_pos=self.env_state['shift'])[:-N_PERSONS].reshape(N_PERSONS,
                                                                                            -1),
                axis=1)])
        # observation = np.concatenate([observation, prev_shifts])
        # observation = np.concatenate([observation, np.array([min(41,self.target_hours[self.env_state['shift']]])])
        self.training_iterations += 1
        # if self.training_iterations * 8 > 200000:
        #     print(max(1-sum(kl_div(awaited, real_dist))*5, 1e-2))
        if done:
            reward = min(reward, 0)
        return np.array(self.env_state['hours']).ravel(), reward, done, {}

    def render(self, mode='human'):
        string_state = env.stringify_state(**self.env_state)
        if mode == "human":
            print(self.target_hours)
            print(string_state)
        elif mode == "image":
            return string_state
