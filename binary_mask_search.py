from copy import copy
import numpy as np
import torch

from stable_baselines3 import PPO, TD3, DDPG, SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from schedule_class import TemplateMaskController

import gym
from gym import spaces

np.random.seed(42)
torch.manual_seed(42)

"""
    фичи:
        1) можно ли вывести сотрудника
        2) сколько дней уже работал
        3) 
        4) 
"""

class TemplatingEnv(gym.Env):
    def __init__(self, n_persons=4, n_shifts=28):
        super().__init__()
        self.action_space = spaces.MultiBinary(n_persons)
        self.observation_space = spaces.Box(low=0, high=12.0, 
        shape=(148, ))
        self.hours = np.random.randint(7, 13, size=(n_persons))
        self.n_persons = n_persons
        self.n_shifts = n_shifts
        self.map_template = self.create_solution()
        self.target_hours = self.map_template.sum(axis=0).ravel()
        self.template = TemplateMaskController(
            n_persons=self.n_persons,
            n_shifts=self.n_shifts,
            const_hours=self.hours
        )
        
    def create_solution(self, ):
        map_template = np.zeros((self.n_persons, self.n_shifts))
        two_two = np.array([1, 1, 0, 0])
        five_two = np.array([1, 1, 1, 1, 1, 0, 0])
        four_three = np.array([1, 1, 1, 1, 0, 0, 0])
        three_three = np.array([1, 1, 1, 0, 0, 0])
        for person_idx in range(self.n_persons):
            p = np.array([10 <= self.hours[person_idx] <= 12,
                 self.hours[person_idx] == 10,
                 7 <= self.hours[person_idx] <= 10,
                 7 <= self.hours[person_idx] <= 9])
            choice = np.random.choice([two_two, three_three, four_three, five_two],
                        p=p/p.sum())
            row = np.repeat(choice.reshape(1, -1),
                            self.n_shifts // choice.size + 1,
                            axis=0).ravel()[:self.n_shifts]
            
            map_template[person_idx] = row * self.hours[person_idx]
        return map_template

    def create_obs(self, done=False):
        obs = self.template.hours.ravel()
        if done:
            return np.concatenate([
                obs, 
                np.ones((self.n_persons, )), 
                self.target_hours,
                self.hours
            ])
        else:
            return np.concatenate([obs, 
                [1-int(copy(self.template).add_shift(np.eye(self.n_persons)[i])) for i in range(self.n_persons)],
                self.target_hours,
                self.hours    
            ])

    def reset(self):
        self.hours = np.random.randint(7, 13, size=(self.n_persons))
        self.map_template = self.create_solution()
        self.target_hours = self.map_template.sum(axis=0).ravel()
        self.template = TemplateMaskController(
            n_persons=self.n_persons,
            n_shifts=self.n_shifts,
            const_hours=self.hours
        )
        return self.create_obs(done=False)

    def step(self, action):
        done = self.template.add_shift(action)
        target_ = self.target_hours[self.template.current_position-1]
        got_ = np.sum(action * self.hours)
        return self.create_obs(done=done), np.exp(-0.4 * np.abs(got_ - target_)), not done, {}

if __name__ == "__main__":
    base_env_ = TemplatingEnv()
    base_env_2 = copy(base_env_)

    model_kwargs = dict(
        tensorboard_log='tensorboard_logs/model_dqn',
        # policy_kwargs=policy_kwargs,
        seed=42,
        n_steps=256,
        ent_coef=0.5, # requires tuning
        vf_coef=0.5,
        gamma=0.3,
        clip_range=0.1,
        batch_size=256,
        learning_rate=0.0001,
        verbose=2
    )

    policy_kwargs = dict(
        activation_fn=lambda: torch.nn.GELU(),
        net_arch=[256, 256, dict(pi=[128, 128], vf=[128, 128])]
        # features_extractor_class=TransformerExtractor,
        # features_extractor_kwargs=dict(features_dim=128)
    )

    env_ = make_vec_env(lambda: base_env_, 1)
    model = PPO("MlpPolicy", env_, policy_kwargs=policy_kwargs, **model_kwargs)
    # model = PPO("MlpPolicy", env_, policy_kwargs, **model_kwargs)

    try:
        model.learn(total_timesteps=3000000)
    except KeyboardInterrupt:
        pass

    # model.save(f"schedule_generator_{N_PERSONS}")
    # env_ = TemplatingEnv()
    obs = base_env_2.reset()
    # print(model.predict(obs,deterministic=True))
    # print(env_.env_state['shift'])
    # print(obs.flatten())
    # model = pret.policy
    print('solution is', )
    print('\n'.join([' '.join(map(lambda x: '1' if x > 0 else '0', i)) for i in base_env_2.map_template.astype(np.int32)]))

    while True:
        action, _states = model.predict(obs, deterministic=True)
        # print(model.policy(torch.Tensor(obs)))
        obs, reward, done, info = base_env_2.step(action)
        # action, _states = model.predict(obs, deterministic=True)
        print('reward is', reward)
        print('\n'.join([' '.join(map(str, i)) for i in base_env_2.template.hours.astype(np.int32)]))
        print()
        if done:
            break
