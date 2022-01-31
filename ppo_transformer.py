import argparse

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.qmix import QMixTrainer
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.catalog import ModelCatalog

import time
import numpy as np
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

import seaborn as sns
import matplotlib.pyplot as plt

import env
import gym
from gym import spaces
import torch

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn


np.random.seed(42)
torch.manual_seed(42)

## TODO
## 1) Add useful reward
## 2) implement transformer - GTrXl (permutation-invariant network)
## 3) rework action taking process

N_PERSONS = 4
INTERMEDIATE_SIZE = 64

# wrapper for cython environment
class ScheduleGym(gym.Env):
    def __init__(self, env_config):
        super().__init__()

        n_persons = env_config['n_persons']
        n_shifts = env_config['n_shifts']

        self.training_iterations = 0
        self.action_space = spaces.MultiDiscrete([15, ] * n_persons)
        self.observation_space = spaces.Dict({
             #, shape=(n_persons, 10 + n_shifts), dtype=np.float64),
            "obs": spaces.Box(low=-100.0, high=100.0, shape=(n_persons * 10 + 42 * 2, ), dtype=np.float64),
            "action_mask": spaces.Box(low=0, high=1, shape=(15,), dtype=np.float64)
        })
        self.env_state = env.new_state(
            n_persons=n_persons,
            n_shifts=n_shifts
        )
        self.target_hours = np.random.choice([16], size=(n_shifts, ))
        self.target_hours[1::2] = 0
        self.target_hours[10::14] = 8
        self.target_hours[12::14] = 8
        self.env_state = env.prepare_env(**self.env_state)
        self.target_hours[:16] = np.sum(self.env_state['hours'][:, :16], axis=0)
        # self.target_hours[14:28] = self.target_hours[:14]
        # self.target_hours[28:] = self.target_hours[:14]

        self.acc_reward = 0
        self.recent_acc_rewards = [0]
            #  action=np.random.choice(np.arange(10)[env.get_possible_moves(**self.env_state)]))

    def reset(self):
        self.env_state = env.new_state(
            n_persons=self.env_state['n_persons'],
            n_shifts=self.env_state['n_shifts']
        )
        self.acc_reward = 0

        self.target_hours = np.random.choice([16], size=(42, ))
        self.target_hours[1::2] = 0
        self.target_hours[10::14] = 8
        self.target_hours[12::14] = 8
        self.env_state = env.prepare_env(**self.env_state)
        self.target_hours[:8] = np.sum(self.env_state['hours'][:, :8], axis=0)
        # self.target_hours[14:28] = self.target_hours[:14]
        # self.target_hours[28:] = self.target_hours[:14]
        observation = env.get_observation(**self.env_state).ravel()
        observation = observation[:-4].reshape(4, -1)
        observation = observation.ravel()
        upcoming_shifts = self.target_hours.reshape(-1, )-20
        upcoming_shifts[:self.env_state['shift']] = 0

        prev_shifts = self.target_hours.reshape(-1, )-20
        prev_shifts[self.env_state['shift']:] = 0
        observation = np.concatenate([observation, upcoming_shifts])
        observation = np.concatenate([observation, prev_shifts])
        # observation = np.concatenate([observation, np.repeat(self.target_hours.reshape(1, -1) - 20, 4, axis=0)], axis=1)
        # self.training_iterations += 1
        try:
            action_mask = np.array(env.get_possible_moves(**self.env_state))
        except:
            action_mask = np.zeros((15,))
        return {"obs": observation, "action_mask": action_mask}

    def step(self, action):
        done = False
        violation_done = 0
        reward = 0
        workers = 0 if self.env_state['shift'] % 2 == 0 else self.env_state['n_persons']
        for i in range(self.env_state['n_persons']):
            possible_moves = np.array(env.get_possible_moves(**self.env_state)).ravel()
            self.env_state, reward_ = env.next_state(**self.env_state, action=action[i])
            workers += 1 if action[i] != 0 else 0
            # if reward_ >= 0:
            #     reward_ = np.exp((0.23) * -reward_)
            # elif reward_ < 0:
            #     reward_ = np.exp((0.2148) * reward_)
            # reward *= reward_
            
            # violation_done += possible_moves[action[i]] == 0
            done = done or violation_done > 0 or env.is_terminal(**self.env_state)
        done = done or workers < 1
        hours = np.sum(np.array(self.env_state['hours'])[:, self.env_state['shift']-1])
        if violation_done == 0 and workers >= 1:
            a_coef = 0.1  # + (self.training_iterations*16 / 10e5)
            # if self.training_iterations < 100000/16:
            reward = 1
            reward *= np.exp(-a_coef * np.abs(hours - self.target_hours[self.env_state['shift'] - 1]))
            # if self.training_iterations > 200000:
            # for i in range(self.env_state['n_persons']):
            #     reward *= np.exp(-0.1 * np.abs(np.sum(self.env_state['hours'][i, self.env_state['shift']-14:self.env_state['shift']]) - 40))
            # else:
            #     reward = 9 * reward + 1
        else:
            reward = -0.5 * violation_done / (i+1) - 0.5 * (workers < 1) # if np.random.random() < 0.3 else 0
        observation = env.get_observation(**self.env_state).ravel()
        observation = observation[:-4].reshape(4, -1)
        observation = observation.ravel()

        upcoming_shifts = self.target_hours.reshape(-1, )-20
        upcoming_shifts[:self.env_state['shift']] = 0

        prev_shifts = self.target_hours.reshape(-1, )-20
        prev_shifts[self.env_state['shift']:] = 0
        observation = np.concatenate([observation, upcoming_shifts])
        observation = np.concatenate([observation, prev_shifts])
        # observation = np.concatenate([observation, np.repeat(self.target_hours.reshape(1, -1), 4, axis=0)], axis=1)
        self.training_iterations += 1
        try:
            action_mask = np.array(env.get_possible_moves(**self.env_state))
        except:
            action_mask = np.zeros((15,))
        # self.acc_reward += reward
        # model_rew = 0
        # if done:
        #     perc = np.percentile(self.recent_acc_rewards, 90)
        #     if self.acc_reward > perc:
        #         model_rew = 1
        #     elif self.acc_reward < perc:
        #         model_rew = -1
        #     else:
        #         model_rew = 2 * round(np.random.random()) - 1
        #     self.recent_acc_rewards += [self.acc_reward]
        #     self.recent_acc_rewards = self.recent_acc_rewards[-5000:]
        return {"obs": observation, "action_mask": action_mask}, reward, done, {}

    def render(self, mode='human'):
        string_state = env.stringify_state(**self.env_state)
        if mode == "human":
            print(self.target_hours)
            print(string_state)
        elif mode == "image":
            return string_state


class TransfModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.preprocessor = get_preprocessor(obs_space.original_space)(obs_space.original_space)

        # takes person OHE and hours for every person + target hours and creates
        # can't decide what is better: to pass target hours as another person or as part of every person hours
        class HiddenModel(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.hours_processing = nn.Sequential(
                    nn.Linear(in_features=52, out_features=INTERMEDIATE_SIZE),
                    nn.TransformerEncoder(nn.TransformerEncoderLayer(INTERMEDIATE_SIZE, 2, INTERMEDIATE_SIZE, batch_first=True), 1),
                )

            def forward(self, x):
                return self.hours_processing(x)

        self._hidden_layers = HiddenModel(obs_space.original_space["obs"].shape[0])
        self._logits = nn.Sequential(
            nn.Linear(INTERMEDIATE_SIZE, 15),
        )
        self._value_branch = nn.Sequential(nn.Linear(INTERMEDIATE_SIZE, 1))

    def forward(self, input_dict, state, seq_lens):
        out = self._hidden_layers(input_dict["obs"]["obs"])
        # out = torch.sum(out * input_dict["person"].view(-1, 4, 1), dim=1)
        self._value_out = self._value_branch(torch.sum(out, dim=1).view(-1, INTERMEDIATE_SIZE))
        # self._value_out = self._value_branch(out)
        return self._logits(out).view(-1, 60), state

    def value_function(self):
        return self._value_out[0].view(-1, )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--training-iteration", default=10000, type=int)
    parser.add_argument("--ray-num-cpus", default=1, type=int)
    args = parser.parse_args()
    ray.init(num_cpus=args.ray_num_cpus)

    # register_env('schedule_env')

    ModelCatalog.register_custom_model("transformer_model", FullyConnectedNetwork)

    config={
        "env": ScheduleGym,
        "framework": "tf",
        "num_workers": args.num_workers,
        "env_config": {"n_persons": 4, "n_shifts": 42},
        "lr": 1e-4,
        'train_batch_size': 128,
        # "sgd_minibatch_size": 64,
        "entropy_coeff": 0.005,
        "vf_loss_coeff": 0.5,
        "gamma": 1.0,
        'rollout_fragment_length': 128,
        # "batch_size": 32,
        "model": {
            "custom_model": "transformer_model",
            "fcnet_hiddens": [256, 128, 128]
        },
    }
    config["exploration_config"] = {
        "type": "Curiosity",  # <- Use the Curiosity module for exploring.
        "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
        "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
        "feature_dim": 288,  # Dimensionality of the generated feature vectors.
        # Setup of the feature net (used to encode observations into feature (latent) vectors).
        "feature_net_config": {
            "fcnet_hiddens": [],
            "fcnet_activation": "relu",
        },
        "inverse_net_hiddens": [256, ],  # Hidden layers of the "inverse" model.
        "inverse_net_activation": "relu",  # Activation of the "inverse" model.
        "forward_net_hiddens": [256, ],  # Hidden layers of the "forward" model.
        "forward_net_activation": "relu",  # Activation of the "forward" model.
        "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
        # Specify, which exploration sub-type to use (usually, the algo's "default"
        # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
        "sub_exploration": {
            "type": "StochasticSampling",
        }
    }

    analysis = tune.run(
        "PPO",
        stop={"training_iteration": args.training_iteration},
        max_failures=0,
        config=config,
        checkpoint_at_end=True,
        checkpoint_freq=10,
    )

    last_checkpoint = analysis.get_last_checkpoint()
    # config={
    #     "env": ScheduleGym,
    #     "framework": "torch",
    #     "num_workers": args.num_workers,
    #     "env_config": {"n_persons": 4, "n_shifts": 42},
    #     "lr": 1e-4,
    #     "optimizer": "adam",
    #     'train_batch_size': 2048,
    #     # "sgd_minibatch_size": 64,
    #     "entropy_coeff": 0.02,
    #     "vf_loss_coeff": 0.5,
    #     'rollout_fragment_length': 128,
    #     # "batch_size": 32,
    #     "model": {
    #         "custom_model": "transformer_model",
    #     },
    # }
    agent = PPOTrainer(config=config, env=ScheduleGym)
    agent.restore(last_checkpoint)
    agent.evaluate()

    env_ = ScheduleGym({'n_persons': 4, "n_shifts": 42})

    # run until episode ends
    episode_reward = 0
    done = False
    obs = env_.reset()
    while not done:
        action = agent.compute_single_action(obs, )
        print(action)
        obs, reward, done, info = env_.step(action)
        episode_reward += reward
        env_.render()

    field = env_.env_state['hours']

    field = np.concatenate([
        field, 
        np.sum(field, axis=0).reshape(1, -1), 
        env_.target_hours.reshape(1, -1)
    ], axis=0)
    plt.figure(figsize=(30, 5))
    sns.heatmap(field, annot=True)
    plt.show()
