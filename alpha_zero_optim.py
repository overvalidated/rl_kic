import argparse

import ray
from ray import tune
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.tune.registry import register_env
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel, convert_to_tensor
# from ray.rllib.contrib.alpha_zero.environments.cartpole import CartPole
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.catalog import ModelCatalog

from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.evaluation.episode import MultiAgentEpisode

import time
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import env
import gym
from gym import spaces
import torch

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn

from copy import deepcopy, copy

from ppo_transformer import INTERMEDIATE_SIZE

np.random.seed(42)
torch.manual_seed(42)

## TODO
## 1) Add useful reward
## 2) implement transformer - GTrXl (permutation-invariant network)
## 3) rework action taking process

N_PERSONS = 4
INTERMEDIATE_SIZE = 256

# wrapper for cython environment
class ScheduleGym(gym.Env):
    def __init__(self, env_config):
        super().__init__()
        n_persons=4; n_shifts=42
        self.action_space = spaces.Discrete(15, )
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=-100, high=100, shape=(n_persons, 52), dtype=np.float64),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.action_space.n, ), dtype=np.float32),
            "person": spaces.Box(low=0, high=1, shape=(n_persons, ), dtype=np.float32)
        })
        self.env_state = env.new_state(
            n_persons=n_persons,
            n_shifts=n_shifts
        )
        self.target_hours = np.random.choice([18, 19, 20, 21, 22], size=(n_shifts, ))
        self.target_hours[1::2] = 0
        # self.target_hours[10::14] = 0
        # self.target_hours[12::14] = 0
        self.env_state = env.prepare_env(**self.env_state)
        self.target_hours[:8] = np.sum(self.env_state['hours'][:, :8], axis=0)
        # self.target_hours[14:28] = self.target_hours[:14]
        self.target_hours[28:] = self.target_hours[14:28]
        self.running_reward = 0
            #  action=np.random.choice(np.arange(10)[env.get_possible_moves(**self.env_state)]))

    def reset(self):
        self.env_state = env.new_state(
            n_persons=self.env_state['n_persons'],
            n_shifts=self.env_state['n_shifts']
        )
        self.running_reward = 0
        self.target_hours = np.random.choice([13, 14, 15, 16, 17, 18, 19, 20, 21, 22], size=(42, ))
        self.target_hours[1::2] = 0
        # self.target_hours[10::14] = 0
        # self.target_hours[12::14] = 0
        self.env_state = env.prepare_env(**self.env_state)
        self.target_hours[:8] = np.sum(self.env_state['hours'][:, :8], axis=0)
        self.target_hours[14:28] = self.target_hours[:14]
        self.target_hours[28:] = self.target_hours[:14]
        observation = env.get_observation(**self.env_state).ravel()
        human = observation[-4:]
        observation = observation[:-4].reshape(4, -1)
        observation = np.concatenate([observation, np.repeat(self.target_hours.reshape(1, -1) - 20, 4, axis=0)], axis=1)
        try:
            action_mask = np.array(env.get_possible_moves(**self.env_state))
        except:
            action_mask = np.zeros((15,))
        return {"obs": observation, "action_mask": action_mask, "person": human}

    def set_state(self, state):
        self.env_state = state[0]
        self.target_hours = state[1]
        self.running_reward = state[2]
        observation = env.get_observation(**self.env_state).ravel()
        human = observation[-4:]
        observation = observation[:-4].reshape(4, -1)
        observation = np.concatenate([observation, np.repeat(self.target_hours.reshape(1, -1) - 20, 4, axis=0)], axis=1)
        try:
            action_mask = np.array(env.get_possible_moves(**self.env_state))
        except:
            action_mask = np.zeros((15,))
        return {"obs": observation, "action_mask": action_mask, "person": human}

    def copy_(self, field, hours, possible_moves, worked_days, 
        person, shift, n_persons, n_shifts):
            return dict(field=np.array(field),
            hours=np.array(hours),
            possible_moves=np.array(possible_moves),
            worked_days=np.array(worked_days),
            person=person,
            shift=shift,
            n_persons=n_persons,
            n_shifts=n_shifts)

    def get_state(self):
        return self.copy_(**self.env_state), copy(self.target_hours), self.running_reward

    def step(self, action):
        done = False
        violation_done = 0
        possible_moves = np.array(env.get_possible_moves(**self.env_state)).ravel()
        self.env_state, reward_ = env.next_state(**self.env_state, action=action)
        self.running_reward += reward_
        done = done or env.is_terminal(**self.env_state) or possible_moves[action] == 0
        observation = env.get_observation(**self.env_state).ravel()
        human = observation[-4:]
        observation = observation[:-4].reshape(4, -1)
        observation = np.concatenate([observation, np.repeat(self.target_hours.reshape(1, -1) - 20, 4, axis=0)], axis=1)
        if self.env_state['shift'] % 2 == 1:
            done = done or (np.array(self.env_state['hours'])[:, self.env_state['shift']-1] > 0).sum() < 1
        try:
            action_mask = np.array(env.get_possible_moves(**self.env_state)).astype(np.float32)
        except:
            action_mask = np.zeros((15, ))
            done = True
        score = self.running_reward if not done else 0
        return {'obs': observation.astype(np.float32), 'action_mask': action_mask, 'person': human}, score, done, {}

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
                    nn.ReLU(),
                    nn.TransformerEncoder(nn.TransformerEncoderLayer(INTERMEDIATE_SIZE, 4, INTERMEDIATE_SIZE, batch_first=True), 1),
                )

            def forward(self, x):
                return self.hours_processing(x)

        self._hidden_layers = HiddenModel(obs_space.original_space["obs"].shape[0])
        self._logits = nn.Sequential(
            nn.Linear(INTERMEDIATE_SIZE, 15),
            
        )
        self._value_branch = nn.Sequential(nn.Linear(INTERMEDIATE_SIZE, 1))

    def forward(self, input_dict, state, seq_lens):
        out = self._hidden_layers(input_dict["obs"])
        out = torch.sum(out * input_dict["person"].view(-1, 4, 1), dim=1)
        # self._value_out = self._value_branch(torch.sum(out, dim=1).view(-1, 256))
        self._value_out = self._value_branch(out)
        return self._logits(out).view(-1, 15), state

    def value_function(self):
        return self._value_out[0].view(-1, )

    def compute_priors_and_value(self, obs):
        obs = convert_to_tensor([self.preprocessor.transform(obs)])
        input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

        with torch.no_grad():
            model_out = self.forward(input_dict, None, [1])
            logits, _ = model_out
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()

            return priors, value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=7, type=int)
    parser.add_argument("--training-iteration", default=100000, type=int)
    parser.add_argument("--ray-num-cpus", default=8, type=int)
    args = parser.parse_args()
    ray.init(num_cpus=args.ray_num_cpus)

    # register_env('schedule_env')

    ModelCatalog.register_custom_model("transformer_model", TransfModel)

    config={
            "env": ScheduleGym,
            "framework": "torch",
            "num_workers": args.num_workers,
            "env_config": {"n_persons": 4, "n_shifts": 42},
            "rollout_fragment_length": 50,
            "train_batch_size": 1024,
            "sgd_minibatch_size": 64,
            "lr": 1e-4,
            "num_sgd_iter": 1,
            "mcts_config": {
                "puct_coefficient": 1.5,
                "num_simulations": 300,
                "temperature": 1.0,
                "dirichlet_epsilon": 0.20,
                "dirichlet_noise": 0.03,
                "argmax_tree_policy": False,
                "add_dirichlet_noise": True,
            },
            "ranked_rewards": {
                "enable": True,
            },
            "model": {
                "custom_model": "transformer_model",
            },
        }

    analysis = tune.run(
        "contrib/AlphaZero",
        stop={"training_iteration": args.training_iteration},
        max_failures=0,
        config=config,
        checkpoint_at_end=True,
        checkpoint_freq=100
    )

    last_checkpoint = analysis.get_last_checkpoint()
    agent = AlphaZeroTrainer(config=config, env=ScheduleGym)
    agent.restore(last_checkpoint)
    
    policy = agent.get_policy(DEFAULT_POLICY_ID)

    episode = MultiAgentEpisode(
        PolicyMap(0,0),
        lambda _, __: DEFAULT_POLICY_ID,
        lambda: None,
        lambda _: None,
        0
    )

    env_ = ScheduleGym({'n_persons': 4, "n_shifts": 42})
    episode.user_data['initial_state'] = env_.get_state()
    # run until episode ends
    episode_reward = 0
    
    done = False
    obs = env_.reset()
    while not done:
        action, _, _ = policy.compute_single_action(obs, episode=episode)
        print(action)
        obs, reward, done, info = env_.step(action)
        episode_reward += reward
        env_.render()
        episode.length += 1

    ray.shutdown()
