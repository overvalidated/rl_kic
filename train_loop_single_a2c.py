#%%
import pathlib
import pickle
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import torchga
from schedule_env import ScheduleGym
from imitation.algorithms.bc import BC
from imitation.data import rollout

np.random.seed(42)
torch.manual_seed(42)

N_PERSONS = 8

def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function, env_

    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                       weights_vector=solution)

    # Use the current solution as the model parameters.
    model.load_state_dict(model_weights_dict)

    total_reward = 0
    # Playing 100 games
    with torch.no_grad():
        for i in range(1):
            obs = env_.reset()
            done = False
            while not done:
                action, _ = model(torch.Tensor(obs).view(1, -1), torch.Tensor(env_.env_state['hours']))
                # print(action.numpy().reshape(-1, ))
                obs, reward, done, info = env_.step(np.clip(action.numpy().reshape(-1, ), 0, 14.49))
                # obs, reward, done, _ = env_.step(action[0].argmax(axis=1))
                total_reward += reward
        # print(total_reward / 30)
    return total_reward


def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


# if __name__ == "__main__":
#     model = TransformerExtractor(features_dim=64).eval()
#     env_ = ScheduleGym()

#     torch_ga = torchga.TorchGA(model=model,
#                             num_solutions=200)

#     # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
#     num_generations = 100 # Number of generations
#     num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.
#     initial_population = torch_ga.population_weights # Initial population of network weights
#     ga_instance = pygad.GA(num_generations=num_generations,
#                         num_parents_mating=num_parents_mating,
#                         initial_population=initial_population,
#                         fitness_func=fitness_func,
#                         on_generation=callback_generation)

#     ga_instance.run()

#     solution, solution_fitness, solution_idx = ga_instance.best_solution()
#     best_solution_weights = torchga.model_weights_as_dict(model=model,
#                                                         weights_vector=solution)
#     model.load_state_dict(best_solution_weights)
#     # torch.save(model.state_dict, f"model_state_dict_{num_generations}_{solution_fitness}")

#     env_ = ScheduleGym()
#     obs = env_.reset()
#     print(env_.env_state['shift'])
#     print(obs.flatten())
#     # model = pret.policy
#     with torch.no_grad():
#         while True:
#             action, _states = model(torch.Tensor(obs).view(1, -1), torch.Tensor(env_.env_state['hours']))
#             # print(model.policy(torch.Tensor(obs)))
#             obs, reward, done, info = env_.step(np.clip(action.numpy().reshape(-1, ), 0, 14.49))
#             # obs, reward, done, _ = env_.step(action[0].argmax(axis=1))
#             print('reward is', reward)
#             # env_.render()
#             if done:
#                 break
#         field = env_.env_state['hours']

#         field = np.concatenate([
#             field, 
#             np.sum(field, axis=0).reshape(1, -1), 
#             env_.target_hours.reshape(1, -1)
#             ], axis=0)
#         plt.figure(figsize=(30, 5))
#         sns.heatmap(field, annot=True)
#         plt.savefig('big_schedule_2.png')
#         plt.show()

#%%
if __name__ == "__main__":
    model_kwargs = dict(
        tensorboard_log='tensorboard_logs/model_dqn',
        # policy_kwargs=policy_kwargs,
        seed=42,
        n_steps=128,
        ent_coef=0.5, # requires tuning
        vf_coef=0.5,
        gamma=0.0,
        clip_range=0.1,
        batch_size=128,
        learning_rate=0.0001,
        verbose=2
    )

    policy_kwargs = dict(
        activation_fn=lambda: torch.nn.ReLU(),
        net_arch=[512, 256, dict(pi=[256, 128], vf=[256, 128])]
        # features_extractor_class=TransformerExtractor,
        # features_extractor_kwargs=dict(features_dim=128)
    )
#%%
    with open('traces.pkl', 'rb') as f:
        expert_data = pickle.load(f)
    tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    tempdir_path = pathlib.Path(tempdir.name)
    # gail_logger = logger.configure(tempdir_path / "GAIL/")
    env_ = make_vec_env(ScheduleGym, 1)
    model = PPO("MlpPolicy", env_, policy_kwargs=policy_kwargs, **model_kwargs)
#%%
    transitions = rollout.flatten_trajectories(expert_data)
    pret = BC(
        action_space=ScheduleGym().action_space,
        observation_space=ScheduleGym().observation_space,
        # venv=env_,
        demonstrations=transitions,
        # demo_batch_size=256,
        batch_size=1024, 
        # optimizer_cls=SGD,
        # optimizer_kwargs=dict(lr=0.0001),
        policy=model.policy)
        # gen_algo=model)
    # pret.train(n_batches=2000)
    # if N_PERSONS > 1:
    #     model_single = PPO.load('schedule_generator_2')
    #     transfer_weights(model_single, model, 2)
    # model.set_parameters('schedule_generator_2')
    # print([(pos, model.get_parameters()['policy'][pos].shape) for pos in model.get_parameters()['policy']])
#%%
    try:
        model.learn(total_timesteps=10000000)
    except KeyboardInterrupt:
        pass

#%%
    # model.save(f"schedule_generator_{N_PERSONS}")
    env_ = ScheduleGym()
    obs = env_.reset()
    # print(model.predict(obs,deterministic=True))
    # print(env_.env_state['shift'])
    # print(obs.flatten())
    # model = pret.policy
    while True:
        action, _states = model.predict(obs, deterministic=True)
        # print(model.policy(torch.Tensor(obs)))
        obs, reward, done, info = env_.step(action)
        # action, _states = model.predict(obs, deterministic=True)
        print('reward is', reward)
        # env_.render()
        if done:
            break
    field = env_.env_state['hours']

    field = np.concatenate([
        field, 
        np.sum(field, axis=0).reshape(1, -1), 
        env_.target_hours.reshape(1, -1)
        ], axis=0)
    plt.figure(figsize=(30, 5))
    sns.heatmap(field, annot=True)
    # plt.savefig('big_schedule_2.png')
    plt.show()

# %%

# # %%
# #%%
#     from copy import deepcopy
#     def evaluate_position(solution, sol_idx=0):
#         env_state_copy = copy(env_.env_state)
#         env_copy = ScheduleGym()
#         env_copy.target_hours = copy(env_.target_hours)
#         env_copy.env_state = env_state_copy
#         obs, reward, done, info = env_copy.step(solution)
#         if done:
#             return -1
#         val = model.policy(torch.Tensor(obs).view(1, -1))[1].detach().numpy()
#         if render:
#             print(reward, val)
#         # field = env_copy.env_state['hours']

#         # field = np.concatenate([
#         #     field, 
#         #     np.sum(field, axis=0).reshape(1, -1), 
#         #     env_copy.target_hours.reshape(1, -1)
#         #     ], axis=0)
#         # plt.figure(figsize=(30, 5))
#         # sns.heatmap(field, annot=True)
#         # plt.savefig('big_schedule_2.png')
#         # plt.show()
#         return reward + val

# #%%
#     render = False
#     with torch.no_grad():
#         env_ = ScheduleGym()
#         obs = env_.reset()
#         print(model.predict(obs,deterministic=True))
#         print(env_.env_state['shift'])
#         print(obs.flatten())

#         print('bad eval', evaluate_position([0] * 8))
#         print('good eval', evaluate_position([4] * 8))

#         obs, _, _, _ = env_.step([4]*8)
#         print('bad eval', evaluate_position([4] * 8))
#         print('good eval', evaluate_position([0] * 8)) 
#         for i in range(10):
#             obs, _, _, _ = env_.step([0]*8) 
#             print('bad eval', evaluate_position([0] * 8))
#             print('good eval', evaluate_position([4] * 8)) 
#             ga_instance = pygad.GA(30, 30, 
#                 evaluate_position, init_range_low=0,
#                 init_range_high=14, gene_type=int,
#                 num_genes=8, sol_per_pop=60
#             )
#             print('plain solution', evaluate_position(
#                 model.predict(obs, deterministic=True)[0]
#             ))
#             print(model.predict(obs, deterministic=True)[0])
#             ga_instance.run()
#             solution, solution_fitness, solution_idx = ga_instance.best_solution()
#             render=True
#             print('best found solution', evaluate_position(solution))
#             render=False
#             obs, _, _, _ = env_.step(solution) 

#             field = env_.env_state['hours']

#             field = np.concatenate([
#                 field, 
#                 np.sum(field, axis=0).reshape(1, -1), 
#                 env_.target_hours.reshape(1, -1)
#                 ], axis=0)
#             plt.figure(figsize=(30, 5))
#             sns.heatmap(field, annot=True)
#             # plt.savefig('big_schedule_2.png')
#             plt.show()
