import numpy as np
from schedule_env import Environment
from open_spiel.python import rl_environment
import tensorflow.compat.v1 as tf
tf.disable_v2_behaviour()


def print_beautiful_schedule(obs):
    field = np.zeros(obs.shape[:2] + (10,))
    field[:, :, 1:] = obs[:, :, :9]
    field[:, :, 0] = 1 - field[:, :, 1:].sum(axis=2)
    print(field.reshape(-1, 10).dot(np.array([0] + list(range(4, 13))).reshape(10, 1)).reshape(obs.shape[:2]))


if __name__ == "__main__":
    # env = Environment(12, 62, np.random.random(size=(12, 1)), np.random.random(size=(1, 62)))
    # time_step = env.reset()
    # i=0
    # while time_step.step_type != rl_environment.StepType.LAST:
    #     i+=1
    #     actions = env.action_spec()
    #     action = np.random.choice(range(actions['num_actions']))
    #     time_step = env.step([action])
    
    # print(print_beautiful_schedule(time_step.observations['info_state'][0]))


    with tf.Session() as sess:
        pass