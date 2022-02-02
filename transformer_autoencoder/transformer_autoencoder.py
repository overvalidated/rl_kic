import torch
import numpy as np
from torch import optim

from schedule_env import ScheduleGym
from transformer_autoencoder.simple_autoencoder import SimpleAutoEncoder

"""
    код, ответственный за обучение модели для сжатия расписаний

    к черту трансформер, обучим простой автоэнкодер
    пока не ясно как можно разделить человека и общество

"""


if __name__ == "__main__":
    # training code here
    # model = TransformerAutoEncoder()
    schedule = ScheduleGym(n_persons=8)

    encoder = SimpleAutoEncoder(8 * 16 * 10)
    adam_optim = optim.AdamOptimizer(encoder.parameters())

    obs = schedule.reset()

    for _ in range(N_BATCHES):
        adam_optim.zero_grad()

