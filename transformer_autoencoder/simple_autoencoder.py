import torch
from torch import nn
import numpy as np

class SimpleAutoEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__(self)
        self.hidden1 = nn.Linear(in_dim, 128)
        self.relu_activation1 = nn.ReLU()

        self.hidden2 = nn.Linear(128, 32)
        self.relu_activation2 = nn.ReLU()

        self.hidden3 = nn.Linear(32, 128)
        self.relu_activation3 = nn.ReLU()

        self.out = nn.Linear(128, in_dim)

    def forward(self, x):
        hl1 = self.hidden1(x)
        act1 = self.relu_activation1(hl1)

        hl2 = self.hidden2(act1)
        act2 = self.relu_activation2(hl2)

        hl3 = self.hidden3(act2)
        act3 = self.relu_activation3(hl3)
        
        return self.out(act3)
