
import torch
from torch import nn, optim
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

class MLP(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.fc1 = nn.Conv2d(11, 8, (3, 3), padding='same')
        # self.fc2 = nn.Conv2d(32, 32, (3, 3), padding='same')
        # self.fc3 = nn.Conv2d(32, 32, (3, 3), padding='same')
        # self.fc4 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.act4 = nn.ReLU()
        self.fc5 = nn.Linear(8, 10)
        self.q_ = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.Tensor(x)
        x = torch.reshape(x, (-1, 4, 62, 11))
        inp_ = torch.transpose(x, 3, 1)
        x = self.act1(self.fc1(inp_))
        # x = self.act2(self.fc2(x)+x)
        # x = self.act3(self.fc3(x)+x)
        # x = self.act4(self.fc4(x)+x)
        x = torch.mean(x, 2)
        x = torch.mean(x, 2)
        act = self.fc5(x)
        q_ = self.q_(x)
        return act, nn.Softmax(dim=-1)(act), q_

    def step(self, x):
        _, act, q_ = self.forward(x)
        return act.numpy(), q_.numpy()