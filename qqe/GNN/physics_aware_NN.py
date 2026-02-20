

import torch
import torch.nn as nn
import numpy as np



class Parameters_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.linear1(x)
        h = self.relu(h)
        h = self.linear2(h)
        h = self.relu(h)
        h = self.linear3(h)
        return h


class GNN(nn.Module):...
