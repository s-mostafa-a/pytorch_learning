import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn


class RegressionClassifierUsingNN:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        Xs = torch.from_numpy(x)
        Ys = torch.from_numpy(y)
        self.dataset = TensorDataset(Xs, Ys)
        self.model = nn.Linear(in_features=Xs.size()[1], out_features=Ys.size()[1])

    def predict(self, inputs):
        return self.model(inputs)

