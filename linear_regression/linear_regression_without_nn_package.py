import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.nn import functional as F


class RegressionClassifierUsingNN:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        self.model = nn.Linear(in_features=Xs.size()[1], out_features=Ys.size()[1])

    def predict(self, inputs):
        return self.model(inputs)

    @staticmethod
    def absolute(t1, t2):
        diff = torch.abs(t1 - t2)
        return torch.sum(diff) / diff.numel()

    def train(self, epochs=500, step=1e-5, loss_func='mse'):
        opt = torch.optim.SGD(self.model.parameters(), lr=step)
        for i in range(epochs):
            train_dl = DataLoader(self.dataset, batch_size=5, shuffle=True)
            for x_b, y_b in train_dl:
                predictions = self.predict(x_b)
                if loss_func == 'mse':
                    loss = F.mse_loss(predictions, y_b)
                elif loss_func == 'abs':
                    loss = self.absolute(predictions, y_b)
                else:
                    raise Exception('Not implemented such loss function')
                loss.backward()
                opt.step()
                opt.zero_grad()


Xs = np.array([[73, 67, 43, 12],
               [91, 88, 64, 23],
               [87, 134, 58, 10],
               [102, 43, 37, 42],
               [69, 96, 70, 91]], dtype='float32')
Ys = np.array([[56, 70, 31],
               [81, 101, 10],
               [119, 133, 211],
               [22, 37, 49],
               [103, 119, 20]], dtype='float32')
classifier = RegressionClassifierUsingNN(x=Xs, y=Ys)
classifier.train(epochs=10000)

Y_tilde = classifier.predict(classifier.dataset.tensors[0])
print('final loss:', F.mse_loss(Y_tilde, classifier.dataset.tensors[1]).item())
print(Y_tilde)
print(classifier.dataset.tensors[1])
