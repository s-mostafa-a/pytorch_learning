import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class CIFARModel:

    def __init__(self, batch_size=100):
        self.trains = CIFAR10(root='../data/', train=True, transform=ToTensor())
        self.tests = CIFAR10(root='../data/', train=False, transform=ToTensor())
        height = self.trains.data.shape[2]
        width = self.trains.data.shape[1]
        in_features = width * height
        classes = len(set(self.tests.targets))
        channels = self.trains.data.shape[3]
        self.model = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: bs x 16 x 16 x 16
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: bs x 16 x 8 x 8
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: bs x 16 x 4 x 4
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: bs x 16 x 2 x 2
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: bs x 16 x 1 x 1,
            nn.Flatten(),  # output: bs x 16
            nn.Linear(16, classes)  # output: bs x 10
        )
        self.train_dataloader = DataLoader(self.trains, batch_size)
        self.test_dataloader = DataLoader(self.tests, batch_size)

    def loss_on_batch(self, loss_func, xb, yb, opt=None):
        probabilities = self.model(xb)
        loss = loss_func(probabilities, yb)
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()
        _, predictions = torch.max(probabilities, dim=1)
        metric_result = torch.sum(predictions == yb).item() / len(predictions)
        return loss.item(), len(xb), metric_result

    def accuracy(self, loss_fn):
        with torch.no_grad():
            results = [self.loss_on_batch(loss_fn, xb, yb)
                       for xb, yb in self.test_dataloader]
            losses, nums, metrics = zip(*results)
            total = np.sum(nums)
            avg_loss = np.sum(np.multiply(losses, nums)) / total
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
        return avg_loss, total, avg_metric

    def fit(self, epochs=10, lr=0.005, loss_fn=F.cross_entropy, opt_fn=torch.optim.Adam):
        losses, metrics = [], []
        opt = opt_fn(self.model.parameters(), lr=lr)
        result = self.accuracy(loss_fn)
        loss_on_test, total, test_metric = result
        print(
            f'''Epoch [0/{epochs}], Loss: {loss_on_test}, accuracy percentage: {
            test_metric}''')

        for epoch in range(epochs):
            self.model.train()
            for xb, yb in self.train_dataloader:
                loss, _, _ = self.loss_on_batch(loss_fn, xb, yb, opt)
            self.model.eval()
            result = self.accuracy(loss_fn)
            loss_on_test, total, test_metric = result
            losses.append(loss_on_test)
            metrics.append(test_metric)
            print(
                f'''Epoch [{epoch + 1}/{epochs}], Loss: {loss_on_test}, accuracy percentage: {
                test_metric}''')
        return losses, metrics


cifar = CIFARModel()
cifar.fit()
