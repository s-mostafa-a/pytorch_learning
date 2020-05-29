import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from simple_deep_neural_net.dnn_model import DNNModel


class MNISTModel:

    def __init__(self, batch_size=100):
        self.trains = MNIST(root='../data/', train=True, transform=transforms.ToTensor())
        self.tests = MNIST(root='../data/', train=False, transform=transforms.ToTensor())
        in_features = self.trains.data.size()[1] * self.trains.data.size()[2]
        out_features = len(self.tests.targets.unique())
        self.model = DNNModel(in_features=in_features, hidden_features=32,
                              out_features=out_features)
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

    def fit(self, epochs=5, lr=0.5, loss_fn=F.cross_entropy, opt_fn=torch.optim.SGD):
        losses, metrics = [], []
        opt = opt_fn(self.model.parameters(), lr=lr)
        result = self.accuracy(loss_fn)
        loss_on_test, total, test_metric = result
        print(
            f'''Epoch [0/{epochs}], Loss: {loss_on_test}, accuracy percentage: {
            test_metric}''')

        for epoch in range(epochs):
            for xb, yb in self.train_dataloader:
                loss, _, _ = self.loss_on_batch(loss_fn, xb, yb, opt)

            result = self.accuracy(loss_fn)
            loss_on_test, total, test_metric = result
            losses.append(loss_on_test)
            metrics.append(test_metric)
            print(
                f'''Epoch [{epoch + 1}/{epochs}], Loss: {loss_on_test}, accuracy percentage: {
                test_metric}''')
        return losses, metrics


mnist = MNISTModel()
mnist.fit()
