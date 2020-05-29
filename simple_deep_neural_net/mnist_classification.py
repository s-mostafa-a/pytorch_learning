import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from small_unittest_for_torch import MyTorchTest
import torch.nn.functional as F


class DNNModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.hidden = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.output = nn.Linear(in_features=hidden_features, out_features=out_features)

    def forward(self, xb):
        xb = xb.view(xb.size(0), -1)
        before_activation_func = self.hidden(xb)
        after_activation_func = F.relu(before_activation_func)
        return self.output(after_activation_func)


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

    def loss_on_batch(self, loss_func, xb, yb, opt=None, metric=None):
        predictions = self.model(xb)
        loss = loss_func(predictions, yb)
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()
        metric_result = None
        if metric is not None:
            metric_result = metric(predictions, yb)
        return loss.item(), len(xb), metric_result
