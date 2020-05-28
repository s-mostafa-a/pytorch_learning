import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from small_unittest_for_torch import MyTorchTest
import torch.nn.functional as F
import numpy as np


def split_train_and_test_indices(n, test_percentage):
    number_of_tests = int(test_percentage * n)
    per = np.random.permutation(n)
    return per[number_of_tests:], per[:number_of_tests]


dataset = MNIST(root='./data/', train=True, transform=transforms.ToTensor())
test_dataset = MNIST(root='./data/', train=False)
tensor_of_image, label = dataset[0]
m = MyTorchTest()
m.tensorAssertShape(tensor_of_image, (1, 28, 28))


class LogisticRegressionModel(nn.Module):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size=100):
        super().__init__()
        in_features = x.size()[1] * x.size()[2]
        out_features = len(y.unique())
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.train_indices, self.test_indices = split_train_and_test_indices(len(dataset),
                                                                             test_percentage=0.2)
        self.batch_size = batch_size

    def accuracy_on_train(self):
        test_sampler = SubsetRandomSampler(self.test_indices)
        test_loader = DataLoader(dataset, self.batch_size, sampler=test_sampler)
        all_trues = 0
        for xb, yb in test_loader:
            xb = xb.reshape(-1, self.linear.in_features)
            outs = self.linear(xb)
            _, preds = torch.max(outs, dim=1)
            all_trues += torch.sum(preds == yb).item()
        return all_trues / len(self.test_indices)

    def train(self, epochs=1, step=0.001):
        opt = torch.optim.SGD(self.parameters(), lr=step)
        for i in range(epochs):
            train_sampler = SubsetRandomSampler(self.train_indices)
            train_loader = DataLoader(dataset, self.batch_size, sampler=train_sampler)
            for xb, yb in train_loader:
                xb = xb.reshape(-1, self.linear.in_features)
                outs = self.linear(xb)
                # bellow function named cross entropy has implemented softmax internally and maps labels to one-hot arrays(i.e 9:[0,0,0,0,0,0,0,0,0,1]).# noqa
                loss = F.cross_entropy(outs, yb)
                loss.backward()
                opt.step()
                opt.zero_grad()


model = LogisticRegressionModel(dataset.data, dataset.targets)
print(model.accuracy_on_train())
model.train()
print(model.accuracy_on_train())
