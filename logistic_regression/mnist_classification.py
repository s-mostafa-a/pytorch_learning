from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from small_unittest_for_torch import MyTorchTest
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
train_indices, test_indices = split_train_and_test_indices(len(dataset), test_percentage=0.2)
batch_size = 100
train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(dataset, batch_size, sampler=train_sampler)
test_sampler = SubsetRandomSampler(test_indices)
test_loader = DataLoader(dataset, batch_size, sampler=test_sampler)


class LogisticRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, xb):
        xb = xb.reshape(-1, self.linear.in_features)
        return self.linear(xb)


model = LogisticRegressionModel(tensor_of_image.size()[-1] ** 2, 10)
