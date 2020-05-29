import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from small_unittest_for_torch import MyTorchTest
import torch.nn.functional as F


class LogisticRegressionModel(nn.Module):
    def __init__(self, train_dataset, test_dataset, batch_size=100):
        super().__init__()
        in_features = train_dataset.data.size()[1] * train_dataset.data.size()[2]
        out_features = len(train_dataset.targets.unique())
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def accuracy(self, indices, dataset):
        test_sampler = SubsetRandomSampler(indices)
        test_loader = DataLoader(dataset, self.batch_size, sampler=test_sampler)
        all_trues = 0
        for xb, yb in test_loader:
            xb = xb.reshape(-1, self.linear.in_features)
            outs = self.linear(xb)
            _, preds = torch.max(outs, dim=1)
            all_trues += torch.sum(preds == yb).item()
        return all_trues / len(indices)

    def train(self, epochs=1, step=0.001):
        opt = torch.optim.SGD(self.parameters(), lr=step)
        for i in range(epochs):
            print(f'''epoch: {i}/{epochs}, accuracy on test: {self.accuracy(
                indices=[i for i in range(len(self.test_dataset.targets))],
                dataset=self.test_dataset)}''')
            train_sampler = SubsetRandomSampler(
                [i for i in range(len(self.train_dataset.targets))])
            train_loader = DataLoader(self.train_dataset, self.batch_size, sampler=train_sampler)
            for xb, yb in train_loader:
                xb = xb.reshape(-1, self.linear.in_features)
                outs = self.linear(xb)
                # bellow function named cross entropy has implemented softmax internally and maps labels to one-hot arrays(i.e 9:[0,0,0,0,0,0,0,0,0,1]).# noqa
                loss = F.cross_entropy(outs, yb)
                loss.backward()
                opt.step()
                opt.zero_grad()


trains = MNIST(root='../data/', train=True, transform=transforms.ToTensor())
tests = MNIST(root='../data/', train=False, transform=transforms.ToTensor())
tensor_of_image, label = trains[0]
m = MyTorchTest()
m.tensorAssertShape(tensor_of_image, (1, 28, 28))

model = LogisticRegressionModel(trains, tests)
model.train(epochs=10)
