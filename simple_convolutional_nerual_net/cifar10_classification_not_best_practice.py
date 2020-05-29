import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler


class CNNModel:
    def __init__(self, train_dataset, test_dataset, batch_size=100):
        super().__init__()
        classes = len(set(tests.targets))
        channels = trains.data.shape[3]
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

        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def accuracy(self, indices, dataset):
        with torch.no_grad():
            test_sampler = SubsetRandomSampler(indices)
            test_loader = DataLoader(dataset, self.batch_size, sampler=test_sampler)
            all_trues = 0
            for xb, yb in test_loader:
                outs = self.model(xb)
                _, preds = torch.max(outs, dim=1)
                all_trues += torch.sum(preds == yb).item()
        return all_trues / len(indices)

    def train(self, epochs=10, step=0.005):
        opt = torch.optim.Adam(self.model.parameters(), lr=step)
        for i in range(epochs):
            self.model.eval()
            print(f'''epoch: {i}/{epochs}, accuracy on test: {self.accuracy(
                indices=[i for i in range(len(self.test_dataset.targets))],
                dataset=self.test_dataset)}''')
            train_sampler = SubsetRandomSampler(
                [i for i in range(len(self.train_dataset.targets))])
            train_loader = DataLoader(self.train_dataset, self.batch_size, sampler=train_sampler)
            self.model.train()
            for xb, yb in train_loader:
                outs = self.model(xb)
                # bellow function named cross entropy has implemented softmax internally and maps labels to one-hot arrays(i.e 9:[0,0,0,0,0,0,0,0,0,1]).# noqa
                loss = F.cross_entropy(outs, yb)
                loss.backward()
                opt.step()
                opt.zero_grad()


trains = CIFAR10(root='../data/', train=True, transform=ToTensor())
tests = CIFAR10(root='../data/', train=False, transform=ToTensor())

model = CNNModel(trains, tests)
model.train()
