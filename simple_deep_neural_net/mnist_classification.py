import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from small_unittest_for_torch import MyTorchTest
import torch.nn.functional as F


class DNNModel(nn.Module):
    pass


trains = MNIST(root='../data/', train=True, transform=transforms.ToTensor())
tests = MNIST(root='../data/', train=False, transform=transforms.ToTensor())
