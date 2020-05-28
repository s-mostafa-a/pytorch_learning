import torch
import torchvision
from torchvision.datasets import MNIST
dataset = MNIST('./data/', download=True)
len(dataset)
