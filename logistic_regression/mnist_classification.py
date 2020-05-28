import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from small_unittest_for_torch import MyTorchTest

dataset = MNIST(root='./data/', train=True, transform=transforms.ToTensor())
test_dataset = MNIST(root='./data/', train=False)
tensor_of_image, label = dataset[0]
m = MyTorchTest()
m.tensorAssertShape(tensor_of_image, (1, 28, 28))
nt = tensor_of_image.view(28 * 28)
nt[nt.argmax()] = 0.000
nt = nt.view(1, 28, 28)
plt.imshow(nt[0, :, :], cmap='gray')
plt.show()
