from torchvision.transforms import ToTensor, Compose, Grayscale, ToPILImage
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
### simple composition of transforms
transforms = Compose([ToTensor(),
                      ToPILImage(),
                      Grayscale(),
                      ToTensor()])
cifar = CIFAR10(root='./../data', train=True, transform=transforms)
image_size = 1024
plt.ion()
plt.ioff()
dl = DataLoader(cifar, batch_size=100, shuffle=True)
for images, labels in dl:
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images[:100], 10).permute(1, 2, 0))
    fig.show()
    break
