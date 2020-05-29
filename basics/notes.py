import torch
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from small_unittest_for_torch import MyTorchTest


class MyCustomDataSet(Dataset):
    def __init__(self, dataset, targets):
        self.dataset = dataset
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.dataset[idx], self.targets[idx]


##################
# how to initialize
x = torch.tensor([5.5, 3])
print(x)
y = x.new_ones((2, 2, 2, 2))
print(y)
z = x.new_ones(5, 3, dtype=torch.float32)
print(z)
a = torch.randn_like(z, dtype=torch.float)
print(a)

##################
# operators
####
# checking all shapes of add

res1 = a + z
res2 = torch.add(a, z)
res3 = torch.empty(5, 3)
torch.add(a, z, out=res3)
a.add_(z)
m = MyTorchTest()
m.tensorAssertEqual(a, res1)
m.tensorAssertEqual(a, res2)
m.tensorAssertEqual(a, res3)

####
# checking shape and in-place operands
old_shape = z.size()
z_t = z.t()
m.tensorAssertShape(z, old_shape)

z.t_()
m.tensorAssertShape(z, z_t.shape)

####
# resize
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
m.tensorAssertShape(x, (4, 4))
m.tensorAssertShape(y, (16,))
m.tensorAssertShape(z, (2, 8))

####
# matrix multiplication
a1 = torch.ones([4, 5])
v1 = torch.randn([1, 5])
b1 = torch.tensor(10)
print(a1 @ v1.t() + b1)

####
# elementwise multiplication

a2 = torch.ones([4, 5])
a3 = torch.randn([4, 5])
v2 = torch.randn([1, 5])
print(a2 * v2)
print(a2 * a3)

##################
# numpy bridge
b = a.numpy()
c = torch.from_numpy(b)
m.tensorAssertEqual(a, c)
a.add_(1)
m.tensorAssertEqual(a, c)
# it shows that there is always a shallow copy on numpy bridges

##################
# pytorch and matplot differences
# pytorch expects color channels to be first dimension of tensor. but matplot expects it to
#   be the last one (or not being displayed at all)

##################
# parameters of model
# assume a model which inherits nn.module class. model.parameter shows all parameters that
#   model has got (weights, biases, etc)  # noqa

##################
# more emphasis on reshape!
complicated_matrx = torch.ones((12, 12, 100))
simple_matrix = complicated_matrx.reshape([-1, 120])
m.tensorAssertShape(simple_matrix, (120, 120))

##################
# sampler and data loader
data_to_be_sampled = torch.ones((20, 10))
target_to_be_sampled = torch.ones((20, 1))
whole_data = MyCustomDataSet(data_to_be_sampled, target_to_be_sampled)
whole_data.targets = target_to_be_sampled
whole_data.data = data_to_be_sampled

train_sampler = SubsetRandomSampler([0, 2, 4, 6, 8, 10, 12, 14])
train_loader = DataLoader(whole_data, batch_size=1, sampler=train_sampler)
for xb, yb in train_loader:
    print(xb)
    print(len(xb))
    print(yb)
    print(len(yb))

##################
# pytorch dataset
Xs = torch.tensor([[73, 67, 43, 12],
                   [91, 88, 64, 23],
                   [87, 134, 58, 10],
                   [102, 43, 37, 42],
                   [69, 96, 70, 91]], dtype=torch.float32)
Ys = torch.tensor([[56, 70, 31],
                   [81, 101, 10],
                   [119, 133, 211],
                   [22, 37, 49],
                   [103, 119, 20]], dtype=torch.float32)
dataset = TensorDataset(Xs, Ys)
print(dataset[[1, 3]])

##################
# About relu
# It brings non linearity to model and it is wonderful, because there would be a label which is
# not sum of weighted inputs!

##################
# Pros of kernel in convolutional neural networks
##
# Fewer parameters: A small set of parameters (the kernel) is used to calculate outputs of the
# entire image, so the model has much fewer parameters compared to a fully connected layer.
##
# Sparsity of connections: In each layer, each output element only depends on a small number of
# input elements, which makes the forward and backward passes more efficient.
##
# Parameter sharing and spatial invariance: The features learned by a kernel in one part of the
# image can be used to detect similar pattern in a different part of another image.
