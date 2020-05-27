import torch
from torch_test import MyTorchTest

m = MyTorchTest()
x = torch.ones(2, 2, requires_grad=True)
y = x * x * 9
y.backward(x)
z = x * 18

with torch.no_grad():
    y = y ** 2

# d(9 * x^2)/dx = 18 * x
m.tensorAssertEqual(z, x.grad)
