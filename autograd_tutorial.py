import torch
from torch_test import MyTorchTest

m = MyTorchTest()
x = torch.ones(2, 2, requires_grad=True)
xd = x.detach()
m.assertEqual(xd.grad_fn, None)
y = x * x * 9
with torch.no_grad():
    yd = y ** 2
m.assertEqual(yd.grad_fn, None)
y.backward(x)
# d(9 * x^2 * 10)/dx = 180 * x
z = x * 18
m.tensorAssertEqual(z, x.grad)
