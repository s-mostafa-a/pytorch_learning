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
z = x * 18
# d(9 * x^2)/dx = 18 * x
m.tensorAssertEqual(z, x.grad)
