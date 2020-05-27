import torch

################## matrix multiplication
a1 = torch.ones([4, 5])
v1 = torch.randn([1, 5])
b1 = torch.tensor(10)
print(a1 @ v1.t() + b1)

################## elementwise multiplication

a2 = torch.ones([4, 5])
a3 = torch.randn([4, 5])
v2 = torch.randn([1, 5])
print(a2 * v2)
print(a2 * a3)
