import torch

################## matrix multiplication
a1 = torch.ones([4, 5])
v1 = torch.randn([1, 5])
b1 = torch.tensor(10)
print(a1 @ v1.t() + b1)
