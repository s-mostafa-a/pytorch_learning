import torch

################## matrix multiplication
a1 = torch.ones([4, 5])
v1 = torch.randn([1, 5])
print(a1 @ v1.t())
