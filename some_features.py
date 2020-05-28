import torch
from torch.utils.data import TensorDataset

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

################## pytorch dataset
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
