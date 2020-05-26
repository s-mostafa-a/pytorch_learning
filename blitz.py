from __future__ import print_function
import numpy as np
import torch
import time

t1 = time.time()
for i in range(1000000):
    x_tensor = torch.empty(5, 3)
t2 = time.time()
for i in range(1000000):
    x_ndarr = np.empty((5, 3))
t3 = time.time()
print('making empty array comparison:')
delta1 = t2 - t1
delta2 = t3 - t1
print(f'torch: {delta1} sec')
print(f'numpy: {delta2} sec')
print(f'''winner: {'torch' if delta1 < delta2 else 'numpy'}''')
# my computer's outputs (macbook pro without cuda):
# making empty array comparison:
# torch: 2.2384519577026367 sec
# numpy: 2.758033275604248 sec
# winner: torch

t4 = time.time()
for i in range(1000000):
    x_tensor = torch.zeros(5, 3)
t5 = time.time()
for i in range(1000000):
    x_ndarr = np.zeros((5, 3))
t6 = time.time()
print('making zeros array comparison:')
delta3 = t5 - t4
delta4 = t6 - t5
print(f'torch: {delta3} sec')
print(f'numpy: {delta4} sec')
print(f'''winner: {'torch' if delta3 < delta4 else 'numpy'}''')
# my computer's outputs (macbook pro without cuda):
# making zeros array comparison:
# torch: 3.497465133666992 sec
# numpy: 0.5160698890686035 sec
# winner: numpy
