import unittest
import torch


class MyTorchTest(unittest.TestCase):
    def tensorAssertEqual(self, first: torch.tensor, second: torch.tensor, msg=None):
        self.assertEqual(torch.all(torch.eq(first, second)), torch.tensor(True), msg=msg)

    def tensorAssertShape(self, tensor: torch.tensor, shape: tuple):
        size = torch.Size(shape)
        self.assertEqual(tensor.size(), size)
