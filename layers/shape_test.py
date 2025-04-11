import unittest
import numpy as np
import autograd2 as ag
from .shape import *
import optimizer

class TestShapingLayers(unittest.TestCase):

    def test_flatten(self):
        layer = Flatten()
        context = {"optimizer": optimizer.SGD(0.01)}
        X = ag.Tensor(np.random.rand(2, 1, 3, 3), requires_grad=True)
        output = layer.forward(X)
        self.assertEqual(output.shape, (18, 1))
        output.backward()
        layer.backward(context)
        self.assertEqual(X.grad.shape, X.shape)


if __name__ == '__main__':
    unittest.main()

