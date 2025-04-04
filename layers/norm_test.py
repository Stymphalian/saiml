import unittest
import numpy as np
import autograd2 as ag
from .norm import *

class TestLayerNorm(unittest.TestCase):
    def test_layer_norm(self):
        layer = LayerNorm()
        x1 = np.arange(2*3).reshape(2,3) + 1
        x = ag.Tensor(x1, requires_grad=True)
        output = layer.forward(x)

        self.assertEqual(output.shape, x.shape)
        self.assertAlmostEquals(np.mean(output.value()), 0.0)
        self.assertAlmostEqual(np.std(output.value()), 1.0)
        output.backward()

if __name__ == '__main__':
    unittest.main()
