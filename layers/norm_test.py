import unittest
from devices import xp
import autograd2 as ag
from .norm import *

class TestLayerNorm(unittest.TestCase):
    def test_layer_norm(self):
        layer = LayerNorm()
        x1 = xp.arange(2*3).reshape(2,3) + 1
        x = ag.Tensor(x1, requires_grad=True)
        output = layer.forward(x)

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(xp.allclose(xp.mean(output.value()), 0.0))
        self.assertTrue(xp.allclose(xp.std(output.value()), 1.0))
        output.backward()

if __name__ == '__main__':
    unittest.main()
