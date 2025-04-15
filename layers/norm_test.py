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

    def test_layer_norm2(self):
        xp.random.seed(1)
        batches = 2
        x = ag.arange((batches, 2,3,4,5,6)) + 1.0
        layer = LayerNorm2((x.shape[1:]))
        y = layer.forward(x)
        y.backward()
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(xp.allclose(xp.mean(y.value()[0]), 0.0))
        self.assertTrue(xp.allclose(xp.std(y.value()[0]), 1.0))


    @unittest.skip("Numerically unstable for numeric gradient checking")
    def test_layer_norm_gradient(self):
        batches = 2
        input_shape = (4,3,2)
        layer = LayerNorm2(input_shape)
        x = ag.Parameter(xp.random.rand(batches, *input_shape))
        def do():
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        def forward(params):
            self.unravel_params(params, x)
            return do()
        self.numeric_check(forward, x)

if __name__ == '__main__':
    unittest.main()
