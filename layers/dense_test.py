import unittest
from devices import xp
import autograd2 as ag
from .dense import Linear
import base_gradient_test

class TestDense(base_gradient_test.NumericalGradientTest):

    def test_linear(self):
        layer = Linear(4,3)
        # TODO: Bug with +1.0 on arange, the resulting Tensor doesn't share the requires_grad
        x0 = xp.arange(2*3*4).reshape(2,3,4) + 1.0
        x = ag.Parameter(x0)
        y = layer.forward(x)
        y.backward()
        self.assertEqual(y.shape, (2,3,3))

        want1 = xp.matmul(x0[0], layer.w.value()) + layer.b.value()
        want2 = xp.matmul(x0[1], layer.w.value()) + layer.b.value()
        self.assertTrue(xp.allclose(y.value()[0], want1))
        self.assertTrue(xp.allclose(y.value()[1], want2))

        def forward(params):
            self.unravel_params(params, x)
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        self.numeric_check(forward, x)

if __name__ == '__main__':
    unittest.main()
