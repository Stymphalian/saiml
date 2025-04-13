import unittest
import autograd2 as ag
import base_gradient_test
from devices import xp

class TestAutogradOperators(base_gradient_test.NumericalGradientTest):

    def test_gradient(self):
        x1 = ag.Tensor(2, dtype=xp.float64, requires_grad=True)
        x2 = ag.Tensor(5, dtype=xp.float64, requires_grad=True)
        v7 = ag.log(x1) + x1*x2 - ag.sin(x2)
        v7.requires_grad = True

        pred = v7.value()
        v7.backward()
        v7.grad.backward()
        self.assertAlmostEquals(pred, 11.652071455223084)
        self.assertAlmostEquals(x1.grad.numpy(), 5.5)
        self.assertAlmostEquals(x2.grad.numpy(), 1.7163378)


if __name__ == '__main__':
    unittest.main()
