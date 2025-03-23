import unittest
import numpy as np
import autograd2 as ag

class TestAutogradOperators(unittest.TestCase):

    def test_gradient(self):
        x1 = ag.Tensor(2, dtype=np.float64, requires_grad=True)
        x2 = ag.Tensor(5, dtype=np.float64, requires_grad=True)
        v7 = ag.log(x1) + x1*x2 - ag.sin(x2)

        pred = v7.value()
        v7.backward()
        self.assertAlmostEquals(pred, 11.652071455223084)
        self.assertAlmostEquals(x1.grad, 5.5)
        self.assertAlmostEquals(x2.grad, 1.7163378)    


if __name__ == '__main__':
    unittest.main()
