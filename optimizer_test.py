import unittest
import numpy as np
import autograd2 as ag
import optimizer

class TestOptimizers(unittest.TestCase):

    def test_sgd_momentum(self):
        np.random.seed(1)
        lr = 0.5
        B = 0.9
        opt = optimizer.SGDMomentum(lr=lr, momentum=B)

        x1 = ag.Tensor(np.random.normal(2,3,5), requires_grad=True)
        loss = ag.mean(x1)
        x2 = opt.step(x1, loss)
        self.assertTrue(np.array_equal(x2.value(), x1.value() - lr*(1-B)*loss.value()))

        x1._data = x2.value()
        loss = ag.mean(x1)
        x3 = opt.step(x1, loss)


if __name__ == '__main__':
    unittest.main()

