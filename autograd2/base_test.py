import unittest
import numpy as np
import autograd2 as ag

class TestAutogradOperators(unittest.TestCase):

    def numeric_check(self, forward, *inputs, do_print=False):
        params = np.concatenate([x.value().reshape(-1) for x in inputs])
        forward(params).backward()
        predGrads = np.concatenate([x.grad.value().reshape(-1) for x in inputs])
        
        def forward2(params):
            z1 = forward(params)
            z2 = ag.summation(z1)  # loss is only defined against a single scalar
            return z2.value()

        grads, diff = ag.utils.numeric_gradient_check(forward2, params, predGrads)
        if do_print:
            print(grads)
            print(predGrads)
            print(diff)
        self.assertTrue(diff < 1e-6, "diff = {0}\ngrads= {1}".format(diff, grads))

    def unravel_params(self, params, *inputs):
        count = 0
        for x in inputs:
            x.data = params[count:count+x.data.size].reshape(x.data.shape)
            count += x.data.size

    def test_sum(self):
        x1 = ag.Tensor(np.arange(3*3).reshape(3,3), dtype=np.float64, requires_grad=True)
        x2 = ag.Tensor(np.arange(3*3).reshape(3,3) + 9, dtype=np.float64, requires_grad=True)
        print(x1 * x2)
        # print(ag.summation(x1))
        # x2 = ag.Tensor(np.arange(3*3).reshape(3,3) + 9, dtype=np.float64, requires_grad=True)
        # ls = [x1, x2]
        # print(sum(ls))

    def test_gradient(self):
        x1 = ag.Tensor(2, dtype=np.float64, requires_grad=True)
        x2 = ag.Tensor(5, dtype=np.float64, requires_grad=True)
        v7 = ag.log(x1) + x1*x2 - ag.sin(x2)
        v7.requires_grad = True

        pred = v7.value()
        v7.backward()
        v7.grad.backward()
        self.assertAlmostEquals(pred, 11.652071455223084)
        self.assertAlmostEquals(x1.grad.value(), 5.5)
        self.assertAlmostEquals(x2.grad.value(), 1.7163378)


if __name__ == '__main__':
    unittest.main()
