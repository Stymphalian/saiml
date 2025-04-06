import numpy as np
import unittest
import autograd2 as ag

class NumericalGradientTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        np.random.seed(1)

    def numeric_check(self, forward, *inputs, threshold=1e-6, do_print=False):
        params = np.concatenate([x.value().reshape(-1) for x in inputs])
        y = forward(params)
        y.backward()
        predGrads = np.concatenate([x.grad.value().reshape(-1) for x in inputs])
        
        def forward2(params):
            z1 = forward(params)
            z2 = ag.summation(z1)  # loss is only defined against a single scalar
            return z2.value()

        grads, diff = ag.utils.numeric_gradient_check(forward2, params, predGrads)
        if do_print:
            print()
            print("grads =", grads)
            print("predGrads =", predGrads)
            print("diff = ", diff)
        self.assertTrue(diff < threshold, "diff = {0}\ngrads= {1}".format(diff, grads))
        # self.assertTrue(diff < 10, "diff = {0}\ngrads= {1}".format(diff, grads))

    def unravel_params(self, params, *inputs):
        count = 0
        for x in inputs:
            x._data = params[count:count+x.size].reshape(x.shape)
            count += x.size