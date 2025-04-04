import unittest
import numpy as np
import autograd2 as ag
from .sampling import *
import base_gradient_test

class TestSamplingLayer(base_gradient_test.NumericalGradientTest):

    def test_sampling(self):
        np.random.seed(1)
        layer = Sampling()
        z_mean = ag.Parameter(np.random.rand(6,1))
        z_var = ag.Parameter(np.random.rand(6,1))

        def do():
            layer.rng = np.random.default_rng(layer.seed)
            got = layer.forward(z_mean, z_var)  
            loss = ag.mean(got)
            return loss
        def forward(params):
            self.unravel_params(params, z_mean, z_var)
            return do()
        self.numeric_check(forward, z_mean, z_var)

        



if __name__ == '__main__':
    unittest.main()

