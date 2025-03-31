import unittest
import numpy as np
from . import reference

class TestConvReference(unittest.TestCase):
    def test_conv_single_step(self):
        np.random.seed(1)
        a_slice_prev = np.random.randn(3, 4, 4)
        W = np.random.randn(3, 4, 4)
        b = np.random.randn(1, 1, 1)
        z = reference.conv_single_step(a_slice_prev, W, b)
        self.assertAlmostEqual(z, -6.999089450680221)

    def test_conv_forward(self):
        np.random.seed(1)
        A_prev = np.random.randn(10,3,4,4)
        W = np.random.randn(8,3,2,2)
        b = np.random.randn(8,1,1,1)
        hparameters = {"pad" : 2, "stride": 2}

        Z, cache = reference.conv_forward(A_prev, W, b, hparameters)
        self.assertEqual(Z.shape, (10, 8, 4, 4))

    def test_conv_backward(self):
        np.random.seed(1)
        A_prev = np.random.randn(10,3,4,4)
        W = np.random.randn(8,3,2,2)
        b = np.random.randn(8,1,1,1)
        hparameters = {"pad" : 2, "stride": 2}
        Z, cache = reference.conv_forward(A_prev, W, b, hparameters)

        np.random.seed(1)
        dA, dW, db = reference.conv_backward(Z, cache)
        self.assertAlmostEqual(np.mean(dA), 2.9470599583887345)
        self.assertAlmostEqual(np.mean(dW), 5.6470033527213745)
        self.assertAlmostEqual(np.mean(db), 8.184838514561605)
    

    

if __name__ == '__main__':
    unittest.main()

