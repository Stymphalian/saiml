import unittest
from devices import xp
from . import reference

class TestConvReference(unittest.TestCase):

    @unittest.skip("need to fix to work with device.xp")
    def test_conv_single_step(self):
        xp.random.seed(1)
        a_slice_prev = xp.random.randn(3, 4, 4)
        W = xp.random.randn(3, 4, 4)
        b = xp.random.randn(1, 1, 1)
        z = reference.conv_single_step(a_slice_prev, W, b)
        self.assertAlmostEqual(xp.asnumpy(z), -6.999089450680221)

    @unittest.skip("need to fix to work with device.xp")
    def test_conv_forward(self):
        xp.random.seed(1)
        A_prev = xp.random.randn(10,3,4,4)
        W = xp.random.randn(8,3,2,2)
        b = xp.random.randn(8,1,1,1)
        hparameters = {"pad" : 2, "stride": 2}

        Z, cache = reference.conv_forward(A_prev, W, b, hparameters)
        self.assertEqual(Z.shape, (10, 8, 4, 4))

    @unittest.skip("need to fix to work with device.xp")
    def test_conv_backward(self):
        xp.random.seed(1)
        A_prev = xp.random.randn(10,3,4,4)
        W = xp.random.randn(8,3,2,2)
        b = xp.random.randn(8,1,1,1)
        hparameters = {"pad" : 2, "stride": 2}
        Z, cache = reference.conv_forward(A_prev, W, b, hparameters)

        xp.random.seed(1)
        dA, dW, db = reference.conv_backward(Z, cache)
        self.assertAlmostEqual(xp.asnumpy(xp.mean(dA)), 2.9470599583887345)
        self.assertAlmostEqual(xp.asnumpy(xp.mean(dW)), 5.6470033527213745)
        self.assertAlmostEqual(xp.asnumpy(xp.mean(db)), 8.184838514561605)
    

    

if __name__ == '__main__':
    unittest.main()

