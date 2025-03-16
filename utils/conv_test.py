import unittest
import numpy as np
import utils
from scipy import signal

class TestConv(unittest.TestCase):


    def test_convole2d_stride(self):
        X = np.array([
            [0,0,0,0,0,0,0],
            [0,1,2,3,4,5,0],
            [0,6,5,4,3,2,0],
            [0,2,3,4,5,6,0],
            [0,7,6,5,4,3,0],
            [0,3,4,5,6,7,0],
            [0,0,0,0,0,0,0]
        ])
        kernel = np.array([
            [0,1,0],
            [1,2,1],
            [0,1,0],
        ])
        got = utils.convolve2D(X, kernel, stride=2)
        want = np.array([
            [10,16,16],
            [20,25,22],
            [17,25,23],
        ])
        self.assertTrue(np.array_equal(got, want))

    def test_convolve2d_against_scipy(self):
        X = np.arange(20).reshape(4, 5) + 1
        kernel = np.array([[1,0],[0,1]])
        got = utils.convolve2D(X, kernel)
        want = signal.correlate2d(X, kernel, mode='valid')
        self.assertTrue(np.array_equal(got, want))

    def test_full_convolve2d_against_scipy(self):
        X = np.arange(9).reshape(3,3) + 1
        kernel = np.array([[1,0],[0,1]])
        got = utils.full_convolve2D(X, kernel)
        want = signal.convolve2d(X, kernel, mode='full')
        self.assertTrue(np.array_equal(got, want))

    def test_convolve3d_simple(self):
        X = np.arange(40).reshape(4, 5, 2) + 1
        kernel = np.array([
            [[1,0],[0,1]],
            [[0,-1],[1,0]]
        ])
        got = utils.convolve3D(X, kernel)
        want = np.array([
            [6, 10, 14, 18],
            [26, 30, 34, 38],
            [46, 50, 54, 58]
        ])
        self.assertTrue(np.array_equal(got, want))

    def test_convolve3d_against_scipy(self):
        X = np.arange(40).reshape(4, 5, 2) + 1
        kernel = np.array([
            [[1,0],[0,1]],
            [[0,-1],[1,0]]
        ])
        got = utils.convolve3D(X, kernel)
        x1 = signal.correlate2d(X[:,:,0], kernel[:,:,0], mode='valid')
        x2 = signal.correlate2d(X[:,:,1], kernel[:,:,1], mode='valid')
        want = x1 + x2
        self.assertTrue(np.array_equal(got, want))

    def test_full_convolve3d_simple(self):
        X = np.arange(9).reshape(3,3,1) + 1
        kernel = np.array([[[1],[2]],[[2],[3]]])
        got = utils.full_convolve3D(X, kernel)
        want = np.array([
            [1,4,7,6],
            [6,20,28,21],
            [15,44,52,36],
            [14,37,42,27]
        ])
        self.assertTrue(np.array_equal(got, want))

    def test_full_convolve3d_against_scipy(self):
        X = np.arange(9).reshape(3,3,1) + 1
        kernel = np.array([[[1],[2]],[[2],[3]]])
        got = utils.full_convolve3D(X, kernel)
        want = signal.convolve2d(X[:,:,0], kernel[:,:,0], mode='full')
        self.assertTrue(np.array_equal(got, want))
     
    def test_conv_single_step(self):
        np.random.seed(1)
        a_slice_prev = np.random.randn(3, 4, 4)
        W = np.random.randn(3, 4, 4)
        b = np.random.randn(1, 1, 1)
        z = utils.conv_single_step(a_slice_prev, W, b)
        self.assertAlmostEqual(z, -6.999089450680221)

    def test_conv_forward(self):
        np.random.seed(1)
        A_prev = np.random.randn(10,3,4,4)
        W = np.random.randn(8,3,2,2)
        b = np.random.randn(8,1,1,1)
        hparameters = {"pad" : 2, "stride": 2}

        Z, cache = utils.conv_forward(A_prev, W, b, hparameters)
        self.assertEqual(Z.shape, (10, 8, 4, 4))

    def test_conv_backward(self):
        np.random.seed(1)
        A_prev = np.random.randn(10,3,4,4)
        W = np.random.randn(8,3,2,2)
        b = np.random.randn(8,1,1,1)
        hparameters = {"pad" : 2, "stride": 2}
        Z, cache = utils.conv_forward(A_prev, W, b, hparameters)

        np.random.seed(1)
        dA, dW, db = utils.conv_backward(Z, cache)
        self.assertAlmostEqual(np.mean(dA), 2.9470599583887345)
        self.assertAlmostEqual(np.mean(dW), 5.6470033527213745)
        self.assertAlmostEqual(np.mean(db), 8.184838514561605)

    

if __name__ == '__main__':
    unittest.main()

