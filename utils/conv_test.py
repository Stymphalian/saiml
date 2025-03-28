import unittest
import numpy as np
import utils
from utils.conv import (
    _convolve2d_iterative,
    _convolve2d_gradient_iterative,
    _convolve2d_vectorized,
    _convolve2d_gradient_vectorized
)
from scipy import signal
import torch

class TestConv(unittest.TestCase):

    def test_convolve2d_dilation(self):
        np.random.seed(1)
        x = np.round(np.random.rand(1,5,5), 2)
        k = np.round(np.random.rand(1,3,3), 2)
        pred = utils.convolve2d(x, k, dilate=2)
        outGrad = np.random.rand(*pred.shape)
        got = utils.convolve2d_gradient(x, k, outGrad, dilate=2)

    def test_convolve2d_simple(self):
        x = np.array([[
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]])
        kernel = np.array([[
            [0,1],
            [2,1]
        ]])
        got = utils.convolve2d(x, kernel)
        want = np.array([[
            [15, 19],
            [27, 31]
        ]])
        self.assertTrue(np.array_equal(got, want))

    def test_convole2d_stride(self):
        x = np.array([[
            [0,0,0,0,0,0,0],
            [0,1,2,3,4,5,0],
            [0,6,5,4,3,2,0],
            [0,2,3,4,5,6,0],
            [0,7,6,5,4,3,0],
            [0,3,4,5,6,7,0],
            [0,0,0,0,0,0,0]
        ]])
        kernel = np.array([[
            [0,1,0],
            [1,2,1],
            [0,1,0],
        ]])
        got = utils.convolve2d(x, kernel, stride=2)
        want = np.array([[
            [10,16,16],
            [20,25,22],
            [17,25,23],
        ]])
        self.assertTrue(np.array_equal(got, want))

    def test_convole2d_padding(self):
        x = np.array([[
            [1,2,3,4,5],
            [6,5,4,3,2],
            [2,3,4,5,6],
            [7,6,5,4,3],
            [3,4,5,6,7],
        ]])
        kernel = np.array([[
            [0,1,0],
            [1,2,1],
            [0,1,0],
        ]])
        got = utils.convolve2d(x, kernel, padding=1, stride=2)
        want = np.array([[
            [10,16,16],
            [20,25,22],
            [17,25,23],
        ]])
        self.assertTrue(np.array_equal(got, want))

    def test_convolve2d_against_scipy(self):
        x = np.arange(25).reshape(1, 5, 5) + 1
        kernel = np.array([[[1,0],[0,1]]])
        got = utils.convolve2d(x, kernel)
        want = signal.correlate2d(x[0], kernel[0], mode='valid')
        self.assertTrue(np.allclose(got[0], want))

    def test_full_convolve2d_against_scipy(self):
        x = np.arange(9).reshape(3,3) + 1
        kernel = np.array([[1,0],[0,1]])
        got = utils.full_convolve2D(x, kernel)
        want = signal.convolve2d(x, kernel, mode='full')
        self.assertTrue(np.array_equal(got[0], want))

    def test_convolve2d_multiple_channels_against_scipy(self):
        np.random.seed(1)
        x = np.random.rand(2, 5, 5)
        kernel = np.random.rand(2, 3, 3)
        got = utils.convolve2d(x, kernel)
        want = np.zeros((3, 3))
        for i in range(2):
            want += signal.correlate2d(x[i], kernel[i], mode='valid')
        self.assertTrue(np.allclose(got, want))

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

    def test_convolve2d_transpose(self):
        x = np.array([[
            [1,2,1],
            [2,1,2],
            [1,1,2]
        ]])
        kernel = np.array([[
            [55,52],
            [57,50]
        ]])
        got = utils.convolve2d_transpose(x, kernel)
        want = np.array([[
            [55, 162, 159, 52],
            [167, 323, 319, 154],
            [169, 264, 326, 204],
            [57, 107, 164, 100]
        ]])
        self.assertTrue(np.array_equal(got, want))

    def test_convolve2d_transpose_with_channels(self):
        np.random.seed(1)
        x = np.array([[
            [1,2,1],
            [2,1,2],
            [1,1,2]
        ]])
        kernel = np.array([
            [
                [55,52],
                [57,50]
            ],
            [
                [-55,-52],
                [-57,-50]
            ]
        ])
        got = utils.convolve2d_transpose(x, kernel)
        want = np.array([[
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,0],
        ]])
        self.assertTrue(np.array_equal(got, want))


    def test_get_conv_positions(self):
        np.random.seed(0)
        stride=1
        padding=0
        xc, xh, xw = 1, 5, 5
        kc, kh, kw = 1, 2, 2
        x = np.arange(xh*xw).reshape(xc,xh,xw) * -1
        kernel = np.random.rand(kh*kw).reshape(kc,kh,kw)

        rows, cols = utils.get_convolution_positions(
            x.shape, kernel.shape, stride, padding)
        got = x[:, rows, cols]
        want = np.array([[
            [  0,  -1,  -5,  -6],
            [ -1,  -2,  -6,  -7],
            [ -2,  -3,  -7,  -8],
            [ -3,  -4,  -8,  -9],
            [ -5,  -6, -10, -11],
            [ -6,  -7, -11, -12],
            [ -7,  -8, -12, -13],
            [ -8,  -9, -13, -14],
            [-10, -11, -15, -16],
            [-11, -12, -16, -17],
            [-12, -13, -17, -18],
            [-13, -14, -18, -19],
            [-15, -16, -20, -21],
            [-16, -17, -21, -22],
            [-17, -18, -22, -23],
            [-18, -19, -23, -24]
        ]])
        self.assertTrue(np.array_equal(got, want))

    def test_get_conv_positions_with_strides(self):
        np.random.seed(0)
        stride=2
        padding=0
        xc, xh, xw = 1, 5, 5
        kc, kh, kw = 1, 2, 2
        x = np.arange(xh*xw).reshape(xc,xh,xw) * -1
        kernel = np.random.rand(kh*kw).reshape(kc,kh,kw)

        rows, cols = utils.get_convolution_positions(
            x.shape, kernel.shape, stride, padding)
        got = x[:, rows, cols]
        want = np.array([[
            [  0,  -1,  -5,  -6],
            [ -2,  -3,  -7,  -8],
            [-10, -11, -15, -16],
            [-12, -13, -17, -18]
        ]])
        self.assertTrue(np.array_equal(got, want))

    def test_vectorize_input_for_conv(self):
        np.random.seed(0)
        x = np.arange(5*5).reshape(1, 5, 5) + 1
        kernel = (1, 3, 3)
        got = utils.vectorize_input_for_convolution(x, kernel, stride=2, padding=1)
        want = np.array([[
            [ 0,  0,  0,  0,  1,  2,  0,  6,  7],
            [ 0,  0,  0,  2,  3,  4,  7,  8,  9],
            [ 0,  0,  0,  4,  5,  0,  9, 10,  0],
            [ 0,  6,  7,  0, 11, 12,  0, 16, 17],
            [ 7,  8,  9, 12, 13, 14, 17, 18, 19],
            [ 9, 10,  0, 14, 15,  0, 19, 20,  0],
            [ 0, 16, 17,  0, 21, 22,  0,  0,  0],
            [17, 18, 19, 22, 23, 24,  0,  0,  0],
            [19, 20,  0, 24, 25,  0,  0,  0,  0],
        ]])
        self.assertTrue(np.array_equal(got, want))


    @unittest.skip("Test against torch implementation")
    def test_convolve2d_transpose_against_torch(self):
        stride = 0
        padding = 0
        np.random.seed(1)
        x = np.random.rand(28, 28)
        kernel = np.random.rand(3,3)
        print()
        print("x shape = ", x.shape)
        print("kernel shape = ", kernel.shape)
        # y = np.random.rand(3, 3)
        for stride in range(1, x.shape[0]):
            y = utils.convolve2d(x, kernel, stride, padding)
            got = utils.convolve2d_transpose(y, kernel, stride, padding)
            got = np.round(got, 2)

            y1 = torch.Tensor(y.copy().reshape((1,1) + y.shape))
            k1 = torch.Tensor(kernel.copy().reshape((1,1) + kernel.shape))
            want = torch.nn.functional.conv_transpose2d(y1, k1, stride=stride, padding=padding).numpy()

            print()
            print("stride = ", stride)
            print("y shape = ", y.shape)
            print("got shape = ", got.shape)
            print("want shape = ", want.shape[2:])
            if got.shape != x.shape:            
                print("Not matching shape!")
            
        # print(got)
        # print(want)
        # self.assertTrue(np.allclose(got, want))

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

    def test_vectorize_kernel(self):
        np.random.seed(1)
        x = np.random.rand(3,3,3)
        k = np.arange(3*2*2).reshape(3,2,2) + 1
        got = utils.vectorize_kernel(x.shape, k)

        want = np.array([
            [
                [ 1.,  2.,  0.,  3.,  4.,  0.,  0.,  0.,  0.],
                [ 0.,  1.,  2.,  0.,  3.,  4.,  0.,  0.,  0.],
                [ 0.,  0.,  0.,  1.,  2.,  0.,  3.,  4.,  0.],
                [ 0.,  0.,  0.,  0.,  1.,  2.,  0.,  3.,  4.],
            ],
            [
                [ 5.,  6.,  0.,  7.,  8.,  0.,  0.,  0.,  0.],
                [ 0.,  5.,  6.,  0.,  7.,  8.,  0.,  0.,  0.],
                [ 0.,  0.,  0.,  5.,  6.,  0.,  7.,  8.,  0.],
                [ 0.,  0.,  0.,  0.,  5.,  6.,  0.,  7.,  8.],
            ],
            [
                [ 9., 10.,  0., 11., 12.,  0.,  0.,  0.,  0.],
                [ 0.,  9., 10.,  0., 11., 12.,  0.,  0.,  0.],
                [ 0.,  0.,  0.,  9., 10.,  0., 11., 12.,  0.],
                [ 0.,  0.,  0.,  0.,  9., 10.,  0., 11., 12.],
            ]
        ])
        self.assertTrue(np.array_equal(got, want))
    
    def test_convolve_vectorized(self):
        np.random.seed(1)
        x = np.arange(5*5).reshape(1,5,5) + 1
        k = np.identity(3).reshape(1,3,3)
        grad = np.ones((1, 5-3+1, 5-3+1)) / (3*3)
        want = _convolve2d_iterative(x, k)
        want_dx, want_dk = _convolve2d_gradient_iterative(x, k, grad)
        got = _convolve2d_vectorized(x, k)
        got_dx, got_dk = _convolve2d_gradient_vectorized(x, k, grad)

        self.assertEqual(want.shape, got.shape)
        self.assertEqual(want_dx.shape, got_dx.shape)
        self.assertEqual(want_dk.shape, got_dk.shape)
        self.assertTrue(np.allclose(want, got))
        self.assertTrue(np.allclose(want_dx, got_dx))
        self.assertTrue(np.allclose(want_dk, got_dk))

    def test_convolve_vectorized_with_stride_padding(self):
        np.random.seed(1)
        stride=2
        padding=2
        x = np.arange(8*8, dtype=np.float64).reshape(1,8,8) + 1
        k = np.random.rand(3,3).reshape(1,3,3)
        want = _convolve2d_iterative(x, k, stride, padding)
        grad = np.ones(want.shape) / want.size
        want_dx, want_dk = _convolve2d_gradient_iterative(x, k, grad, stride, padding)
         
        got = _convolve2d_vectorized(x, k, stride, padding)
        got_dx, got_dk = _convolve2d_gradient_vectorized(x, k, grad, stride, padding)

        self.assertEqual(want.shape, got.shape)
        self.assertEqual(want_dx.shape, got_dx.shape)
        self.assertEqual(want_dk.shape, got_dk.shape)
        self.assertTrue(np.allclose(want, got))
        self.assertTrue(np.allclose(want_dx, got_dx))
        self.assertTrue(np.allclose(want_dk, got_dk))

    def test_max_pool(self):
        np.random.seed(1)
        x = np.arange(5*5)+1
        np.random.shuffle(x)
        x = np.reshape(x, (1, 5, 5))
        got = utils.max_pool2d(x, 2)
        want = np.array([[
            [19, 20, 20, 22],
            [21, 20, 23, 23],
            [21, 16, 25, 25],
            [17, 16, 25, 25],
        ]])
        self.assertTrue(np.array_equal(got, want))

    

if __name__ == '__main__':
    unittest.main()

