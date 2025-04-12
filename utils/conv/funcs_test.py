import unittest
import numpy
from devices import xp
import utils
from .iterative import (
    _convolve2d_iterative,
    _convolve2d_gradient_iterative,
)
from .vectorized import (
    _convolve2d_vectorized,
    _convolve2d_gradient_vectorized
)
from scipy import signal
import torch

class TestConvFuncs(unittest.TestCase):

    def test_convolve2d_dilation(self):
        xp.random.seed(1)
        x = xp.round(xp.random.rand(1,5,5), 2)
        k = xp.round(xp.random.rand(1,3,3), 2)
        pred = utils.convolve2d(x, k, dilate=2)
        outGrad = xp.random.rand(*pred.shape)
        got = utils.convolve2d_gradient(x, k, outGrad, dilate=2)

    def test_convolve2d_simple(self):
        x = xp.array([[
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]])
        kernel = xp.array([[
            [0,1],
            [2,1]
        ]])
        got = utils.convolve2d(x, kernel)
        want = xp.array([[
            [15, 19],
            [27, 31]
        ]])
        self.assertTrue(xp.array_equal(got, want))

    def test_convole2d_stride(self):
        x = xp.array([[
            [0,0,0,0,0,0,0],
            [0,1,2,3,4,5,0],
            [0,6,5,4,3,2,0],
            [0,2,3,4,5,6,0],
            [0,7,6,5,4,3,0],
            [0,3,4,5,6,7,0],
            [0,0,0,0,0,0,0]
        ]])
        kernel = xp.array([[
            [0,1,0],
            [1,2,1],
            [0,1,0],
        ]])
        got = utils.convolve2d(x, kernel, stride=2)
        want = xp.array([[
            [10,16,16],
            [20,25,22],
            [17,25,23],
        ]])
        self.assertTrue(xp.array_equal(got, want))

    def test_convole2d_padding(self):
        x = xp.array([[
            [1,2,3,4,5],
            [6,5,4,3,2],
            [2,3,4,5,6],
            [7,6,5,4,3],
            [3,4,5,6,7],
        ]])
        kernel = xp.array([[
            [0,1,0],
            [1,2,1],
            [0,1,0],
        ]])
        got = utils.convolve2d(x, kernel, padding=1, stride=2)
        want = xp.array([[
            [10,16,16],
            [20,25,22],
            [17,25,23],
        ]])
        self.assertTrue(xp.array_equal(got, want))

    def test_convolve2d_against_scipy(self):
        x = numpy.arange(25).reshape(1, 5, 5) + 1
        kernel = numpy.array([[[1,0],[0,1]]])

        x_device = xp.array(x)
        kernel_device = xp.array(kernel)
        got_device = utils.convolve2d(x_device, kernel_device)
        want = signal.correlate2d(
            x[0],
            kernel[0],
            mode='valid')
        want_device = xp.array(want)
        self.assertTrue(xp.allclose(got_device[0], want_device))
    
    def test_convolve2d_multiple_channels_against_scipy(self):
        numpy.random.seed(1)
        x = numpy.random.rand(2, 5, 5)
        kernel = numpy.random.rand(2, 3, 3)

        got = utils.convolve2d(xp.array(x), xp.array(kernel))
        want = numpy.zeros((3, 3))
        for i in range(2):
            want += signal.correlate2d(
                x[i],
                kernel[i],
                mode='valid'
            )
        self.assertTrue(xp.allclose(got, xp.array(want)))

    def test_convolve2d_transpose(self):
        x = xp.array([[
            [1,2,1],
            [2,1,2],
            [1,1,2]
        ]])
        kernel = xp.array([[
            [55,52],
            [57,50]
        ]])
        got = utils.convolve2d_transpose(x, kernel)
        want = xp.array([[
            [55, 162, 159, 52],
            [167, 323, 319, 154],
            [169, 264, 326, 204],
            [57, 107, 164, 100]
        ]])
        self.assertTrue(xp.array_equal(got, want))

    def test_convolve2d_transpose_with_channels(self):
        xp.random.seed(1)
        x = xp.array([[
            [1,2,1],
            [2,1,2],
            [1,1,2]
        ]])
        kernel = xp.array([
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
        want = xp.array([[
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,0],
        ]])
        self.assertTrue(xp.array_equal(got, want))

    @unittest.skip("Test against torch implementation")
    def test_convolve2d_transpose_against_torch(self):
        stride = 0
        padding = 0
        xp.random.seed(1)
        x = xp.random.rand(1, 6, 6)
        kernel = xp.random.rand(1, 3,3)
        print()
        print("x shape = ", x.shape)
        print("kernel shape = ", kernel.shape)
        # y = xp.random.rand(3, 3)
        for stride in range(1, x.shape[1]):
            y = utils.convolve2d(x, kernel, stride, padding)
            got = utils.convolve2d_transpose(y, kernel, stride, padding)
            # got = xp.round(got, 2)

            y1 = torch.Tensor(y.copy().reshape((1,) + y.shape))
            k1 = torch.Tensor(kernel.copy().reshape((1,) + kernel.shape))
            want = torch.nn.functional.conv_transpose2d(y1, k1, stride=stride, padding=padding).numpy()
            want = want[0]

            if got.shape != x.shape:
                continue
            if got.shape != want.shape:
                continue

            print(stride)
            print(got)
            print(want)
            self.assertTrue(xp.allclose(got, want))


            # print()
            # print("stride = ", stride)
            # print("y shape = ", y.shape)
            # print("got shape = ", got.shape)
            # print("want shape = ", want.shape[2:])
            # if got.shape == x.shape:
            #     print("Not matching shape!")
            
        # print(got)
        # print(want)
        # self.assertTrue(xp.allclose(got, want)) 
    
    def test_convolve_vectorized(self):
        xp.random.seed(1)
        x = xp.arange(5*5, dtype=xp.float64).reshape(1,5,5) + 1
        k = xp.identity(3).reshape(1,3,3)
        grad = xp.ones((1, 5-3+1, 5-3+1)) / (3*3)
        want = _convolve2d_iterative(x, k)
        want_dx, want_dk = _convolve2d_gradient_iterative(x, k, grad)
        got = _convolve2d_vectorized(x, k)
        got_dx, got_dk = _convolve2d_gradient_vectorized(x, k, grad)

        self.assertEqual(want.shape, got.shape)
        self.assertEqual(want_dx.shape, got_dx.shape)
        self.assertEqual(want_dk.shape, got_dk.shape)
        self.assertTrue(xp.allclose(want, got))
        self.assertTrue(xp.allclose(want_dx, got_dx))
        self.assertTrue(xp.allclose(want_dk, got_dk))

    def test_convolve_vectorized_with_stride_padding(self):
        xp.random.seed(1)
        stride=2
        padding=2
        x = xp.arange(8*8, dtype=xp.float64).reshape(1,8,8) + 1
        k = xp.random.rand(3,3).reshape(1,3,3)
        want = _convolve2d_iterative(x, k, stride, padding)
        grad = xp.ones(want.shape) / want.size
        want_dx, want_dk = _convolve2d_gradient_iterative(x, k, grad, stride, padding)
         
        got = _convolve2d_vectorized(x, k, stride, padding)
        got_dx, got_dk = _convolve2d_gradient_vectorized(x, k, grad, stride, padding)

        self.assertEqual(want.shape, got.shape)
        self.assertEqual(want_dx.shape, got_dx.shape)
        self.assertEqual(want_dk.shape, got_dk.shape)
        self.assertTrue(xp.allclose(want, got))
        self.assertTrue(xp.allclose(want_dx, got_dx))
        self.assertTrue(xp.allclose(want_dk, got_dk))

    def test_max_pool(self):
        xp.random.seed(1)

        x = xp.array([[
            [15, 14, 18,  4, 22],
            [11, 19, 20,  5,  3],
            [21,  7,  8, 23,  2],
            [17,  1, 16, 25, 24],
            [10,  9, 13, 12,  6]
        ]])
        # x = xp.arange(5*5)+1
        # xp.random.shuffle(x)
        # x = xp.reshape(x, (1, 5, 5))
        got = utils.max_pool2d(x, 2)
        want = xp.array([[
            [19, 20, 20, 22],
            [21, 20, 23, 23],
            [21, 16, 25, 25],
            [17, 16, 25, 25],
        ]])
        self.assertTrue(xp.array_equal(got, want))

    

if __name__ == '__main__':
    unittest.main()

