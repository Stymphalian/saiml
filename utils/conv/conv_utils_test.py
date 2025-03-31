import unittest
import numpy as np
import utils.conv as utils
# import utils

class TestConvUtils(unittest.TestCase):

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

    def test_vectorize_kernel_maxpool(self):
        np.random.seed(1)
        x  = np.arange(2*5*3)
        np.random.shuffle(x)
        x = x.reshape(2,5,3)
        k = np.arange(2*2*2).reshape(2,2,2) + 1
        got = utils.vectorize_kernel_maxpool(x, k.shape)

        identity = np.identity(2*5*3)
        flat_x = x.reshape(-1)
        want = np.array([
            identity[np.where(flat_x == 21)[0][0]], # 21
            identity[np.where(flat_x == 21)[0][0]], # 21
            identity[np.where(flat_x == 26)[0][0]], # 26
            identity[np.where(flat_x == 24)[0][0]], # 24
            identity[np.where(flat_x == 26)[0][0]], # 26
            identity[np.where(flat_x == 24)[0][0]], # 24
            identity[np.where(flat_x == 25)[0][0]], # 25
            identity[np.where(flat_x == 25)[0][0]], # 25
            identity[np.where(flat_x == 27)[0][0]], # 27
            identity[np.where(flat_x == 16)[0][0]], # 16
            identity[np.where(flat_x == 27)[0][0]], # 27
            identity[np.where(flat_x == 29)[0][0]], # 29
            identity[np.where(flat_x == 28)[0][0]], # 28
            identity[np.where(flat_x == 29)[0][0]], # 29
            identity[np.where(flat_x == 28)[0][0]], # 28
            identity[np.where(flat_x == 11)[0][0]], # 11
        ])
        self.assertEqual(got.shape, want.shape)
        self.assertTrue(np.array_equal(got, want))

    

if __name__ == '__main__':
    unittest.main()

