import unittest
import numpy as np
import utils
from .pool2d import *
from .conv2d import *

class TestConvs(unittest.TestCase):

    def test_pool2d_max_forward(self):
        layer = PoolLayer((1,5,5), mode=PoolLayer.MAX, kernel_size=3)
        x = np.array([
            [1,2,3,4,5],
            [6,5,4,3,2],
            [2,3,4,5,6],
            [0,1,2,3,4],
            [0,9,2,6,7]
        ]).reshape(1,1,5,5)
        context = {}
        got = layer.forward(context, x)
        want = np.array([
            [6,5,6],
            [6,5,6],
            [9,9,7]
        ]).reshape(1,1,3,3)
        self.assertEquals(got.shape,(1,1,3,3))
        self.assertTrue(np.allclose(got, want))

    def test_pool2d_average_forward(self):
        layer = PoolLayer((1,5,5), mode=PoolLayer.AVERAGE, kernel_size=3)
        x = np.array([
            [1,2,3,4,5],
            [6,5,4,3,2],
            [2,3,4,5,6],
            [0,1,2,3,4],
            [0,9,2,6,7]
        ]).reshape(1,1,5,5)
        context = {}
        got = layer.forward(context, x)
        want = np.array([
            [30, 33, 36],
            [27, 30, 33],
            [23, 35, 39]
        ], dtype=np.float64).reshape(1,1,3,3) / 9.0
        self.assertEquals(got.shape,(1,1,3,3))
        self.assertTrue(np.allclose(got, want))
    
    def test_pool2d_max_backward(self):
        layer = PoolLayer((1,3,3), mode=PoolLayer.MAX, kernel_size=2)
        x = np.array([
            [1,2,3],
            [6,4,5],
            [2,1,3]
        ]).reshape(1,1,3,3)
        context = {}
        dEdY = np.array([
            [0.1, 0.2],
            [0.3, 0.4]
        ]).reshape(1,1,2,2)
        layer.forward(context, x)
        got = layer.backward(context, dEdY)

        want = np.array([
            [0,0,0],
            [0.4, 0, 0.6],
            [0,0,0]
        ]).reshape(1,1,3,3)
        self.assertEquals(got.shape,(1,1,3,3))
        self.assertTrue(np.allclose(got, want))

    def test_pool2d_average_backward(self):
        layer = PoolLayer((1,3,3), mode=PoolLayer.AVERAGE, kernel_size=2)
        x = np.array([
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]).reshape(1,1,3,3)
        context = {}
        dEdY = np.array([
            [0.1, 0.2],
            [0.3, 0.4]
        ]).reshape(1,1,2,2)
        layer.forward(context, x)
        got = layer.backward(context, dEdY)

        want = np.array([
            [0.025, 0.075, 0.05],
            [0.1, 0.25, 0.15],
            [0.075, 0.175, 0.1]
        ]).reshape(1,1,3,3)
        self.assertEquals(got.shape,(1,1,3,3))
        self.assertTrue(np.allclose(got, want))

    def test_pool2d_max_full(self):
        layer = PoolLayer((1,5,5), mode=PoolLayer.MAX, kernel_size=3, stride=2)
        x = np.array([
            [1,  2,  3,  4,  5],
            [6,  7,  8,  9,  10],
            [11, 12, 13, 12, 11],
            [13, 14, 15, 16, 17],
            [18, 17, 16, 17, 19],

        ]).reshape(1,1,5,5)
        context = {}
        got_pred = layer.forward(context, x)
        got_grad = layer.backward(context, np.ones(got_pred.shape)*0.1)

        want_pred = np.array([
            [13, 13],
            [18, 19],
        ]).reshape(1,1,2,2)
        want_grad = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.1],
        ]).reshape(1,1,5,5)

        self.assertEquals(got_pred.shape, want_pred.shape)
        self.assertEquals(got_grad.shape, want_grad.shape)
        self.assertTrue(np.allclose(got_pred, want_pred))
        self.assertTrue(np.allclose(got_grad, want_grad))

    def test_pool2d_average_full(self):
        layer = PoolLayer((1,5,5), mode=PoolLayer.AVERAGE, kernel_size=3, stride=2)
        x = np.array([
            [1,1,1,1,1],
            [2,2,2,2,2],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [3,3,3,3,3],
        ]).reshape(1,1,5,5)
        context = {}
        got_pred = layer.forward(context, x)
        got_grad = layer.backward(context, np.ones(got_pred.shape)*0.1)

        want_pred = np.array([
            [12/9, 12/9],
            [15/9, 15/9],
        ]).reshape(1,1,2,2)
        want_grad = np.array([
            [1/9, 1/9, 2/9, 1/9, 1/9],
            [1/9, 1/9, 2/9, 1/9, 1/9],
            [2/9, 2/9, 4/9, 2/9, 2/9],
            [1/9, 1/9, 2/9, 1/9, 1/9],
            [1/9, 1/9, 2/9, 1/9, 1/9],
        ]).reshape(1,1,5,5) * 0.1

        self.assertEquals(got_pred.shape, want_pred.shape)
        self.assertEquals(got_grad.shape, want_grad.shape)
        self.assertTrue(np.allclose(got_pred, want_pred))
        self.assertTrue(np.allclose(got_grad, want_grad))

if __name__ == '__main__':
    unittest.main()

