import unittest
import numpy as np
import utils
from .conv2d import *

class TestConvs(unittest.TestCase):

    def test_conv2d_forward_with_stride_and_padding(self):
        layer = Conv2d(
            input_shape=(1,5,5),
            num_kernels=1,
            kernel_size=3,
            stride=2,
            padding=1
        )
        layer.W = [ag.Tensor(np.array([[
            [0,1,0],
            [1,2,1],
            [0,1,0]
        ]]), requires_grad=True)]
        layer.b = [ag.Tensor(np.array([[0]]), requires_grad=True)]
        np.random.seed(1)
        X = ag.Tensor(np.array([[
            [1,2,3,4,5],
            [6,5,4,3,2],
            [2,3,4,5,6],
            [7,6,5,4,3],
            [3,4,5,6,7]
        ]]), requires_grad=True)

        pred = layer.forward(X)
        want_pred = np.array([[
            [10,16,16],
            [20,25,22],
            [17,25,23],
        ]])
        self.assertEquals(pred.shape, (1,3,3))
        self.assertTrue(np.array_equal(pred.value(), want_pred))


        dEdY = np.array([
            [
                [0.1,0.2,0.3],
                [0.4,0.5,0.4],
                [0.3,0.2,0.1],
            ]
        ])
        pred.backward(dEdY)
        dEdX = X.grad
        dEdW = layer.W[0].grad
        dEdb = layer.b[0].grad

        
        want_dEdW = np.array([[
            [5.3,8.6,6.1],
            [6.5,10,7.1],
            [6.5,8.5,5.5]
        ]])
        want_dEdb = np.array([
            [np.sum(dEdY)]
        ])
        want_dEdX = np.array([[
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.5, 0.0, 0.7, 0.0, 0.7],
            [0.8, 0.9, 1.0, 0.9, 0.8],
            [0.7, 0.0, 0.7, 0.0, 0.5],
            [0.6, 0.5, 0.4, 0.3, 0.2]
        ]])

        # print('dW')
        # print(layer.dW)
        # print(want_dEdW)
        self.assertEquals(dEdW.shape, want_dEdW.shape)
        self.assertTrue(np.allclose(dEdW, want_dEdW))

        # print('db')
        # print(layer.db)
        # print(want_dEdb)
        self.assertEquals(dEdb.shape, want_dEdb.shape)
        self.assertTrue(np.array_equal(dEdb, want_dEdb))

        # print('dEdX')
        # print(dEdX)
        # print(want_dEdX)
        self.assertEquals(dEdX.shape, want_dEdX.shape)
        self.assertTrue(np.allclose(dEdX, want_dEdX))

if __name__ == '__main__':
    unittest.main()

