import unittest
import numpy as np
import utils
from .conv2d import *

class TestConvs(unittest.TestCase):

    def setUp(self):
        layer = Conv2DLayer(
            input_shape=(2, 5, 5),
            num_kernels=2,
            kernel_size=3
        )
        layer.W = np.array([
            [
                [[1,0,0],[0,1,0],[0,0,1]],
                [[0,1,0],[1,0,0],[0,0,1]],
            ],
            [
                [[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]],
                [[0.25,0.25,0.25],[0.25,0.25,0.25],[0.25,0.25,0.25]],
            ]
        ])
        layer.b = np.array([
            [[0,0,0],[0,0,0],[0,0,0]],
            [[1,1,1],[1,1,1],[1,1,1]],
        ])
        self.conv2dLayer = layer

    def test_flatten(self):
        flatten_layer = FlattenLayer()
        context = {}
        X = np.random.rand(2, 1, 3, 3)  # Example input
        output = flatten_layer.forward(context, X)
        self.assertEqual(output.shape, (2, 9, 1))
        dE = np.random.rand(2, 9, 1)
        grads = flatten_layer.backward(context, dE)
        self.assertEqual(grads.shape, X.shape)

    def test_conv2d_forward_with_reference(self):
        height, width, channels = (4, 5, 3)
        num_kernels = 4
        kernel_size = 3
        input_shape = (height, width, channels)
        np.random.seed(1)
        layer = Conv2DLayer(input_shape, num_kernels=num_kernels, kernel_size=kernel_size)
        np.random.seed(1)
        reference = Conv2DLayerReference(input_shape, num_kernels=num_kernels, kernel_size=kernel_size)
        assert(np.array_equal(layer.W, reference.W))
        assert(np.array_equal(layer.b.reshape(-1), reference.b.reshape(-1)))

        np.random.seed(3)
        batches = 2
        X = np.random.rand(batches*height*width*channels).reshape(batches, height, width, channels)

        context = {"learning_rate": 0.01}
        Y1 = layer.forward(context, X)
        Y2 = reference.forward(context, X)
        self.assertTrue(np.allclose(Y1, Y2))

        np.random.seed(4)
        dEdY = np.random.rand(Y1.size).reshape(Y1.shape)
        dEdX1 = layer.backward(context, dEdY)
        dEdX2 = reference.backward(context, dEdY)
        self.assertTrue(np.allclose(dEdX1, dEdX2))
        # assert(np.array_equal(Y1, Y2))

    def test_conv2d_forward(self):
        layer = self.conv2dLayer

        got_shape = layer.get_output_shape()
        want_shape = (2, 3, 3)
        self.assertEquals(got_shape, want_shape)

        context = {}
        X = np.arange(2*2*5*5).reshape(2,2,5,5) + 1 # Example input
        X = X.astype(np.float64)
        Y = layer.forward(context, X)

        kernel1_c1 = layer.W[0, 0]
        kernel1_c2 = layer.W[0, 1]
        kernel2_c1 = layer.W[1, 0]
        kernel2_c2 = layer.W[1, 1]
        bias1 = layer.b[0]
        bias2 = layer.b[1]
        z11  = utils.convolve2d(X[0,0], kernel1_c1)
        z11 += utils.convolve2d(X[0,1], kernel1_c2)
        z11 += bias1
        z12  = utils.convolve2d(X[0,0], kernel2_c1)
        z12 += utils.convolve2d(X[0,1], kernel2_c2)
        z12 += bias2
        z21  = utils.convolve2d(X[1,0], kernel1_c1)
        z21 += utils.convolve2d(X[1,1], kernel1_c2)
        z21 += bias1
        z22  = utils.convolve2d(X[1,0], kernel2_c1)
        z22 += utils.convolve2d(X[1,1], kernel2_c2)
        z22 += bias2
        want = np.array([
            [z11, z12],
            [z21, z22]
        ])

        # print("\n")
        # for batch in range(2):
        #     print(Y[batch, :, :, :])
        #     print(want[batch, :, :, :])
        self.assertTrue(np.array_equal(Y, want))

    def test_conv2d_backward(self):
        layer = self.conv2dLayer

        context = {}
        X = np.arange(2*2*5*5).reshape(2,2,5,5) + 1 # Example input
        Y = layer.forward(context, X)
        dEdY = np.ones(Y.shape) * 0.25
        context["learning_rate"] = 0.01
        dEdX = layer.backward(context, dEdY)

        self.assertTrue(dEdX.shape, X.shape)

    def test_conv2d_forward_with_stride_and_padding(self):
        layer = Conv2DLayer(
            input_shape=(1,5,5),
            num_kernels=1,
            kernel_size=3,
            stride=2,
            padding=1
        )
        layer.W = np.array([[[
            [0,1,0],
            [1,2,1],
            [0,1,0]
        ]]])
        layer.b = np.array([[[0]]])
        np.random.seed(1)
        X = np.array([[[
            [1,2,3,4,5],
            [6,5,4,3,2],
            [2,3,4,5,6],
            [7,6,5,4,3],
            [3,4,5,6,7]
        ]]])

        context = {
            "learning_rate": 0.01
        }
        pred = layer.forward(context, X)
        want_pred = np.array([[[
            [10,16,16],
            [20,25,22],
            [17,25,23],
        ]]])
        self.assertEquals(pred.shape, (1,1,3,3))
        self.assertTrue(np.array_equal(pred, want_pred))

        dEdY = np.array([[
            [
                [0.1,0.2,0.3],
                [0.4,0.5,0.4],
                [0.3,0.2,0.1],
            ]
        ]])
        self.assertEquals(dEdY.shape, (1,1,3,3))
        dEdX = layer.backward(context, dEdY)
        self.assertEquals(dEdX.shape, X.shape)

        
        want_dEdW = np.array([[[
            [5.3,8.6,6.1],
            [6.5,10,7.1],
            [6.5,8.5,5.5]
        ]]])
        want_dEdb = np.array([[
            [np.sum(dEdY)]
        ]])
        want_dEdX = np.array([[[
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.5, 0.0, 0.7, 0.0, 0.7],
            [0.8, 0.9, 1.0, 0.9, 0.8],
            [0.7, 0.0, 0.7, 0.0, 0.5],
            [0.6, 0.5, 0.4, 0.3, 0.2]
        ]]])

        # print('dW')
        # print(layer.dW)
        # print(want_dEdW)
        self.assertEquals(layer.dW.shape, want_dEdW.shape)
        self.assertTrue(np.allclose(layer.dW, want_dEdW))

        # print('db')
        # print(layer.db)
        # print(want_dEdb)
        self.assertEquals(layer.db.shape, want_dEdb.shape)
        self.assertTrue(np.array_equal(layer.db, want_dEdb))

        # print('dEdX')
        # print(dEdX)
        # print(want_dEdX)
        self.assertEquals(dEdX.shape, want_dEdX.shape)
        self.assertTrue(np.allclose(dEdX, want_dEdX))

if __name__ == '__main__':
    unittest.main()

