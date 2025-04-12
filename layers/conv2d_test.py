import unittest
from devices import xp
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
        layer.W = [ag.Tensor(xp.array([[
            [0,1,0],
            [1,2,1],
            [0,1,0]
        ]]), requires_grad=True)]
        layer.b = [ag.Tensor(xp.array([[0]]), requires_grad=True)]
        xp.random.seed(1)
        X = ag.Tensor(xp.array([[
            [1,2,3,4,5],
            [6,5,4,3,2],
            [2,3,4,5,6],
            [7,6,5,4,3],
            [3,4,5,6,7]
        ]]), requires_grad=True)

        pred = layer.forward(X)
        want_pred = xp.array([[
            [10,16,16],
            [20,25,22],
            [17,25,23],
        ]])
        self.assertEquals(pred.shape, (1,3,3))
        self.assertTrue(xp.array_equal(pred.value(), want_pred))


        dEdY = xp.array([
            [
                [0.1,0.2,0.3],
                [0.4,0.5,0.4],
                [0.3,0.2,0.1],
            ]
        ])
        pred.backward(ag.Tensor(dEdY))
        dEdX = X.grad.value()
        dEdW = layer.W[0].grad.value()
        dEdb = layer.b[0].grad.value()
        
        want_dEdW = xp.array([[
            [5.3,8.6,6.1],
            [6.5,10,7.1],
            [6.5,8.5,5.5]
        ]])
        want_dEdb = xp.array([
            [xp.sum(dEdY)]
        ])
        want_dEdX = xp.array([[
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
        self.assertTrue(xp.allclose(dEdW, want_dEdW))

        # print('db')
        # print(layer.db)
        # print(want_dEdb)
        self.assertEquals(dEdb.shape, want_dEdb.shape)
        self.assertTrue(xp.array_equal(dEdb, want_dEdb))

        # print('dEdX')
        # print(dEdX)
        # print(want_dEdX)
        self.assertEquals(dEdX.shape, want_dEdX.shape)
        self.assertTrue(xp.allclose(dEdX, want_dEdX))

if __name__ == '__main__':
    unittest.main()

