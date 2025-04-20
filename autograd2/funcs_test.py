import unittest
import math
import autograd2 as ag
import base_gradient_test
from devices import xp

class OperatorsTest(base_gradient_test.NumericalGradientTest):

    def test_sigmoid(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.normal(1000), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.sigmoid(x)
            # return ag.Tensor(1) / (1.0 + ag.exp(-x))
        self.numeric_check(forward, x)

    def test_cross_entropy_loss(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.rand(10), requires_grad=True)
        y = ag.Tensor(xp.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=xp.float64))
        def forward(params):
            self.unravel_params(params, x)
            return ag.cross_entropy_loss(x, y)
        self.numeric_check(forward, x, threshold=1e-4)

    def test_softplus(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.rand(10), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.softplus(x)
        self.numeric_check(forward, x)

    def test_softmax_gradient(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.rand(10), requires_grad=True)
        y = ag.softmax(x)
        y.backward()

    def test_softmax_forward(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.rand(2, 10), requires_grad=True)
        y = ag.softmax(x, axis=1)
        self.assertTrue(xp.sum(y.value()[0]), xp.array(1.0))
        self.assertTrue(xp.sum(y.value()[1]), xp.array(1.0))

    def test_softmax(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.rand(10), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.softmax(x)
        self.numeric_check(forward, x, threshold=5.0)
        # self.numeric_check(forward, x)

    def test_batch_softmax(self):
        xp.random.seed(1)
        x1 = xp.arange(2*5*2).reshape(2,5,2)
        x = ag.Tensor(x1, requires_grad=True)

        got = ag.softmax(x, axis=(1,2))
        want1 = ag.softmax(x[0])
        want2 = ag.softmax(x[1])
        self.assertTrue(xp.allclose(got[0].value(), want1.value()))
        self.assertTrue(xp.allclose(got[1].value(), want2.value()))

    def test_log_softmax(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.rand(2,5,2), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.log_softmax(x, axis=(1,2))
        self.numeric_check(forward, x)
    
    def test_sequence(self):
        # xp.random.seed(1)
        
        x_train = xp.array([
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        ], dtype=xp.float64)
        y_train = xp.array([1,0,0,0,0, 0,0,0,0,0])
        y = ag.Tensor(y_train.reshape(-1, 1), requires_grad=True)
        x = ag.Tensor(x_train.reshape(-1, 1), requires_grad=True)
        w = ag.Tensor(xp.random.rand(10, 784), requires_grad=True)
        b = ag.Tensor(xp.random.rand(10, 1), requires_grad=True)
        def forward(params):
            self.unravel_params(params, w, b)
            z1 = ag.matmul(w, x) + b
            z2 = ag.sigmoid(z1)
            z3 = ag.softmax(z2)
            z4 = ag.cross_entropy_loss(z3, y)
            return z4
        self.numeric_check(forward, w, b)
    
    def test_relu(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.rand(10), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.relu(x)
        self.numeric_check(forward, x)
    
    def test_convolve2d(self):
        x = ag.Tensor(xp.arange(9, dtype=xp.float64).reshape(1,3,3) + 1.0, requires_grad=True)
        k = ag.Tensor(xp.arange(4, dtype=xp.float64).reshape(1,2,2) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x, k)
            return ag.convolve2d(x, k)
        self.numeric_check(forward, x, k)

    def test_convolve2d_with_stride(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.rand(1,5,5), requires_grad=True)
        k = ag.Tensor(xp.random.rand(1,3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x, k)
            return ag.convolve2d(x, k, stride=2)
        self.numeric_check(forward, x, k)
    
    def test_convolve2d_with_padding(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.rand(1,5,5), requires_grad=True)
        k = ag.Tensor(xp.random.rand(1,3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x, k)
            return ag.convolve2d(x, k, padding=2)
        self.numeric_check(forward, x, k)

    @unittest.skip("Dilation is no longer supported")
    def test_convolve2d_with_dilation(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.rand(1,5,5), requires_grad=True)
        k = ag.Tensor(xp.random.rand(1,3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x, k)
            return ag.convolve2d(x, k, dilate=2)
        self.numeric_check(forward, x, k)

    def test_convolve2d_with_stride_padding_dilation(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.rand(1,5,5), requires_grad=True)
        k = ag.Tensor(xp.random.rand(1,3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x, k)
            return ag.convolve2d(x, k, stride=2, padding=2)
        self.numeric_check(forward, x, k)

    def test_convolve2d_transpose(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.rand(1,5,5), requires_grad=True)
        k = ag.Tensor(xp.random.rand(1,3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x, k)
            return ag.convolve2d_transpose(x, k)
        self.numeric_check(forward, x, k)

    def test_convolve2d_transpose_with_stride(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.rand(1,5,5), requires_grad=True)
        k = ag.Tensor(xp.random.rand(1,3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x, k)
            return ag.convolve2d_transpose(x, k, stride=2)
        self.numeric_check(forward, x, k)
    
    def test_convolve2d_transpose_with_padding(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.rand(1,8,8), requires_grad=True)
        k = ag.Tensor(xp.random.rand(1,3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x, k)
            return ag.convolve2d_transpose(x, k, padding=1, outer_padding=1)
        self.numeric_check(forward, x, k)

    def test_variance(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.rand(10), requires_grad=True)
        got = ag.variance(x)
        want = xp.var(x.value())
        self.assertEquals(got.value(), want)

        def forward(params):
            self.unravel_params(params, x)
            return ag.variance(x)
        self.numeric_check(forward, x)

    def test_batch_matmul_with_1d(self):
        xp.random.seed(1)
        # a is 3d, b is 1d
        a = ag.Parameter(xp.arange(2*3*4, dtype=xp.float64).reshape(2,3,4) + 1.0)
        b = ag.Parameter(xp.arange(4, dtype=xp.float64)+1.0)
        got = ag.batch_matmul(a, b)
        self.assertEqual(got.shape, (2,3,1))
        want1 = xp.matmul(a.value()[0], b.value())
        want2 = xp.matmul(a.value()[1], b.value())
        self.assertTrue(xp.allclose(got.value()[0][:,0], want1))
        self.assertTrue(xp.allclose(got.value()[1][:,0], want2))

        got.backward()
        self.assertEqual(a.grad.shape, (2,3,4))
        self.assertEqual(b.grad.shape, (4,))

        # a is 1d, b is 3d
        a = ag.Parameter(xp.arange(3, dtype=xp.float64)+1.0)
        b = ag.Parameter(xp.arange(2*3*4, dtype=xp.float64).reshape(2,3,4) + 1.0)
        got = ag.batch_matmul(a, b)
        self.assertEqual(got.shape, (2,1,4))
        want1 = xp.matmul(a.value(), b.value()[0])
        want2 = xp.matmul(a.value(), b.value()[1])
        self.assertTrue(xp.allclose(got.value()[0][0,:], want1))
        self.assertTrue(xp.allclose(got.value()[1][0,:], want2))

        got.backward()
        self.assertEqual(b.grad.shape, (2,3,4))
        self.assertEqual(a.grad.shape, (3,))

    def test_batch_matmul_with_2d(self):
        # a is 3d, b is 2d
        a = ag.Parameter(xp.arange(2*3*4, dtype=xp.float64).reshape(2,3,4) + 1.0)
        b = ag.Parameter(xp.arange(4*3, dtype=xp.float64).reshape(4,3)+1.0)
        got = ag.batch_matmul(a, b)
        self.assertEqual(got.shape, (2,3,3))
        want1 = xp.matmul(a.value()[0], b.value())
        want2 = xp.matmul(a.value()[1], b.value())
        self.assertTrue(xp.allclose(got.value()[0], want1))
        self.assertTrue(xp.allclose(got.value()[1], want2))

        got.backward()
        self.assertEqual(a.grad.shape, (2,3,4))
        self.assertEqual(b.grad.shape, (4,3))

        # a is 2d, b is 3d
        a = ag.Parameter(xp.arange(4*3, dtype=xp.float64).reshape(4,3)+1.0)
        b = ag.Parameter(xp.arange(2*3*4, dtype=xp.float64).reshape(2,3,4) + 1.0)
        got = ag.batch_matmul(a, b)
        self.assertEqual(got.shape, (2,4,4))
        want1 = xp.matmul(a.value(), b.value()[0])
        want2 = xp.matmul(a.value(), b.value()[1])
        self.assertTrue(xp.allclose(got.value()[0], want1))
        self.assertTrue(xp.allclose(got.value()[1], want2))

        got.backward()
        self.assertEqual(a.grad.shape, (4,3))
        self.assertEqual(b.grad.shape, (2,3,4))

    @unittest.skip("Not supported anymore")
    def test_batch_matmul_2dimensions(self):
        # bi,bi->b
        x1 = ag.Tensor(xp.arange(2*3).reshape(2,3) + 1.0, requires_grad=True)
        x2 = ag.Tensor(xp.arange(2*3).reshape(2,3) + 5.0, requires_grad=True)
        got = ag.batch_matmul(x1, x2)
        got.backward()
        want1 = xp.matmul(x1[0].value(), x2[0].value())
        want2 = xp.matmul(x1[1].value(), x2[1].value())
        self.assertTrue(xp.allclose(got[0].value(), want1))
        self.assertTrue(xp.allclose(got[1].value(), want2))

        # bi,i->b
        x1 = ag.Tensor(xp.arange(2*3).reshape(2,3) + 1.0, requires_grad=True)
        x2 = ag.Tensor(xp.arange(3) + 5.0, requires_grad=True)
        got = ag.batch_matmul(x1, x2)
        got.backward()
        want1 = xp.matmul(x1[0].value(), x2.value())
        want2 = xp.matmul(x1[1].value(), x2.value())
        self.assertTrue(xp.allclose(got[0].value(), want1))
        self.assertTrue(xp.allclose(got[1].value(), want2))

        # i,bi->b
        x1 = ag.Tensor(xp.arange(3) + 1.0, requires_grad=True)
        x2 = ag.Tensor(xp.arange(2*3).reshape(2,3) + 5.0, requires_grad=True)
        got = ag.batch_matmul(x1, x2)
        got.backward()
        want1 = xp.matmul(x1.value(), x2[0].value())
        want2 = xp.matmul(x1.value(), x2[1].value())
        self.assertTrue(xp.allclose(got[0].value(), want1))
        self.assertTrue(xp.allclose(got[1].value(), want2))

    def test_batch_matmul_gradient(self):
        x1 = ag.Tensor(xp.arange(2*3*4).reshape(2,3,4) + 1.0, requires_grad=True)
        x2 = ag.Tensor(xp.arange(4*3).reshape(4,3) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1, x2)
            return ag.batch_matmul(x1, x2)
        self.numeric_check(forward, x1, x2)


    def test_mask_fill(self):
        x1 = ag.Tensor(xp.arange(3*3, dtype=xp.float64).reshape(3,3)+1.0, requires_grad=True)
        mask = xp.triu(xp.ones(x1.shape)) > 0

        got = ag.mask_fill(x1, ag.Tensor(mask), -math.inf)
        want = xp.where(mask, -math.inf, x1.value())
        self.assertTrue(xp.allclose(got.value(), want))
        got.backward()

    def test_mask_fill_with_different_dimensions(self):
        x1 = ag.Tensor(xp.arange(2*3*3, dtype=xp.float64).reshape(2,3,3)+1.0, requires_grad=True)
        mask = xp.array([
            [
                [True, False, False],
                [False, True, False],
                [False, False, True]
            ]
        ])

        got = ag.mask_fill(x1, ag.Tensor(mask), -math.inf)
        want1 = xp.where(mask, -math.inf, x1.value()[0])
        want2 = xp.where(mask, -math.inf, x1.value()[1])
        self.assertTrue(xp.allclose(got.value()[0], want1))
        self.assertTrue(xp.allclose(got.value()[1], want2))
        got.backward()

    def test_inverse_dropout(self):
        x1 = xp.arange(2*3, dtype=xp.float64).reshape(2,3) + 1
        x = ag.Tensor(x1, requires_grad=True)
        rng_seed = 3 if xp.__name__ == 'cupy' else 268

        # In training, with dropout
        got = ag.inverse_dropout(x, True, p=0.3, rng_seed=rng_seed)
        want = xp.array([
            [1.0, 0.0, 3.0],
            [0.0, 5.0, 6.0]
        ])
        self.assertEqual(got.shape, want.shape)
        self.assertTrue(xp.array_equal(got.value(), want))
        got.backward()
        want_dx = xp.array([
            [1.0/0.3, 0.0    , 1.0/0.3],
            [0.0    , 1.0/0.3, 1.0/0.3]
        ])
        self.assertEqual(x.grad.shape, want_dx.shape)
        self.assertTrue(xp.array_equal(x.grad.value(), want_dx))

        # In test time, no dropout should occur
        got = ag.inverse_dropout(x, False, p=0.5, rng_seed=1)
        want = xp.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        self.assertEqual(got.shape, want.shape)
        self.assertTrue(xp.array_equal(got.value(), want))
        got.backward()
        want_dx = xp.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ])
        self.assertEqual(x.grad.shape, want_dx.shape)
        self.assertTrue(xp.array_equal(x.grad.value(), want_dx))

if __name__ == '__main__':
    unittest.main()
    