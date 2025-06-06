import unittest
from devices import xp
import numpy
import autograd2 as ag
import base_gradient_test

# import autograd.numpy as np2  # Thinly-wrapped numpy
# from autograd import grad    # The only autograd function you may ever need

class OperatorsTest(base_gradient_test.NumericalGradientTest):

    def test_add(self):
        xp.random.seed(1)
        shapes = [
            ((3,), (3,)),
            ((3,), (2,3,)),
            ((1,3,), (2,3,)),
            ((1,2,1), (4,2,3)),
            ((1,2,1), (4,2,3)),
        ]
        for shape1, shape2 in shapes:
            x1 = xp.arange(numpy.prod(shape1), dtype=xp.float64).reshape(shape1) + 1.0
            x2 = xp.arange(numpy.prod(shape2), dtype=xp.float64).reshape(shape2) + 1.0
            x1 = ag.Tensor(x1, requires_grad=True)
            x2 = ag.Tensor(x2, requires_grad=True)
            def forward(params):
                self.unravel_params(params, x1, x2)
                return ag.add(x1, x2)
            self.numeric_check(forward, x1, x2)

    def test_add_scalar(self):
        x1 = ag.Tensor(xp.arange(3, dtype=xp.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            return ag.add_scalar(x1, 5.0)
        self.numeric_check(forward, x1)

    def test_sub(self):
        x1 = ag.Tensor(xp.arange(3, dtype=xp.float64) + 1.0, requires_grad=True)
        x2 = ag.Tensor(xp.arange(3, dtype=xp.float64) + 2.0, requires_grad=True)

        def forward(params):
            self.unravel_params(params, x1, x2)
            return ag.sub(x1, x2)
        self.numeric_check(forward, x1, x2)

    def test_sub_scalar(self):
        x1 = ag.Tensor(xp.arange(3, dtype=xp.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            return ag.sub_scalar(x1, 5)
        self.numeric_check(forward, x1)

    def test_mult(self):
        x1 = ag.Tensor(xp.arange(3, dtype=xp.float64) + 1.0, requires_grad=True)
        x2 = ag.Tensor(xp.arange(3, dtype=xp.float64) + 2.0, requires_grad=True)

        def forward(params):
            self.unravel_params(params, x1, x2)
            return ag.mult(x1, x2)
        self.numeric_check(forward, x1, x2)

    def test_mult_scalar(self):
        x1 = ag.Tensor(xp.arange(3, dtype=xp.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            return ag.mult_scalar(x1, 3.0)
        self.numeric_check(forward, x1)

    def test_mult_sum(self):
        x1 = ag.Tensor(xp.arange(3, dtype=xp.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            a = ag.exp(x1)
            b = ag.summation(a)
            c = ag.mult(a, b)
            return c
        self.numeric_check(forward, x1)

    def test_div(self):
        x1 = ag.Tensor(xp.arange(3, dtype=xp.float64) + 1.0, requires_grad=True)
        x2 = ag.Tensor(xp.arange(3, dtype=xp.float64) + 2.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1, x2)
            return ag.div(x1, x2)
        self.numeric_check(forward, x1, x2)
    
    def test_div_scalar(self):
        x1 = ag.Tensor(xp.arange(3, dtype=xp.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            return ag.div_scalar(x1, 3.0)
        self.numeric_check(forward, x1)

    def test_matmul(self):
        x1 = ag.Tensor(xp.arange(9, dtype=xp.float64).reshape(3,3) + 1.0, requires_grad=True)
        x2 = ag.Tensor(xp.arange(3, dtype=xp.float64).reshape(3,1) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1, x2)
            return ag.matmul(x1, x2)
        self.numeric_check(forward, x1, x2)

    def test_matmul_with_shapes(self):
        xp.random.seed(0)
        shapes = [
            ((2,3), (3,4)),
            ((3,3), (3,3)),
            ((3,3), (3,1)),
            # ((3,3), (3,)),
            # ((3,), (3,4)),
            ((2,3,3), (2,3,4)),
            ((1,5,2,3,3), (1,5,2,3,4)),
        ]
        for shape1, shape2 in shapes:
            x1 = xp.arange(numpy.prod(shape1), dtype=xp.float64).reshape(shape1) + 1.0
            x2 = xp.arange(numpy.prod(shape2), dtype=xp.float64).reshape(shape2) + 1.0
            x1 = ag.Tensor(x1, requires_grad=True)
            x2 = ag.Tensor(x2, requires_grad=True)
            def forward(params):
                self.unravel_params(params, x1, x2)
                return ag.matmul(x1, x2)
            self.numeric_check(forward, x1, x2)

    def test_parse_einsum_equation(self):
        equation = " B ij,B jk,Ad -> Bi k   "
        a = xp.arange(2*3*4).reshape(2,3,4)
        b = xp.arange(2*4*3).reshape(2,4,3)
        c = xp.arange(2*4).reshape(2,4)
        inputs = [a,b,c]
        input_eqs, output_eq, operands = ag.parse_einsum_equation(equation, *inputs)
        self.assertEqual(input_eqs[0], "Bij")
        self.assertEqual(input_eqs[1], "Bjk")
        self.assertEqual(input_eqs[2], "Ad")
        self.assertEqual(output_eq, "Bik")
        self.assertEqual(operands[0].shape, (2,3,4))
        self.assertEqual(operands[1].shape, (2,4,3))
        self.assertEqual(operands[2].shape, (2,4))

    def test_einsum(self):
        x1 = ag.Parameter(xp.arange(2*3*4, dtype=xp.float64).reshape(2,3,4) + 1.0, requires_grad=True)
        x2 = ag.Parameter(xp.arange(2*3*4, dtype=xp.float64).reshape(2,4,3) + 1.0, requires_grad=True)
        got = ag.einsum("Bij,Bjk->Bik", x1, x2)
        want = xp.einsum("Bij,Bjk->Bik", x1.value(), x2.value())
        self.assertTrue(xp.allclose(got.value(), want))

        x1 = ag.Parameter(xp.arange(3*4, dtype=xp.float64).reshape(3,4) + 1.0, requires_grad=True)
        x2 = ag.Parameter(xp.arange(3*4, dtype=xp.float64).reshape(3,4) + 1.0, requires_grad=True)
        got = ag.einsum("ij,kj->ik", x1, x2)
        want = xp.matmul(x1.value(), x2.value().T)
        self.assertTrue(xp.allclose(got.value(), want))

        x1 = ag.Parameter(xp.arange(3*4, dtype=xp.float64).reshape(3,4) + 1.0, requires_grad=True)
        x2 = ag.Parameter(xp.arange(3*4, dtype=xp.float64).reshape(4,3) + 1.0, requires_grad=True)
        got = ag.einsum("ij,jk->ik", x1, x2)
        grad = ag.Gradient(xp.zeros(got.shape), requires_grad=True)
        got.backward(grad)
        got_x1_grad = x1.grad
        got_x2_grad = x2.grad
        want_x1_grad = xp.matmul(grad.value(), x2.value().T)
        want_x2_grad = xp.matmul(x1.value().T, grad.value())
        self.assertTrue(xp.allclose(got_x1_grad.value(), want_x1_grad))
        self.assertTrue(xp.allclose(got_x2_grad.value(), want_x2_grad))

        # Test double gradients
        x1.grad.backward()

    def test_einsum_gradient(self):
        x1 = ag.Tensor(xp.arange(2*3*4).reshape(2,3,4) + 1.0, requires_grad=True)
        x2 = ag.Tensor(xp.arange(2*3*4).reshape(2,3,4) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1, x2)
            return ag.einsum("Bij,Bkl->Bijk", x1, x2)
        self.numeric_check(forward, x1, x2)

    def test_sin(self):
        x = ag.Tensor(xp.arange(9, dtype=xp.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.sin(x)
        self.numeric_check(forward, x)
    
    def test_cos(self):
        x = ag.Tensor(xp.arange(9, dtype=xp.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.cos(x)
        self.numeric_check(forward, x)

    def test_tan(self):
        x1 = xp.arange(9, dtype=xp.float64) + 1.0
        x = ag.Tensor(x1, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.tan(x)
        self.numeric_check(forward, x)

    def test_log(self):
        x = ag.Tensor(xp.arange(9, dtype=xp.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.log(x)
        self.numeric_check(forward, x)

    def test_sum(self):        
        x = ag.Tensor(xp.arange(2*2*2, dtype=xp.float64).reshape(2,2,2) + 1.0, requires_grad=True)
        y = ag.summation(x, axis=None)
        want = 36.0
        self.assertTrue(y.value(), want)

        x = ag.Tensor(xp.arange(2*2*2, dtype=xp.float64).reshape(2,2,2) + 1.0, requires_grad=True)
        y = ag.summation(x, axis=0)
        want = xp.array([
            [6.0, 8.0],
            [10.0, 12.0]
        ])
        self.assertTrue(xp.array_equal(y.value(), want))

        x = ag.Tensor(xp.arange(2*2*2, dtype=xp.float64).reshape(2,2,2) + 1.0, requires_grad=True)
        y = ag.summation(x, axis=1)
        want = xp.array([
            [4.0, 6.0],
            [12.0,14.0]
        ])
        self.assertTrue(xp.array_equal(y.value(), want))

        x = ag.Tensor(xp.arange(2*2*2, dtype=xp.float64).reshape(2,2,2) + 1.0, requires_grad=True)
        y = ag.summation(x, axis=2)
        want = xp.array([
            [3.0, 7.0],
            [11.0, 15.0]
        ])
        self.assertTrue(xp.array_equal(y.value(), want))

        x = ag.Tensor(xp.arange(2*2*2, dtype=xp.float64).reshape(2,2,2) + 1.0, requires_grad=True)
        y = ag.summation(x, axis=None, keepdims=True)
        want = xp.array([[[36.0]]])
        self.assertTrue(xp.array_equal(y.value(), want))

        x = ag.Tensor(xp.arange(2*2*2, dtype=xp.float64).reshape(2,2,2) + 1.0, requires_grad=True)
        y = ag.summation(x, axis=0, keepdims=True)
        want = xp.array([[
            [6.0, 8.0],
            [10.0, 12.0]
        ]])
        self.assertTrue(xp.array_equal(y.value(), want))

        x = ag.Tensor(xp.arange(2*2*2, dtype=xp.float64).reshape(2,2,2) + 1.0, requires_grad=True)
        y = ag.summation(x, axis=1, keepdims=True)
        want = xp.array([
            [[4.0, 6.0]],
            [[12.0,14.0]]
        ])
        self.assertTrue(xp.array_equal(y.value(), want))

        x = ag.Tensor(xp.arange(2*2*2, dtype=xp.float64).reshape(2,2,2) + 1.0, requires_grad=True)
        y = ag.summation(x, axis=2, keepdims=True)
        want = xp.array([
            [[3.0], [7.0]],
            [[11.0], [15.0]]
        ])
        self.assertTrue(xp.array_equal(y.value(), want))

    def test_sum_backward(self):
        for axis in [None, 0, 1, 2, (0,1), (0,2), (1,2)]:
            for keepdims in [True, False]:
                x = ag.Tensor(xp.arange(2*3*4, dtype=xp.float64).reshape(2,3,4) + 1.0, requires_grad=True)
                def forward(params):
                    self.unravel_params(params, x)
                    return ag.summation(x, axis=axis, keepdims=keepdims)
                self.numeric_check(forward, x)

    def test_exp(self):
        x = ag.Tensor(xp.arange(3, dtype=xp.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.exp(x)
        self.numeric_check(forward, x)

    def test_mean(self):
        x = ag.Tensor(xp.arange(2*3*4, dtype=xp.float64).reshape(2,3,4) + 1.0, requires_grad=True)
        for axis in [None, 0, 1, 2, (0,1), (0,2), (1,2)]:
            for keepdims in [True, False]:
                z = ag.mean(x, axis=axis, keepdims=keepdims)
                want = xp.mean(x.value(), axis=axis, keepdims=keepdims)
                self.assertTrue(xp.array_equal(z.value(), want))

    def test_mean_gradient(self):
        x = ag.Tensor(xp.arange(2*3*4, dtype=xp.float64).reshape(2,3,4) + 1.0, requires_grad=True)
        z = ag.mean(x, axis=(1,2))
        z.backward()
        got = x.grad.value()
        n = 1.0 / 12.0
        want = xp.array([
            [
                [n, n, n, n],
                [n, n, n, n],
                [n, n, n, n],
            ],
            [
                [n, n, n, n],
                [n, n, n, n],
                [n, n, n, n],
            ]
        ])
        self.assertTrue(xp.array_equal(got, want))

    def test_mean_backward(self):
        x = ag.Tensor(xp.arange(2*3*4, dtype=xp.float64).reshape(2,3,4) + 1.0, requires_grad=True)
        for axis in [None, 0, 1, 2, (0,1), (0,2), (1,2)]:
            for keepdims in [True, False]:
                def forward(params):
                    self.unravel_params(params, x)
                    return ag.mean(x, axis=axis, keepdims=keepdims)
                self.numeric_check(forward, x)
    
    def test_power(self):
        x = ag.Tensor(xp.arange(9, dtype=xp.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.power(x, 2)
        self.numeric_check(forward, x)

    def test_argmax_axes(self):
        xp.random.seed(0)

        # test1
        x = xp.arange(2*3*5)
        xp.random.shuffle(x)
        x = x.reshape(2,3,5)
        got_indices = ag.argmax_axes_vectorized(x, axes=(1,2), keepdims=False)
        got = xp.zeros((2,3,5))
        xp.put(got, got_indices, 1.0)

        want1 = xp.zeros((3,5))
        want2 = xp.zeros((3,5))
        want1[xp.unravel_index(xp.argmax(x[0]), (3,5))] = 1.0
        want2[xp.unravel_index(xp.argmax(x[1]), (3,5))] = 1.0
        self.assertTrue(xp.allclose(got[0], want1))
        self.assertTrue(xp.allclose(got[1], want2))

    def test_argmax_axes_seperated_axes(self):
        # test 2
        x = xp.arange(2*3*5*2)
        xp.random.shuffle(x)
        x = x.reshape(3,2,2,5)
        got_indices = ag.argmax_axes_vectorized(x, axes=(0,3))
        got = xp.zeros((3,2,2,5))
        xp.put(got, got_indices, 1.0)

        want1 = xp.zeros((3,5))
        want2 = xp.zeros((3,5))
        want3 = xp.zeros((3,5))
        want4 = xp.zeros((3,5))
        want1[xp.unravel_index(xp.argmax(x[:,0,0,:]), (3,5))] = 1
        want2[xp.unravel_index(xp.argmax(x[:,0,1,:]), (3,5))] = 1
        want3[xp.unravel_index(xp.argmax(x[:,1,0,:]), (3,5))] = 1
        want4[xp.unravel_index(xp.argmax(x[:,1,1,:]), (3,5))] = 1
        self.assertTrue(xp.allclose(got[:,0,0,:], want1))
        self.assertTrue(xp.allclose(got[:,0,1,:], want2))
        self.assertTrue(xp.allclose(got[:,1,0,:], want3))
        self.assertTrue(xp.allclose(got[:,1,1,:], want4))

    def test_argmax_axes_negative_indexing(self):
        # test negative indexing
        x = xp.arange(2*3*5)
        xp.random.shuffle(x)
        x = x.reshape(2,3,5)
        got_indices = ag.argmax_axes_vectorized(x, axes=(-2,-1))
        got = xp.zeros((2,3,5))
        xp.put(got, got_indices, 1.0)
        want1 = xp.zeros((3,5))
        want2 = xp.zeros((3,5))
        want1[xp.unravel_index(xp.argmax(x[0]), (3,5))] = 1
        want2[xp.unravel_index(xp.argmax(x[1]), (3,5))] = 1
        self.assertTrue(xp.allclose(got[0], want1))
        self.assertTrue(xp.allclose(got[1], want2))

    def test_argmax_axes_last_dimensions(self):
        # test 4 (should use optimized version)
        x = xp.arange(2*3*5*2)
        xp.random.shuffle(x)
        x = x.reshape(2,2,3,5)
        got_indices = ag.argmax_axes_vectorized(x, axes=(3,2))
        got = xp.zeros((2,2,3,5))
        xp.put(got, got_indices, 1.0)

        want1 = xp.zeros((3,5))
        want2 = xp.zeros((3,5))
        want3 = xp.zeros((3,5))
        want4 = xp.zeros((3,5))
        want1[xp.unravel_index(xp.argmax(x[0,0,:,:]), (3,5))] = 1
        want2[xp.unravel_index(xp.argmax(x[0,1,:,:]), (3,5))] = 1
        want3[xp.unravel_index(xp.argmax(x[1,0,:,:]), (3,5))] = 1
        want4[xp.unravel_index(xp.argmax(x[1,1,:,:]), (3,5))] = 1
        self.assertTrue(xp.allclose(got[0,0,:,:], want1))
        self.assertTrue(xp.allclose(got[0,1,:,:], want2))
        self.assertTrue(xp.allclose(got[1,0,:,:], want3))
        self.assertTrue(xp.allclose(got[1,1,:,:], want4))

    @unittest.skipIf(not xp.__name__ == 'cupy', reason='cupy only')
    def test_argmax_axes_kernel(self):
        xp.random.seed(0)

        # test1
        x = xp.arange(2*3*5)
        xp.random.shuffle(x)
        x = x.reshape(2,3,5)
        got_indices = ag.argmax_axes_vectorized_kernel(x, axes=(1,2), keepdims=False)
        got = xp.zeros((2,3,5))
        xp.put(got, got_indices, 1.0)

        want1 = xp.zeros((3,5))
        want2 = xp.zeros((3,5))
        want1[xp.unravel_index(xp.argmax(x[0]), (3,5))] = 1.0
        want2[xp.unravel_index(xp.argmax(x[1]), (3,5))] = 1.0
        self.assertTrue(xp.allclose(got[0], want1))
        self.assertTrue(xp.allclose(got[1], want2))

    @unittest.skipIf(not xp.__name__ == 'cupy', reason='cupy only')
    def test_argmax_axes_kernel_seperated_axes(self):
        # test 2
        x = xp.arange(2*3*5*2)
        xp.random.shuffle(x)
        x = x.reshape(3,2,2,5)
        got_indices = ag.argmax_axes_vectorized_kernel(x, axes=(0,3))
        got = xp.zeros((3,2,2,5))
        xp.put(got, got_indices, 1.0)

        want1 = xp.zeros((3,5))
        want2 = xp.zeros((3,5))
        want3 = xp.zeros((3,5))
        want4 = xp.zeros((3,5))
        want1[xp.unravel_index(xp.argmax(x[:,0,0,:]), (3,5))] = 1
        want2[xp.unravel_index(xp.argmax(x[:,0,1,:]), (3,5))] = 1
        want3[xp.unravel_index(xp.argmax(x[:,1,0,:]), (3,5))] = 1
        want4[xp.unravel_index(xp.argmax(x[:,1,1,:]), (3,5))] = 1
        self.assertTrue(xp.allclose(got[:,0,0,:], want1))
        self.assertTrue(xp.allclose(got[:,0,1,:], want2))
        self.assertTrue(xp.allclose(got[:,1,0,:], want3))
        self.assertTrue(xp.allclose(got[:,1,1,:], want4))

    @unittest.skipIf(not xp.__name__ == 'cupy', reason='cupy only')
    def test_argmax_axes_kernel_negative_indexing(self):
        # test negative indexing
        x = xp.arange(2*3*5)
        xp.random.shuffle(x)
        x = x.reshape(2,3,5)
        got_indices = ag.argmax_axes_vectorized_kernel(x, axes=(-2,-1))
        got = xp.zeros((2,3,5))
        xp.put(got, got_indices, 1.0)
        want1 = xp.zeros((3,5))
        want2 = xp.zeros((3,5))
        want1[xp.unravel_index(xp.argmax(x[0]), (3,5))] = 1
        want2[xp.unravel_index(xp.argmax(x[1]), (3,5))] = 1
        self.assertTrue(xp.allclose(got[0], want1))
        self.assertTrue(xp.allclose(got[1], want2))

    @unittest.skipIf(not xp.__name__ == 'cupy', reason='cupy only')
    def test_argmax_axes_kernel_last_dimensions(self):
        # test 4 (should use optimized version)
        x = xp.arange(2*3*5*2)
        xp.random.shuffle(x)
        x = x.reshape(2,2,3,5)
        got_indices = ag.argmax_axes_vectorized_kernel(x, axes=(3,2))
        got = xp.zeros((2,2,3,5))
        xp.put(got, got_indices, 1.0)

        want1 = xp.zeros((3,5))
        want2 = xp.zeros((3,5))
        want3 = xp.zeros((3,5))
        want4 = xp.zeros((3,5))
        want1[xp.unravel_index(xp.argmax(x[0,0,:,:]), (3,5))] = 1
        want2[xp.unravel_index(xp.argmax(x[0,1,:,:]), (3,5))] = 1
        want3[xp.unravel_index(xp.argmax(x[1,0,:,:]), (3,5))] = 1
        want4[xp.unravel_index(xp.argmax(x[1,1,:,:]), (3,5))] = 1
        self.assertTrue(xp.allclose(got[0,0,:,:], want1))
        self.assertTrue(xp.allclose(got[0,1,:,:], want2))
        self.assertTrue(xp.allclose(got[1,0,:,:], want3))
        self.assertTrue(xp.allclose(got[1,1,:,:], want4))

    def test_max(self):
        x1 = xp.arange(2*3*4, dtype=xp.float64)
        xp.random.shuffle(x1)
        x1 = x1.reshape(2,3,4) + 1.0
        x = ag.Tensor(x1, requires_grad=True)
        for axis in [None, 0, 1, 2, (0,1), (0,2), (1,2)]:
            for keepdims in [True, False]:
                z = ag.max(x, axis=axis, keepdims=keepdims)
                want = xp.max(x.value(), axis=axis, keepdims=keepdims)
                self.assertTrue(xp.array_equal(z.value(), want))

    def test_max_backward(self):
        x1 = xp.arange(2*3*4, dtype=xp.float64)
        xp.random.shuffle(x1)
        x1 = x1.reshape(2,3,4) + 1.0
        x = ag.Tensor(x1, requires_grad=True)
        for axis in [None, 0, 1, 2, (0,1), (0,2), (1,2)]:
            for keepdims in [True, False]:
                def forward(params):
                    self.unravel_params(params, x)
                    return ag.max(x, axis=axis, keepdims=keepdims)
                self.numeric_check(forward, x)

    @unittest.skip("Numerically unstable for numeric gradient checking")
    def test_softmax(self):
        x = ag.Tensor(xp.array([1,2,3]), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            a = ag.exp(x) - ag.max(x)
            b = ag.summation(a)
            c = ag.div(a, b)
            return c
        self.numeric_check(forward, x, log=True)

    def test_negate(self):
        x = ag.Tensor(xp.arange(9, dtype=xp.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return -x
        self.numeric_check(forward, x)

    def test_reshape(self):
        xp.random.seed(1)
        x = ag.Tensor(xp.random.rand(3,4), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.reshape(x, (2,6))
        self.numeric_check(forward, x)

    def test_broadcasting_axes(self):
        tests = [
            ((4,), (2,4), (0,)),
            ((1,4), (2,4), (0,)),
            ((2,1,4), (2,3,4), (1,)),
            ((2,1,2,1), (2,5,2,6), (1,3)),
        ]
        for shape1, shape2, want in tests:
            got = ag.get_broadcasting_axes(shape1, shape2)
            self.assertEquals(got, want)
            got = ag.get_broadcasting_axes(shape2, shape1)
            self.assertEquals(got, want)

    def test_broadcast(self):
        xp.random.seed(1)
        # Test broadcast from 2d matrix to 3d
        x1 = ag.Tensor(xp.random.rand(3,4), requires_grad=True)
        y = ag.broadcast(x1, (2,3,4))
        self.assertEqual(y.shape, (2,3,4))
        self.assertTrue(xp.array_equal(y.value()[0], x1.value()))
        self.assertTrue(xp.array_equal(y.value()[1], x1.value()))
        y.backward()
        self.assertTrue(xp.array_equal(x1.grad.value(), 2*xp.ones((3,4))))

        # Test broadcast from 1d vector to 3d
        x1 = ag.Tensor(xp.random.rand(4), requires_grad=True)
        y = ag.broadcast(x1, (2,3,4))
        self.assertEqual(y.shape, (2,3,4))
        self.assertTrue(xp.array_equal(y.value()[0][0], x1.value()))
        self.assertTrue(xp.array_equal(y.value()[1][2], x1.value()))
        y.backward()
        self.assertTrue(xp.array_equal(x1.grad.value(), 2*3*xp.ones((4))))

    def test_broadcast_gradient(self):
        xp.random.seed(1)

        broadcast_shapes = [
            ((4,), (2,3,4)),
            ((3,4), (2,3,4)),
            ((1,4), (2,3,4)),
            ((1,1), (2,3,4)),
            ((2,1,4), (5,2,3,4)),
        ]
        for input_shape, output_shape in broadcast_shapes:
            x1 = ag.Tensor(xp.random.rand(*input_shape), requires_grad=True)
            def forward(params):
                self.unravel_params(params, x1)
                return ag.broadcast(x1, output_shape)
            self.numeric_check(forward, x1)

    def test_norm(self):
        xp.random.seed(1)
        x1 = ag.Tensor(xp.random.rand(2,3,4), requires_grad=True)
        for axis in [None, 0, 1, 2, (0,1), (0,2), (1,2)]:
            for keepdims in [True, False]:
                got = ag.norm(x1, axis=axis, keepdims=keepdims)
                want = xp.linalg.norm(x1.value(), axis=axis, keepdims=keepdims)
                self.assertTrue(xp.array_equal(got.value(), want))

    def test_norm_backward(self):
        xp.random.seed(1)
        for axis in [None, 0, 1, 2, (0,1), (0,2), (1,2)]:
            for keepdims in [True, False]:
                x1 = ag.Tensor(xp.random.rand(2,3,4), requires_grad=True)
                def forward(params):
                    self.unravel_params(params, x1)
                    return ag.norm(x1, axis=axis, keepdims=keepdims)
                self.numeric_check(forward, x1)

    def test_sqrt(self):
        x1 = ag.Tensor(xp.arange(9).reshape(3,3)+1, requires_grad=True)
        got = ag.sqrt(x1)
        want = xp.sqrt(x1.value())
        self.assertTrue(xp.array_equal(got.value(), want))

        got.backward()
        got_grad = x1.grad
        want_grad = 1.0 / (2*xp.sqrt(x1.value()))
        self.assertTrue(xp.array_equal(got_grad.value(), want_grad))

        x1 = ag.Tensor(xp.array([4,9,16,25], dtype=xp.float64).reshape(2,2), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            return ag.sqrt(x1)
        self.numeric_check(forward, x1)

    def test_where(self):
        x1 = ag.Tensor(xp.ones((3,3))*10, requires_grad=True)
        x2 = ag.Tensor(xp.ones((3,3))*20, requires_grad=True)
        mask = xp.array([[True, False, True], [False, True, False], [True, False, True]], dtype=bool)
        got = ag.where(mask, x1, x2)
        want = xp.where(mask, x1.value(), x2.value())
        self.assertTrue(xp.array_equal(got.value(), want))
        got.backward()

        def forward(params):
            self.unravel_params(params, x1,x2)
            z = x1 * x2
            z = ag.where(mask, x1, x2)
            y = x1 + z
            return y
        self.numeric_check(forward, x1, x2)

    def test_bitwise_or(self):
        x1 = ag.Tensor(xp.random.randint(0, 2, size=(3,3)) == 1, requires_grad=True)
        x2 = ag.Tensor(xp.random.randint(0, 2, size=(3,3)) == 1, requires_grad=True)
        got = ag.bitwise_or(x1, x2)
        want = xp.bitwise_or(x1.value(), x2.value())
        self.assertTrue(xp.array_equal(got.value(), want))
        got.backward()

        # def forward(params):
        #     self.unravel_params(params, x1,x2)
        #     return ag.bitwise_or(x1, x2)
        # self.numeric_check(forward, x1, x2)

    def test_abs(self):
        x1 = ag.Tensor(xp.arange(9).reshape(3,3)-5, requires_grad=True)
        got = ag.abs(x1)
        want = xp.abs(x1.value())
        self.assertTrue(xp.array_equal(got.value(), want))

        got.backward()
        got_grad = x1.grad
        want_grad = xp.sign(x1.value())
        self.assertTrue(xp.array_equal(got_grad.value(), want_grad))

        x1 = ag.Tensor(xp.array([4,9,-16,25], dtype=xp.float64).reshape(2,2), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            return ag.abs(x1)
        self.numeric_check(forward, x1)


if __name__ == '__main__':
    unittest.main()
    