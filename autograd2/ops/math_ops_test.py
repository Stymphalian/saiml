import unittest
import numpy as np
import autograd2 as ag

class OperatorsTest(unittest.TestCase):

    def numeric_check(self, forward, *inputs, do_print=False):
        params = np.concatenate([x.value().reshape(-1) for x in inputs])
        forward(params).backward()
        predGrads = np.concatenate([x.grad.value().reshape(-1) for x in inputs])
        
        def forward2(params):
            z1 = forward(params)
            z2 = ag.summation(z1)  # loss is only defined against a single scalar
            return z2.value()

        grads, diff = ag.utils.numeric_gradient_check(forward2, params, predGrads)
        if do_print:
            print()
            print("numericGrads = ", grads)
            print("predGrads    = ", predGrads)
            print("diff         = ", diff)
        # self.assertTrue(diff < 1e-6, "diff = {0}\ngrads= {1}".format(diff, grads))
        self.assertTrue(diff < 1e-6, "diff = {0}\ngrads= {1}".format(diff, grads))

    def unravel_params(self, params, *inputs):
        count = 0
        for x in inputs:
            x._data = params[count:count+x.size].reshape(x.shape)
            count += x.size

    def test_add(self):
        x1 = ag.Tensor(np.arange(3, dtype=np.float64) + 1.0, requires_grad=True)
        x2 = ag.Tensor(np.arange(3, dtype=np.float64) + 2.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1, x2)
            return ag.add(x1, x2)
        self.numeric_check(forward, x1, x2)

    def test_add_scalar(self):
        x1 = ag.Tensor(np.arange(3, dtype=np.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            return ag.add_scalar(x1, 5.0)
        self.numeric_check(forward, x1)

    def test_sub(self):
        x1 = ag.Tensor(np.arange(3, dtype=np.float64) + 1.0, requires_grad=True)
        x2 = ag.Tensor(np.arange(3, dtype=np.float64) + 2.0, requires_grad=True)

        def forward(params):
            self.unravel_params(params, x1, x2)
            return ag.sub(x1, x2)
        self.numeric_check(forward, x1, x2)

    def test_sub_scalar(self):
        x1 = ag.Tensor(np.arange(3, dtype=np.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            return ag.sub_scalar(x1, 5)
        self.numeric_check(forward, x1)

    def test_mult(self):
        x1 = ag.Tensor(np.arange(3, dtype=np.float64) + 1.0, requires_grad=True)
        x2 = ag.Tensor(np.arange(3, dtype=np.float64) + 2.0, requires_grad=True)

        def forward(params):
            self.unravel_params(params, x1, x2)
            return ag.mult(x1, x2)
        self.numeric_check(forward, x1, x2)

    def test_mult_scalar(self):
        x1 = ag.Tensor(np.arange(3, dtype=np.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            return ag.mult_scalar(x1, 3.0)
        self.numeric_check(forward, x1)

    def test_mult_sum(self):
        x1 = ag.Tensor(np.arange(3, dtype=np.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            a = ag.exp(x1)
            b = ag.summation(a)
            c = ag.mult(a, b)
            return c
        self.numeric_check(forward, x1)

    def test_div(self):
        x1 = ag.Tensor(np.arange(3, dtype=np.float64) + 1.0, requires_grad=True)
        x2 = ag.Tensor(np.arange(3, dtype=np.float64) + 2.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1, x2)
            return ag.div(x1, x2)
        self.numeric_check(forward, x1, x2)
    
    def test_div_scalar(self):
        x1 = ag.Tensor(np.arange(3, dtype=np.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            return ag.div_scalar(x1, 3.0)
        self.numeric_check(forward, x1)

    def test_matmul(self):
        x1 = ag.Tensor(np.arange(9, dtype=np.float64).reshape(3,3) + 1.0, requires_grad=True)
        x2 = ag.Tensor(np.arange(3, dtype=np.float64).reshape(3,1) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1, x2)
            return ag.matmul(x1, x2)
        self.numeric_check(forward, x1, x2)

    def test_sin(self):
        x = ag.Tensor(np.arange(9, dtype=np.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.sin(x)
        self.numeric_check(forward, x)
    
    def test_cos(self):
        x = ag.Tensor(np.arange(9, dtype=np.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.cos(x)
        self.numeric_check(forward, x)

    def test_tan(self):
        x1 = np.arange(9, dtype=np.float64) + 1.0
        x = ag.Tensor(x1, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.tan(x)
        self.numeric_check(forward, x)

    def test_log(self):
        x = ag.Tensor(np.arange(9, dtype=np.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.log(x)
        self.numeric_check(forward, x)

    def test_sum(self):        
        x = ag.Tensor(np.arange(2*2*2, dtype=np.float64).reshape(2,2,2) + 1.0, requires_grad=True)
        y = ag.summation(x, axis=None)
        want = 36.0
        self.assertTrue(np.array_equal(y.value(), want))

        x = ag.Tensor(np.arange(2*2*2, dtype=np.float64).reshape(2,2,2) + 1.0, requires_grad=True)
        y = ag.summation(x, axis=0)
        want = np.array([
            [6.0, 8.0],
            [10.0, 12.0]
        ])
        self.assertTrue(np.array_equal(y.value(), want))

        x = ag.Tensor(np.arange(2*2*2, dtype=np.float64).reshape(2,2,2) + 1.0, requires_grad=True)
        y = ag.summation(x, axis=1)
        want = np.array([
            [4.0, 6.0],
            [12.0,14.0]
        ])
        self.assertTrue(np.array_equal(y.value(), want))

        x = ag.Tensor(np.arange(2*2*2, dtype=np.float64).reshape(2,2,2) + 1.0, requires_grad=True)
        y = ag.summation(x, axis=2)
        want = np.array([
            [3.0, 7.0],
            [11.0, 15.0]
        ])
        self.assertTrue(np.array_equal(y.value(), want))

        x = ag.Tensor(np.arange(2*2*2, dtype=np.float64).reshape(2,2,2) + 1.0, requires_grad=True)
        y = ag.summation(x, axis=None, keepdims=True)
        want = np.array([[[36.0]]])
        self.assertTrue(np.array_equal(y.value(), want))

        x = ag.Tensor(np.arange(2*2*2, dtype=np.float64).reshape(2,2,2) + 1.0, requires_grad=True)
        y = ag.summation(x, axis=0, keepdims=True)
        want = np.array([[
            [6.0, 8.0],
            [10.0, 12.0]
        ]])
        self.assertTrue(np.array_equal(y.value(), want))

        x = ag.Tensor(np.arange(2*2*2, dtype=np.float64).reshape(2,2,2) + 1.0, requires_grad=True)
        y = ag.summation(x, axis=1, keepdims=True)
        want = np.array([
            [[4.0, 6.0]],
            [[12.0,14.0]]
        ])
        self.assertTrue(np.array_equal(y.value(), want))

        x = ag.Tensor(np.arange(2*2*2, dtype=np.float64).reshape(2,2,2) + 1.0, requires_grad=True)
        y = ag.summation(x, axis=2, keepdims=True)
        want = np.array([
            [[3.0], [7.0]],
            [[11.0], [15.0]]
        ])
        self.assertTrue(np.array_equal(y.value(), want))

    def test_sum_backward(self):
        for axis in [None, 0, 1, 2]:
            for keepdims in [True, False]:
                x = ag.Tensor(np.arange(2*3*4, dtype=np.float64).reshape(2,3,4) + 1.0, requires_grad=True)
                def forward(params):
                    self.unravel_params(params, x)
                    return ag.summation(x, axis=axis, keepdims=keepdims)
                self.numeric_check(forward, x)

    def test_exp(self):
        x = ag.Tensor(np.arange(3, dtype=np.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.exp(x)
        self.numeric_check(forward, x)

    def test_mean(self):
        x = ag.Tensor(np.arange(9, dtype=np.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.mean(x)
        self.numeric_check(forward, x)
    
    def test_power(self):
        x = ag.Tensor(np.arange(9, dtype=np.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.power(x, 2)
        self.numeric_check(forward, x)

    def test_max(self):
        x = ag.Tensor(np.arange(9, dtype=np.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.max(x)
        self.numeric_check(forward, x)

    @unittest.skip("Numerically unstable for numeric gradient checking")
    def test_softmax(self):
        x = ag.Tensor(np.array([1,2,3]), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            a = ag.exp(x) - ag.max(x)
            b = ag.summation(a)
            c = ag.div(a, b)
            return c
        self.numeric_check(forward, x, log=True)

    def test_negate(self):
        x = ag.Tensor(np.arange(9, dtype=np.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return -x
        self.numeric_check(forward, x)

    def test_reshape(self):
        np.random.seed(1)
        x = ag.Tensor(np.random.rand(3,4), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.reshape(x, (2,6))
        self.numeric_check(forward, x)

    def test_broadcast(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.random.rand(3,1,3,1), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            return ag.broadcast(x1, (3,3,3,3))
        self.numeric_check(forward, x1)

    def test_norm(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.random.rand(3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            return ag.norm(x1)
        self.numeric_check(forward, x1)

    def test_tuple_make(self):
        x1 = ag.Tensor(1, requires_grad=True)
        x2 = ag.Tensor(2, requires_grad=True)
        x3 = ag.Tensor(3, requires_grad=True)
        x4 = ag.Tensor(4, requires_grad=True)

        z1 = x1 * x2
        z2 = x3 / x4
        z3 = z1 + z2
        z3.backward()
        z3.backward()
        want_x1 = x1.grad.value()
        want_x2 = x2.grad.value()
        want_x3 = x3.grad.value()
        want_x4 = x4.grad.value()

        y1 = ag.make_tuple(z1, z2)
        y2 = ag.tuple_get_item(y1, 0)
        y3 = ag.tuple_get_item(y1, 1)
        y4 = y2 + y3
        y4.backward()
        got_x1 = x1.grad.value()
        got_x2 = x2.grad.value()
        got_x3 = x3.grad.value()
        got_x4 = x4.grad.value()

        self.assertEqual(want_x1, got_x1)
        self.assertEqual(want_x2, got_x2)
        self.assertEqual(want_x3, got_x3)
        self.assertEqual(want_x4, got_x4)


if __name__ == '__main__':
    unittest.main()
    