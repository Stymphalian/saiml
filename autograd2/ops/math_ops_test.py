import unittest
import numpy as np
import autograd2 as ag

class OperatorsTest(unittest.TestCase):

    def numeric_check(self, forward, *inputs, do_print=False):
        params = np.concatenate([x.value().reshape(-1) for x in inputs])
        forward(params).backward()
        predGrads = np.concatenate([x.grad.reshape(-1) for x in inputs])
        
        def forward2(params):
            z1 = forward(params)
            z2 = ag.sum(z1)  # loss is only defined against a single scalar
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
            b = ag.sum(a)
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
        x = ag.Tensor(np.arange(9, dtype=np.float64) + 1.0, requires_grad=True)
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
        x = ag.Tensor(np.arange(3, dtype=np.float64) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.sum(x)
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
            b = ag.sum(a)
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

    def test_convolve2d(self):
        x = ag.Tensor(np.arange(9, dtype=np.float64).reshape(1,3,3) + 1.0, requires_grad=True)
        k = ag.Tensor(np.arange(4, dtype=np.float64).reshape(1,2,2) + 1.0, requires_grad=True)
        def forward(params):
            self.unravel_params(params, x, k)
            return ag.convolve2d(x, k)
        self.numeric_check(forward, x, k)

    def test_convolve2d_with_stride(self):
        np.random.seed(1)
        x = ag.Tensor(np.random.rand(1,5,5), requires_grad=True)
        k = ag.Tensor(np.random.rand(1,3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x, k)
            return ag.convolve2d(x, k, stride=2)
        self.numeric_check(forward, x, k)
    
    def test_convolve2d_with_padding(self):
        np.random.seed(1)
        x = ag.Tensor(np.random.rand(1,5,5), requires_grad=True)
        k = ag.Tensor(np.random.rand(1,3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x, k)
            return ag.convolve2d(x, k, padding=2)
        self.numeric_check(forward, x, k)

    def test_convolve2d_with_dilation(self):
        np.random.seed(1)
        x = ag.Tensor(np.random.rand(1,5,5), requires_grad=True)
        k = ag.Tensor(np.random.rand(1,3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x, k)
            return ag.convolve2d(x, k, dilate=2)
        self.numeric_check(forward, x, k)

    def test_convolve2d_with_stride_padding_dilation(self):
        np.random.seed(1)
        x = ag.Tensor(np.random.rand(1,5,5), requires_grad=True)
        k = ag.Tensor(np.random.rand(1,3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x, k)
            return ag.convolve2d(x, k, stride=2, padding=2, dilate=2)
        self.numeric_check(forward, x, k)

    def test_convolve2d_transpose(self):
        np.random.seed(1)
        x = ag.Tensor(np.random.rand(1,5,5), requires_grad=True)
        k = ag.Tensor(np.random.rand(1,3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x, k)
            return ag.convolve2d_transpose(x, k)
        self.numeric_check(forward, x, k)

    def test_convolve2d_transpose_with_stride(self):
        np.random.seed(1)
        x = ag.Tensor(np.random.rand(1,5,5), requires_grad=True)
        k = ag.Tensor(np.random.rand(1,3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x, k)
            return ag.convolve2d_transpose(x, k, stride=2)
        self.numeric_check(forward, x, k)
    
    def test_convolve2d_transpose_with_padding(self):
        np.random.seed(1)
        x = ag.Tensor(np.random.rand(1,8,8), requires_grad=True)
        k = ag.Tensor(np.random.rand(1,3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x, k)
            return ag.convolve2d_transpose(x, k, padding=1, outer_padding=1)
        self.numeric_check(forward, x, k)

    def test_vstack(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.random.rand(3,3), requires_grad=True)
        x2 = ag.Tensor(np.random.rand(3,3), requires_grad=True)
        x3 = ag.Tensor(np.random.rand(3,3), requires_grad=True)
        x4 = ag.Tensor(np.random.rand(3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1, x2, x3, x4)
            d1 = x1 + x2
            d2 = x3 * x4
            return ag.vstack(d1, d2)
        self.numeric_check(forward, x1, x2, x3, x4)

    def test_vsplit(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.random.rand(10, 3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            return ag.vsplit(x1, 5)
        self.numeric_check(forward, x1)

    def test_broadcast(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.random.rand(3,1,3,1), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            return ag.broadcast(x1, (3,3,3,3))
        self.numeric_check(forward, x1)

if __name__ == '__main__':
    unittest.main()
    