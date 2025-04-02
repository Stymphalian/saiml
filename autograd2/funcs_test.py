import unittest
import numpy as np
import autograd2 as ag

class OperatorsTest(unittest.TestCase):

    def numeric_check(self, forward, *inputs, threshold=1e-6, do_print=False):
        params = np.concatenate([x.value().reshape(-1) for x in inputs])
        y = forward(params)
        y.backward()
        predGrads = np.concatenate([x.grad.value().reshape(-1) for x in inputs])
        
        def forward2(params):
            z1 = forward(params)
            z2 = ag.summation(z1)  # loss is only defined against a single scalar
            return z2.value()

        grads, diff = ag.utils.numeric_gradient_check(forward2, params, predGrads)
        if do_print:
            print()
            print("grads =", grads)
            print("predGrads =", predGrads)
            print("diff = ", diff)
        self.assertTrue(diff < threshold, "diff = {0}\ngrads= {1}".format(diff, grads))
        # self.assertTrue(diff < 10, "diff = {0}\ngrads= {1}".format(diff, grads))

    def unravel_params(self, params, *inputs):
        count = 0
        for x in inputs:
            x._data = params[count:count+x.size].reshape(x.shape)
            count += x.size

    def test_sigmoid(self):
        np.random.seed(1)
        x = ag.Tensor(np.random.normal(1000), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.sigmoid(x)
            # return ag.Tensor(1) / (1.0 + ag.exp(-x))
        self.numeric_check(forward, x)

    def test_cross_entropy_loss(self):
        np.random.seed(1)
        x = ag.Tensor(np.random.rand(10), requires_grad=True)
        y = ag.Tensor(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64))
        def forward(params):
            self.unravel_params(params, x)
            return ag.cross_entropy_loss(x, y)
        self.numeric_check(forward, x, threshold=1e-4)

    def test_softplus(self):
        np.random.seed(1)
        x = ag.Tensor(np.random.rand(10), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.softplus(x)
        self.numeric_check(forward, x)

    def test_softmax_gradient(self):
        np.random.seed(1)
        x = ag.Tensor(np.random.rand(10), requires_grad=True)
        y = ag.softmax(x)
        y.backward()


    def test_softmax(self):
        np.random.seed(1)
        x = ag.Tensor(np.random.rand(10), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.softmax(x)
        self.numeric_check(forward, x, threshold=1.0)
        # self.numeric_check(forward, x)
    
    def test_sequence(self):
        # np.random.seed(1)
        
        x_train = np.array([
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
        ], dtype=np.float64)
        y_train = np.array([1,0,0,0,0, 0,0,0,0,0])
        y = ag.Tensor(y_train.reshape(-1, 1), requires_grad=True)
        x = ag.Tensor(x_train.reshape(-1, 1), requires_grad=True)
        w = ag.Tensor(np.random.rand(10, 784), requires_grad=True)
        b = ag.Tensor(np.random.rand(10, 1), requires_grad=True)
        def forward(params):
            self.unravel_params(params, w, b)
            z1 = ag.matmul(w, x) + b
            z2 = ag.sigmoid(z1)
            z3 = ag.softmax(z2)
            # z4 = ag.cross_entropy_loss(z3, y)
            return z3
        self.numeric_check(forward, w, b)

    def test_relu(self):
        np.random.seed(1)
        x = ag.Tensor(np.random.rand(10), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x)
            return ag.relu(x)
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

if __name__ == '__main__':
    unittest.main()
    