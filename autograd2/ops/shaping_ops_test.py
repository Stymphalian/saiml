import unittest
import numpy as np
import autograd2 as ag

class ShapingOperatorsTest(unittest.TestCase):

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

    def test_transpose(self):
        x1 = ag.Tensor(np.arange(2*9, dtype=np.float64).reshape(2,3,3), requires_grad=True)
        y = ag.transpose(x1, (0,2,1))
        want = np.array([
            [
                [0,3,6],
                [1,4,7],
                [2,5,8]
            ],
            [
                [9,12,15],
                [10,13,16],
                [11,14,17],
            ]
        ])
        self.assertTrue(np.allclose(y.value(), want))

    def test_transpose_gradient(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.random.rand(5,3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            return ag.transpose(x1, (1,0,2))
        self.numeric_check(forward, x1)

    def test_repeat(self):
        np.random.seed(1)
        x = np.random.rand(2,3,4)
        x1 = ag.Tensor(x, requires_grad=True)
        for axis in [None, 0, 1, 2]:
            got = ag.repeat(x1, 2, axis=axis)
            want = np.repeat(x, 2, axis=axis)
            self.assertTrue(np.array_equal(got.value(), want))

    def test_repeat_gradient(self):
        np.random.seed(1)
        x = np.random.rand(2,3,4)
        x1 = ag.Tensor(x, requires_grad=True)
        got = ag.repeat(x1, 2, axis=1)
        got.backward(ag.Tensor(np.array([
            [
                [ 1,  2,  3,  4],
                [ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [ 9, 10, 11, 12],
            ],
            [
                [13, 14, 15, 16],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
                [17, 18, 19, 20],
                [21, 22, 23, 24],
                [21, 22, 23, 24],
            ]
        ])))
        want_grad = np.array([
            [
                [ 2 , 4,  6,  8],
                [10 ,12, 14, 16],
                [18 ,20, 22, 24],
            ],
            [
                [26, 28, 30, 32],
                [34, 36, 38, 40],
                [42, 44, 46, 48],
            ]
        ])
        self.assertTrue(np.array_equal(x1.grad.value(), want_grad))

    def test_repeat_backward(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.random.rand(2,3,4), requires_grad=True)
        for axis in [None, 0, 1, 2]:
            def forward(params):
                self.unravel_params(params, x1)
                return ag.repeat(x1, 2, axis=axis)
            self.numeric_check(forward, x1)

    def test_tiles(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.random.rand(2,3,4), requires_grad=True)
        test_cases = [
            2,
            (2, 1),
            (1, 2),
            (1,1,2),
            (2,1,1,1),
            (2,2,1,1,1),
        ]
        for repeats in test_cases:
            got = ag.tile(x1, repeats)
            want = np.tile(x1.value(), repeats)
            self.assertTrue(np.array_equal(got.value(), want))

    def test_tile_gradient(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.arange(3*2*1).reshape(3,2,1), requires_grad=True)
        
        # test repeat 2, should expand to (1,1,2)
        y = ag.tile(x1, 2)
        y.backward(y)
        want = np.array([
            [[0], [2]],
            [[4], [6]],
            [[8], [10]]
        ])
        self.assertTrue(np.array_equal(x1.grad.value(), want))

        # test repeat (1,2) should expand to (1,1,2)
        y = ag.tile(x1, (1,2))
        y.backward(y)
        want = np.array([
            [[0], [2]],
            [[4], [6]],
            [[8], [10]]
        ])
        self.assertTrue(np.array_equal(x1.grad.value(), want))

        # test repeat (2,1,1,1)
        y = ag.tile(x1, (2,1,1,1))
        y.backward(y)
        want = np.array([
            [[0], [2]],
            [[4], [6]],
            [[8], [10]]
        ])
        self.assertTrue(np.array_equal(x1.grad.value(), want))

        # test repeat (3,1,2)
        y = ag.tile(x1, (3,1,2))
        y.backward(y)
        want = np.array([
            [[0*3], [2*3]],
            [[4*3], [6*3]],
            [[8*3], [10*3]]
        ])
        self.assertTrue(np.array_equal(x1.grad.value(), want))

    def test_tile_backward(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.arange(2*3*4, dtype=np.float64).reshape(2,3,4), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            return ag.tile(x1, (2,1,1,1))
        self.numeric_check(forward, x1)

    def test_stack(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.arange(3*4).reshape(3,4), requires_grad=True)
        x2 = ag.Tensor(np.arange(3*4).reshape(3,4) + 12, requires_grad=True)
        y = ag.stack((x1, x2))
        want = np.stack([x1.value(), x2.value()], axis=0)
        self.assertTrue(np.array_equal(y.value(), want))

        y = ag.stack((x1, x2), axis=1)
        want = np.stack([x1.value(), x2.value()], axis=1)
        self.assertTrue(np.array_equal(y.value(), want))

    def test_stack_gradient(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.arange(3*4).reshape(3,4), requires_grad=True)
        x2 = ag.Tensor(np.arange(3*4).reshape(3,4) + 12, requires_grad=True)

        y = ag.stack((x1, x2), axis=1)
        y.backward(y)
        self.assertTrue(np.array_equal(x1.grad.value(), x1.value()))
        self.assertTrue(np.array_equal(x2.grad.value(), x2.value()))

        y = ag.stack((x1, x2))
        y.backward(y)
        self.assertTrue(np.array_equal(x1.grad.value(), x1.value()))
        self.assertTrue(np.array_equal(x2.grad.value(), x2.value()))

    def test_stack_backward(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.random.rand(3,4), requires_grad=True)
        x2 = ag.Tensor(np.random.rand(3,4), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1, x2)
            return ag.stack((x1, x2), axis=0)
        self.numeric_check(forward, x1, x2)

    def test_unstack(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.arange(2*3*4).reshape(2,3,4), requires_grad=True)
        y = ag.unstack(x1, axis=0)
        self.assertEqual(len(y.value()), 2)
        self.assertTrue(np.array_equal(y[0].value(), x1[0].value()))
        self.assertTrue(np.array_equal(y[1].value(), x1[1].value()))

        y = ag.unstack(x1, axis=1)
        self.assertEqual(len(y.value()), 3)
        self.assertTrue(np.array_equal(y[0].value(), x1[:,0,:].value()))
        self.assertTrue(np.array_equal(y[1].value(), x1[:,1,:].value()))
        self.assertTrue(np.array_equal(y[2].value(), x1[:,2,:].value()))

    def test_unstack_gradient(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.arange(2*3*4).reshape(2,3,4), requires_grad=True)
        y = ag.unstack(x1, axis=0)
        y.backward(y)
        self.assertTrue(np.array_equal(x1.grad.value(), x1.value()))

        y = ag.unstack(x1, axis=1)
        y.backward(y)
        self.assertTrue(np.array_equal(x1.grad.value(), x1.value()))

        y = ag.unstack(x1, axis=2)
        y.backward(y)
        self.assertTrue(np.array_equal(x1.grad.value(), x1.value()))

    def test_unstack_backward(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.random.rand(2,3,4), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            z1 = ag.unstack(x1, axis=0)
            z2 = ag.tuple_get_item(z1, 0)
            z3 = ag.tuple_get_item(z1, 1)
            z4 = z2 + z3
            return z4
        self.numeric_check(forward, x1)
    
    def test_concatenate(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.random.rand(3,3), requires_grad=True)
        x2 = ag.Tensor(np.random.rand(3,3), requires_grad=True)

        for axis in [None, 0, 1]:
            y = ag.concatenate((x1,x2), axis=axis)
            want = np.concatenate((x1.value(), x2.value()), axis=axis)
            self.assertTrue(np.array_equal(y.value(), want))

    def test_concatenate_gradient(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.random.rand(3,3), requires_grad=True)
        x2 = ag.Tensor(np.random.rand(3,3), requires_grad=True)

        for axis in [None, 0, 1]:
            y = ag.concatenate((x1,x2), axis=axis)
            y.backward(y)
            self.assertTrue(np.array_equal(x1.grad.value(), x1.value()))
            self.assertTrue(np.array_equal(x2.grad.value(), x2.value()))

    def test_concatenate_backward(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.random.rand(3,3), requires_grad=True)
        x2 = ag.Tensor(np.random.rand(3,3)+9, requires_grad=True)
        x3 = ag.Tensor(np.random.rand(3,3)+18, requires_grad=True)
        x4 = ag.Tensor(np.random.rand(3,3)+36, requires_grad=True)

        for axis in [None, 0, 1]:
            def forward(params):
                self.unravel_params(params, x1, x2, x3, x4)
                y1 = x1 * x2
                y2 = x3 / x4
                return ag.concatenate((y1, y2), axis=axis)
            self.numeric_check(forward, x1, x2, x3, x4)

    def test_split(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.arange(2*5*1).reshape(2,5,1), requires_grad=True)
        y = ag.split(x1, 5, axis=1)
        self.assertEqual(len(y.value()), 5)
        want = [
            np.array([[[0]],[[5]]]),
            np.array([[[1]],[[6]]]),
            np.array([[[2]],[[7]]]),
            np.array([[[3]],[[8]]]),
            np.array([[[4]],[[9]]]),
        ]
        for i in range(5):
            self.assertTrue(np.array_equal(y[i].value(), want[i]))

    def test_split_gradient(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.arange(2*5*1).reshape(5,2,1), requires_grad=True)
        y = ag.split(x1, 5, axis=0)
        y.backward(y)
        self.assertTrue(np.array_equal(x1.grad.value(), x1.value()))

        x1 = ag.Tensor(np.arange(2*5*1).reshape(2,5,1), requires_grad=True)
        y = ag.split(x1, 5, axis=1)
        y.backward(y)
        self.assertTrue(np.array_equal(x1.grad.value(), x1.value()))

        x1 = ag.Tensor(np.arange(2*5*1).reshape(2,1,5), requires_grad=True)
        y = ag.split(x1, 5, axis=2)
        y.backward(y)
        self.assertTrue(np.array_equal(x1.grad.value(), x1.value()))

    def test_split_backward(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.random.rand(10,3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)

            z1 = ag.split(x1, 5)
            z2 = ag.tuple_get_item(z1, 0)
            z3 = ag.tuple_get_item(z1, 1)
            z4 = z2 + z3
            return z4
        self.numeric_check(forward, x1)

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
            return ag.vstack((d1, d2))
        self.numeric_check(forward, x1, x2, x3, x4)

    def test_vsplit(self):
        np.random.seed(1)
        x1 = ag.Tensor(np.random.rand(10, 3,3), requires_grad=True)
        def forward(params):
            self.unravel_params(params, x1)
            z1 = ag.vsplit(x1, 5)
            z2 = ag.tuple_get_item(z1, 0)
            z3 = ag.tuple_get_item(z1, 1)
            z4 = z2 + z3
            return z4
        self.numeric_check(forward, x1)

if __name__ == '__main__':
    unittest.main()
    