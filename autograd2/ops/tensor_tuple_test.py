import unittest
# import numpy as np
import autograd2 as ag
import base_gradient_test
from devices import xp

class TensorTupleTest(base_gradient_test.NumericalGradientTest):

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
        # y2 = y1[0:1]
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

    def test_make_tuple(self):
        xp.random.seed(1)
        x1 = ag.Tensor(1, requires_grad=True)
        x2 = ag.Tensor(2, requires_grad=True)
        x3 = ag.Tensor(3, requires_grad=True)
        x4 = ag.Tensor(4, requires_grad=True)

        z1 = ag.make_tuple(x1, x2, x3, x4)
        z1.requires_grad = True
        z1.backward()
        self.assertEqual(x1.grad.value(), 1)
        self.assertEqual(x2.grad.value(), 1)
        self.assertEqual(x3.grad.value(), 1)    
        self.assertEqual(x4.grad.value(), 1)

    # def test_tensor_tuple_add(self):
    #     xp.random.seed(1)
    #     x1 = ag.Parameter(1, requires_grad=True)
    #     x2 = ag.Parameter(2, requires_grad=True)
    #     x3 = ag.Parameter(3, requires_grad=True)
    #     x4 = ag.Parameter(4, requires_grad=True)

    #     z1 = ag.make_tuple(x1, x2)
    #     z2 = ag.make_tuple(x3, x4)
    #     z3 = ag.tuple_add(z1, z2)
    #     self.assertEqual(z3.value()[0].value(), 4)
    #     self.assertEqual(z3.value()[1].value(), 6)

    #     z3.backward()
    #     self.assertEqual(x1.grad.value(), 1)
    #     self.assertEqual(x2.grad.value(), 1)
    #     self.assertEqual(x3.grad.value(), 1)    
    #     self.assertEqual(x4.grad.value(), 1)
    
    def test_tensor_tuple_sum(self):
        xp.random.seed(1)
        x1 = ag.Parameter(1, requires_grad=True)
        x2 = ag.Parameter(2, requires_grad=True)
        x3 = ag.Parameter(3, requires_grad=True)
        x4 = ag.Parameter(4, requires_grad=True)

        z1 = ag.make_tuple(x1, x2)
        z2 = ag.make_tuple(x3, x4)
        z3 = ag.tuple_sum(z1, z2)
        z3.backward()

        self.assertEquals(z3.value()[0].value(), 4)
        self.assertEquals(z3.value()[1].value(), 6)


    def test_tuple_get_item(self):
        xp.random.seed(1)
        x1 = ag.Parameter(1, requires_grad=True)
        x2 = ag.Parameter(2, requires_grad=True)

        z1 = ag.make_tuple(x1, x2)
        z2 = ag.tuple_get_item(z1, 0)
        z3 = z1[1]
        z4 = z2 * z3
        z4.backward(z4)

        z5 = x1.grad
        z5.backward(ag.Gradient(2.0))

        self.assertEqual(z2.value(), 1)
        self.assertEqual(z3.value(), 2)

    def test_tuple_get_slice(self):
        xp.random.seed(1)
        x1 = ag.Parameter(1)
        x2 = ag.Parameter(xp.array([2,2]))
        x3 = ag.Parameter(3)
        x4 = ag.Parameter(4)
        z1 = x1*x2

        z1 = ag.make_tuple(z1, x3, x4)
        z2 = ag.tuple_get_slice(z1, slice(0, 2))
        z2.backward()

        y1 = z2[0]
        y2 = z2[1]
        self.assertTrue(xp.array_equal(y1.value(), xp.array([2,2])))
        self.assertEquals(y2.value(), 3)

        z2 = z1[0:2]
        y1 = z2[0]
        y2 = z2[1]
        self.assertTrue(xp.array_equal(y1.value(), xp.array([2,2])))
        self.assertEquals(y2.value(), 3)

    def test_tuple_backward(self):
        xp.random.seed(1)
        x1 = ag.Tensor(xp.random.rand(2,3), requires_grad=True)
        x2 = ag.Tensor(xp.random.rand(2,3), requires_grad=True)
        x3 = ag.Tensor(xp.random.rand(2,3), requires_grad=True)
        x4 = ag.Tensor(xp.random.rand(2,3), requires_grad=True)

        def do():
            z1 = ag.make_tuple(x1, x2)
            z2 = ag.make_tuple(x3, x4)

            z3 = ag.tuple_sum(z1, z2)
            z4 = z1[0]*z2[0] + z1[1]*z2[1]
            z5 = z3[0] - z4
            return z5
        def forward(params):
            self.unravel_params(params, x1, x2, x3, x4)
            return do()            
        self.numeric_check(forward, x1, x2, x3, x4)


        

