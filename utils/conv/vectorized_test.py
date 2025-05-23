import unittest
from devices import xp
from . import vectorized

class TestConvVectorized(unittest.TestCase):

    def test_max_pool2d_vectorized(self):
        x = xp.array([
            [
                [1,2,3,4,5],
                [6,5,4,3,2],
                [2,3,4,5,6],
                [0,1,2,3,4],
                [0,9,2,6,7]
            ],
            [
                [1,2,3,4,5],
                [6,5,4,3,2],
                [2,3,4,5,6],
                [0,1,2,3,4],
                [0,9,2,6,7]
            ]
        ])
        dy = xp.ones((2,3,3))
        y = vectorized._max_pool2d_vectorized(
            x, kernel_size=3, stride=1)
        dx = vectorized._max_pool2d_gradient_vectorized(
            x, kernel_size=3, outGrad=dy)

        want_y = xp.array([
            [
                [6,5,6],
                [6,5,6],
                [9,9,7]
            ],
            [
                [6,5,6],
                [6,5,6],
                [9,9,7]
            ]
        ])
        want_dx = xp.array([
            [
                [0,  0,  0,  0,  0],
                [2,  2,  0,  0,  0],
                [0,  0,  0,  0,  2],
                [0,  0,  0,  0,  0],
                [0,  2,  0,  0,  1],
            ],
            [
                [0,  0,  0,  0,  0],
                [2,  2,  0,  0,  0],
                [0,  0,  0,  0,  2],
                [0,  0,  0,  0,  0],
                [0,  2,  0,  0,  1],
            ]
        ])
        self.assertEqual(y.shape, (2,3,3))
        self.assertTrue(xp.allclose(y, want_y))
        self.assertEqual(dx.shape, x.shape)
        self.assertTrue(xp.allclose(dx, want_dx))

    def test_average_pool2d_vectorized(self):
        x = xp.array([
            [
                [1,2,3,4,5],
                [6,5,4,3,2],
                [2,3,4,5,6],
                [0,1,2,3,4],
                [0,9,2,6,7]
            ],
            [
                [1,2,3,4,5],
                [6,5,4,3,2],
                [2,3,4,5,6],
                [0,1,2,3,4],
                [0,9,2,6,7]
            ]
        ])
        dy = xp.ones((2,3,3))
        y = vectorized._max_pool2d_vectorized(
            x, kernel_size=3, stride=1)
        dx = vectorized._max_pool2d_gradient_vectorized(
            x, kernel_size=3, outGrad=dy)

        want_y = xp.array([
            [
                [6,5,6],
                [6,5,6],
                [9,9,7]
            ],
            [
                [6,5,6],
                [6,5,6],
                [9,9,7]
            ]
        ])
        want_dx = xp.array([
            [
                [0,  0,  0,  0,  0],
                [2,  2,  0,  0,  0],
                [0,  0,  0,  0,  2],
                [0,  0,  0,  0,  0],
                [0,  2,  0,  0,  1],
            ],
            [
                [0,  0,  0,  0,  0],
                [2,  2,  0,  0,  0],
                [0,  0,  0,  0,  2],
                [0,  0,  0,  0,  0],
                [0,  2,  0,  0,  1],
            ]
        ])
        self.assertEqual(y.shape, (2,3,3))
        self.assertTrue(xp.allclose(y, want_y))
        self.assertEqual(dx.shape, x.shape)
        self.assertTrue(xp.allclose(dx, want_dx))

    

if __name__ == '__main__':
    unittest.main()

