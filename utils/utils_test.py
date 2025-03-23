import unittest
import numpy as np
import utils
from scipy import signal


class TestUtils(unittest.TestCase):

    def test_zero_pad_shape(self):
        np.random.seed(1)
        x = np.random.randn(4, 2, 3, 3)
        x_pad = utils.zero_pad(x, 2)
        self.assertEqual(x_pad.shape, (4, 2, 7, 7))

    def test_zero_pad(self):
        image1 = np.array([
            np.ones((3,3)),
            np.ones((3,3))*2
        ])
        image2 = np.array([
            np.ones((3,3))*3,
            np.ones((3,3))*4
        ])
        X = np.array([image1, image2])

        image1 = np.array([
            [
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0],
                [0,0,1,1,1,0,0],
                [0,0,1,1,1,0,0],
                [0,0,1,1,1,0,0],
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0]
            ],
            [
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0],
                [0,0,2,2,2,0,0],
                [0,0,2,2,2,0,0],
                [0,0,2,2,2,0,0],
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0]
            ]
        ])
        image2 = np.array([
            [
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0],
                [0,0,3,3,3,0,0],
                [0,0,3,3,3,0,0],
                [0,0,3,3,3,0,0],
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0]
            ],
            [
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0],
                [0,0,4,4,4,0,0],
                [0,0,4,4,4,0,0],
                [0,0,4,4,4,0,0],
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0]
            ]
        ])
        want = np.array([image1, image2])
        got = utils.zero_pad(X, 2)
        self.assertTrue(np.array_equal(got, want))

    def test_dilate(self):
        X = np.arange(9).reshape(3,3) + 1
        want = np.array([
            [1,0,2,0,3],
            [0,0,0,0,0],
            [4,0,5,0,6],
            [0,0,0,0,0],
            [7,0,8,0,9],
        ])
        got = utils.zero_dilate_2d(X, 1)
        self.assertTrue(np.array_equal(got, want))

    def test_dilate2(self):
        X = np.arange(9).reshape(3,3) + 1
        want = np.array([
            [1,0,0,2,0,0,3],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [4,0,0,5,0,0,6],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [7,0,0,8,0,0,9],
        ])
        got = utils.zero_dilate_2d(X, 2)
        self.assertTrue(np.array_equal(got, want))


    def test_dilate_not_symmetric(self):
        X = np.arange(15).reshape(3,5) + 1
        want = np.array([
            [1,  0,  2,  0,  3,  0,  4,  0,  5],  
            [0,  0,  0,  0,  0,  0,  0,  0,  0],  
            [6,  0,  7,  0,  8,  0,  9,  0,  10],  
            [0,  0,  0,  0,  0,  0,  0,  0,  0],  
            [11, 0,  12, 0,  13, 0,  14, 0,  15],
        ])
        got = utils.zero_dilate_2d(X, 1)
        self.assertTrue(np.array_equal(got, want))

    def test_dilate2d_2(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        dilated = utils.zero_dilate_2d(X, 3)
        expected = np.array([
            [1, 0, 0, 0, 2, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 5, 0, 0, 0, 6],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [7, 0, 0, 0, 8, 0, 0, 0, 9],
        ], dtype=np.float64)
        self.assertTrue(np.array_equal(dilated, expected))

        undilated = utils.zero_undilate_2d(dilated, 3)
        self.assertTrue(np.array_equal(undilated, X))
    

if __name__ == '__main__':
    unittest.main()

