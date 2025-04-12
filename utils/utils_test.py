import unittest
from devices import xp
import utils
from scipy import signal


class TestUtils(unittest.TestCase):

    def test_zero_pad_shape(self):
        xp.random.seed(1)
        x = xp.random.randn(4, 2, 3, 3)
        x_pad = utils.zero_pad2(x, 2)
        self.assertEqual(x_pad.shape, (4, 2, 7, 7))

    def test_zero_pad(self):
        image1 = xp.array([
            xp.ones((3,3)),
            xp.ones((3,3))*2
        ])
        image2 = xp.array([
            xp.ones((3,3))*3,
            xp.ones((3,3))*4
        ])
        X = xp.array([image1, image2])

        image1 = xp.array([
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
        image2 = xp.array([
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
        want = xp.array([image1, image2])
        got = utils.zero_pad2(X, 2)
        self.assertTrue(xp.array_equal(got, want))

    def test_general_dilate(self):
        x = xp.arange(2*9).reshape(2,3,3) + 1
        want = xp.array([
            [
                [1,0,2,0,3],
                [0,0,0,0,0],
                [4,0,5,0,6],
                [0,0,0,0,0],
                [7,0,8,0,9],
            ],
            [
                [10, 0,  11, 0,  12],  
                [0,  0,  0,  0,  0],  
                [13, 0,  14, 0,  15],  
                [0,  0,  0,  0,  0],  
                [16, 0,  17, 0,  18],
            ],
            
        ])
        got = utils.zero_dilate(x, 1, axes=(1,2))
        self.assertTrue(xp.array_equal(got, want))


    def test_dilate(self):
        X = xp.arange(9).reshape(3,3) + 1
        want = xp.array([
            [1,0,2,0,3],
            [0,0,0,0,0],
            [4,0,5,0,6],
            [0,0,0,0,0],
            [7,0,8,0,9],
        ])
        got = utils.zero_dilate_2d(X, 1)
        self.assertTrue(xp.array_equal(got, want))

    def test_dilate2(self):
        X = xp.arange(9).reshape(3,3) + 1
        want = xp.array([
            [1,0,0,2,0,0,3],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [4,0,0,5,0,0,6],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [7,0,0,8,0,0,9],
        ])
        got = utils.zero_dilate_2d(X, 2)
        self.assertTrue(xp.array_equal(got, want))


    def test_dilate_not_symmetric(self):
        X = xp.arange(15).reshape(3,5) + 1
        want = xp.array([
            [1,  0,  2,  0,  3,  0,  4,  0,  5],  
            [0,  0,  0,  0,  0,  0,  0,  0,  0],  
            [6,  0,  7,  0,  8,  0,  9,  0,  10],  
            [0,  0,  0,  0,  0,  0,  0,  0,  0],  
            [11, 0,  12, 0,  13, 0,  14, 0,  15],
        ])
        got = utils.zero_dilate_2d(X, 1)
        self.assertTrue(xp.array_equal(got, want))

    def test_dilate2d_2(self):
        X = xp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=xp.float64)
        dilated = utils.zero_dilate_2d(X, 3)
        expected = xp.array([
            [1, 0, 0, 0, 2, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 5, 0, 0, 0, 6],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [7, 0, 0, 0, 8, 0, 0, 0, 9],
        ], dtype=xp.float64)
        self.assertTrue(xp.array_equal(dilated, expected))

        undilated = utils.zero_undilate_2d(dilated, 3)
        self.assertTrue(xp.array_equal(undilated, X))
    

if __name__ == '__main__':
    unittest.main()

