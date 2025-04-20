import unittest
from devices import xp
import autograd2 as ag
from .dropout import *

class TestDropout(unittest.TestCase):
    def test_dropout(self):
        xp.random.seed(0)
        layer = Dropout(True, p=0.5, rng_seed=3)
        x1 = xp.arange(2*3).reshape(2,3) + 1
        x = ag.Tensor(x1, requires_grad=True)

        got = layer.forward(x)
        if xp.__name__ == 'cupy':
            want = xp.array([
                [1, 0, 3],
                [0, 5, 6]
            ])
        else:
            want = xp.array([
                [0, 0, 3],
                [4, 0, 0]
            ])
        self.assertEqual(got.shape, want.shape)
        self.assertTrue(xp.array_equal(got.value(), want))
        got.backward()

if __name__ == '__main__':
    unittest.main()
