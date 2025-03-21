import unittest
from pprint import pprint
import autograd2 as ag
import numpy as np

class TestAutogradUtils(unittest.TestCase):
        
    def test_toposort(self):
        x1 = ag.Tensor(np.array([2]))
        x2 = ag.Tensor(np.array([5]))
        v7 = ag.log(x1) + x1*x2 - ag.sin(x2)
        nodes = list(ag.toposort(v7))
        got = [x.id for x in nodes]

        for x in nodes:
            current_index = got.index(x.id)
            for y in x.inputs:
                self.assertTrue(current_index < got.index(y.id))


if __name__ == '__main__':
    unittest.main()
