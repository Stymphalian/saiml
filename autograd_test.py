import unittest
import numpy as np
from pprint import pprint
import autograd as ag

class TestAutograd(unittest.TestCase):

    # def setUp(self):
    #     self.nodes = [ag.Node() for _ in range(7)]        
    #     v1 = self.nodes[0]
    #     v2 = self.nodes[1]
    #     v3 = self.nodes[2]
    #     v4 = self.nodes[3]
    #     v5 = self.nodes[4]
    #     v6 = self.nodes[5]
    #     v7 = self.nodes[6]
    #     v3.parents = [v1]
    #     v4.parents = [v1, v2]
    #     v5.parents = [v2]
    #     v6.parents = [v3, v4]
    #     v7.parents = [v5, v6]

    # def test_toposort(self):
    #     v7 = self.nodes[6]
    #     got = list(ag.toposort(v7))
    #     got = [x.id for x in got]

    #     for x in self.nodes:
    #         current_index = got.index(x.id)
    #         for y in x.parents:
    #             self.assertTrue(current_index < got.index(y.id))

    # def test_forward_defn(self):
    #     v1 = ag.constant(2)
    #     v2 = ag.constant(5)
    #     v3 = ag.log(v1)
    #     v4 = ag.add(v1, v2)
    #     v5 = ag.sin(v2)
    #     v6 = ag.add(v3, v4)
    #     v7 = ag.sub(v6, v5)

    # def test_toposort(self):
    #     x1 = ag.constant(2)
    #     x2 = ag.constant(5)
    #     v7 = ag.log(x1) + x1*x2 - ag.sin(x2)
    #     nodes = list(ag.toposort(v7))
    #     got = [x.id for x in nodes]

    #     for x in nodes:
    #         current_index = got.index(x.id)
    #         for y in x.parents:
    #             self.assertTrue(current_index < got.index(y.id))

    def test_gradient(self):
        x1 = ag.constant(2)
        x2 = ag.constant(5)
        v7 = ag.log(x1) + x1*x2 - ag.sin(x2)

        pred = ag.value(v7)
        grads = ag.gradient(v7)
        self.assertAlmostEquals(pred, 11.652071455223084)
        self.assertAlmostEquals(sum(grads[x1.id]), 5.5)
        self.assertAlmostEquals(sum(grads[x2.id]), 1.7163378)
        


if __name__ == '__main__':
    unittest.main()
