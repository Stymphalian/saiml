import unittest
from layers import Module
import autograd2 as ag
import numpy as np

class _TestModule(Module):
    def __init__(self):
        self.w1 = ag.Tensor(np.random.rand(3,1), requires_grad=True) # 3
        self.w2 = ag.Tensor(np.random.rand(3,2), requires_grad=True) # 9
        self.w3 = ag.Tensor(np.random.rand(3,3), requires_grad=True) # 18
        self.w4 = ag.Tensor(np.random.rand(3,4), requires_grad=True) # 30
        self.w5 = ag.Tensor(np.random.rand(3,5), requires_grad=True) # 45
        self.w6 = ag.Tensor(np.random.rand(3,6), requires_grad=True) # 63
        
        self.dict = {'w2': self.w2, 'w3': self.w3}
        self.list = [self.w4, self.w5, self.w6]
        self.params = [
            self.w1, 
            self.dict, 
            self.list
        ]

class TestModule(unittest.TestCase):
    def test_unpack_params(self):
        np.random.seed(1)
        m = _TestModule()
        got = m.get_params()
        want = np.concatenate([
            m.w1.value().reshape(-1),
            m.w2.value().reshape(-1),
            m.w3.value().reshape(-1),
            m.w4.value().reshape(-1),
            m.w5.value().reshape(-1),
            m.w6.value().reshape(-1)
        ])
        self.assertTrue(np.array_equal(got, want))

    def test_pack_params(self):
        np.random.seed(1)
        m1 = _TestModule()
        m2 = _TestModule()
        m1.params.append(m2)

        got = m1.get_params()
        got[0] = 100  # set m1.w1
        got[4] = 200  # set m1.dict[w1][1]
        got[20] = 300 # set m1.list[0][2]
        got[63] = 400 # set m2.w1[0]
        m1.set_params(got)
        self.assertEqual(m1.w1.value()[0][0], 100)
        self.assertEqual(m1.w2.value()[0][1], 200)
        self.assertEqual(m1.w4.value()[0][2], 300)
        self.assertEqual(m2.w1.value()[0][0], 400)

        # print("m1.w1", m1.w1)
        # print("m1.w2", m1.w2)
        # print("m1.w3", m1.w3)
        # print("m1.w4", m1.w4)
        # print("m1.w5", m1.w5)
        # print("m1.w6", m1.w6)
        # print("m2.w1", m2.w1)
        # print("m2.w2", m2.w2)
        # print("m2.w3", m2.w3)
        # print("m2.w4", m2.w4)
        # print("m2.w5", m2.w5)
        # print("m2.w6", m2.w6)
        


if __name__ == '__main__':
    unittest.main()
