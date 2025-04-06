import unittest
from pprint import pprint
from .shakespeare import ShakespeareDataLoader

class ShakespeareDataLoaderTest(unittest.TestCase):

    def test_load_data(self):
        dl = ShakespeareDataLoader("data/shakespeare.txt")
        data = dl.load_data()

        # get alphabet
        # self.assertEqual(len(dl.data), 10000)

if __name__ == '__main__':
    unittest.main()