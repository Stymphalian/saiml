import unittest
import numpy as np
import utils
from tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        vocab = list("abcde")
        tok = Tokenizer(vocab)
        self.vocab = vocab
        self.tok = tok

    def test_tokenizer(self):
        line = "abd"
        x = self.tok.encode(line)
        self.assertEqual(x.shape, (3, len(self.tok.vocab))) 
        y = self.tok.decode(x)
        self.assertEqual(y, list(line))   

    def test_pad_line(self):
        line = "abd"
        seq_len = 5
        x = self.tok.pad_line(list(line), seq_len)
        self.assertEqual(x, ["a", "b", "d", self.tok.PAD, self.tok.EOS])
        y = self.tok.strip(x)
        self.assertEqual(y, ["a", "b", "d"])
    

if __name__ == '__main__':
    unittest.main()

