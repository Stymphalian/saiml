import unittest
import numpy as np
import tokenizer

class TestTokenizer(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        vocab = list("abcde")
        tok = tokenizer.Tokenizer(vocab)
        self.vocab = vocab
        self.tok = tok

    def test_tokenizer(self):
        line = "abd"
        x = self.tok.encode(line)
        y = self.tok.decode(x)
        self.assertEqual(y, line)

    def test_pad_line(self):
        line = "abd"
        seq_len = 5
        x = self.tok.pad_line(list(line), seq_len)
        self.assertEqual(x, ["a", "b", "d", self.tok.END, self.tok.PAD])
        y = self.tok.strip(x)
        self.assertEqual(y, ["a", "b", "d"])

    def test_mask_out_future_positions(self):
        got = tokenizer.mask_out_future_positions(5)
        want = np.array([
            [0,1,1,1,1],
            [0,0,1,1,1],
            [0,0,0,1,1],
            [0,0,0,0,1],
            [0,0,0,0,0]
        ]) == 1
        self.assertTrue(np.allclose(got, want))

    def test_mask_pad_positions(self):
        line = "abc"
        seq_len = 6
        x = self.tok.pad_line(list(line), seq_len)
        x = self.tok.to_index(x)
        got = tokenizer.mask_out_pad_positions(x, self.tok.PAD_INDEX)
        want = np.array([0,0,0,0,1,1]) == 1
        self.assertTrue(np.allclose(got, want))

        batches = ["abc", "deab"]
        seq_len = 6
        batches = [self.tok.pad_line(list(line), seq_len) for line in batches]
        batches = [self.tok.to_index(line) for line in batches]
        batches = np.array(batches)
        got = tokenizer.mask_out_pad_positions(batches, self.tok.PAD_INDEX)
        want = np.array([
            [0,0,0,0,1,1],
            [0,0,0,0,0,1],
        ]) == 1
        self.assertTrue(np.allclose(got, want))

    def test_get_batches(self):
        seq_len = 6
        batch_size = 3
        line = "abcabcabccdeabcdeabd"
        text = self.tok.encode(line)
        x, y = tokenizer.get_batch(text, seq_len, batch_size)
        self.assertTrue(x.shape == y.shape)
        self.assertEqual(x.shape, (3,6))
        self.assertEqual(y.shape, (3,6))
        self.assertTrue(np.array_equal(x[0][1:], y[0][:-1]))
        self.assertTrue(np.array_equal(x[1][1:], y[1][:-1]))

    # def test_convert_batches_to_numpy_with_mask(self):
    #     batch = ["abcd", "bcda", "abc", "a", ""]
    #     batch = [list(line) for line in batch]
    #     seq_len = 6
    #     tokenizer.convert_batches_to_numpy_with_mask(batch, self.tok, seq_len)

if __name__ == '__main__':
    unittest.main()

