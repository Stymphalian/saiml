
import unittest
from tokenizer import BytePairEncoder
from pprint import pprint

class BPETest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open("tokenizer/test_text.txt", encoding="utf-8") as f:
            cls.TEST_TEXT = f.read()
    
    def setUp(self):
        self.bpe = BytePairEncoder()
        self.bpe.fit(self.TEST_TEXT, 20)

    def test_find_byte_pair_counts(self):
        text = "hello worrld  "
        byte_text = self.bpe.to_ints(text)
        counts = self.bpe.find_byte_pair_counts(byte_text)

        l = ord('l')
        r = ord('r')
        self.assertEquals(counts[(l,l)], 1)
        self.assertEquals(counts[(r,r)], 1)

    def test_replace_pairs_with_token(self):
        text = "same same but different"
        byte_text = self.bpe.to_ints(text)
        pair = (ord('m'), ord('e'))
        replacement = 257
        new_text = self.bpe.replace_pairs_with_token(byte_text, pair, replacement)

    def test_fit(self):
        desired_vocab_size = 276
        num_merges = desired_vocab_size - 256
        # num_merges = 20
        vocab, pair_to_index = self.bpe.fit(self.TEST_TEXT, num_merges)
        self.assertEqual(len(vocab), 276)
        self.assertEqual(len(pair_to_index), num_merges)

    def test_decode(self):
        encoded_text = [65, 66, 67]
        decoded_text = self.bpe.decode(encoded_text)
        self.assertEqual(decoded_text, "ABC")

        encoded_text = [128]
        decoded_text = self.bpe.decode(encoded_text)

    def test_encode(self):
        line = "hello world"
        encoded_text = self.bpe.encode(line)
        decoded_text = self.bpe.decode(encoded_text)
        self.assertEqual(line, decoded_text)

        line = "Many common characters, including numerals, punctuation, and other symbols, are unified within the standard and are not treated as specific to any given writing system. Unicode encodes thousands of emoji, with the continued development thereof conducted by the Consortium as a part of the standard.[4] Moreover, the widespread adoption of Unicode was in large part responsible for the initial popularization of emoji outside of Japan. Unicode is ultimately capable of encoding more than 1.1 million characters."
        encoded_text = self.bpe.encode(line)
        decoded_text = self.bpe.decode(encoded_text)
        self.assertEqual(line, decoded_text)


        
        


if __name__ == '__main__':
    unittest.main()