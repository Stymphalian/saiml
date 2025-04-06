import numpy as np

class Tokenizer:
    def __init__(self, vocab):
        self.vocab = np.array(vocab)
        self.char_to_index = {}
        self.index_to_char = {}
        for ci, c in enumerate(vocab):
            self.char_to_index[c] = ci
            self.index_to_char[ci] = c

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, line):
        x = []
        onehots = np.identity(self.vocab.size)
        for c in line:
            ci = self.char_to_index[c]
            x.append(onehots[ci])
        x = np.array(x)
        return x
    
    def decode(self, x):
        indices = np.argmax(x, axis=1)
        line = []
        for ci in indices:
            c = self.index_to_char[ci]
            line.append(c)
        return line