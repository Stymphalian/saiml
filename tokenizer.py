import numpy as np

class Tokenizer:
    EOS = "<EOS>"  # end of sequence
    PAD = "<PAD>"  # padding (for empty parts of the sequence)

    def __init__(self, vocab):
        self.extra = [self.PAD, self.EOS]
        self.vocab = np.array(self.extra + list(vocab) )
        self.char_to_index = {}
        self.index_to_char = {}
        for ci, c in enumerate(self.vocab):
            self.char_to_index[c] = ci
            self.index_to_char[ci] = c

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    def pad_line(self, line, seq_len):
        assert isinstance(line, list)
        assert len(line) <= seq_len - 1    # room for the EOS token
        while len(line) < seq_len-1:
            line.append(self.PAD)
        line.append(self.EOS)
        return line
    
    def strip(self, line):
        return [c for c in line if c not in self.extra]
    
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