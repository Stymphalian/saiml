import numpy as np

class Tokenizer:
    PAD = "<PAD>"  # padding (for empty parts of the sequence)
    BEGIN = "<BEGIN>"  # start of sequence
    END = "<END>"  # end of sequence

    def __init__(self, vocab):
        self.extra = [self.PAD, self.BEGIN, self.END]
        self.vocab = np.array(self.extra + list(vocab) )
        self.char_to_index = {}
        self.index_to_char = {}
        for ci, c in enumerate(self.vocab):
            self.char_to_index[c] = ci
            self.index_to_char[ci] = c

    @property
    def vocab_size(self):
        return len(self.vocab)
    @property
    def PAD_INDEX(self):
        return self.char_to_index[self.PAD]
    @property
    def BEGIN_INDEX(self):
        return self.char_to_index[self.BEGIN]
    @property
    def END_INDEX(self):
        return self.char_to_index[self.END]
    
    def to_index(self, line):
        return np.array([self.char_to_index[c] for c in line])
    
    def pad_line(self, line, seq_len):
        assert isinstance(line, list)
        assert len(line) <= seq_len - 1    # room for the EOS token
        # line.insert(0, self.BEGIN)
        line.append(self.END)
        while len(line) < seq_len:
            line.append(self.PAD)
        return line
    
    def strip(self, line):
        return [c for c in line if c not in self.extra]
    
    def encode(self, line):
        if isinstance(line[0], str):
            line = self.to_index(line)
        x = []
        onehots = np.identity(self.vocab.size)
        for ci in line:
            # ci = self.char_to_index[c]
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
    

def mask_out_future_positions(seq_len):
    mask = np.ones((seq_len, seq_len))
    mask = np.tril(mask) == 0
    return mask

def mask_out_pad_positions(src, pad_index):
    return (src == pad_index)
    # if src.ndim == 1:
    #     seq_len = src.size
    #     mask = (src == pad_index)
    #     mask = np.broadcast_to(mask, (seq_len, seq_len))
    #     return mask
    # else:
    #     src_shape = src.shape
    #     seq_len = src.shape[-1]

    #     src = np.reshape(src, (-1, 1, seq_len))
    #     src = np.broadcast_to(src, src_shape + (seq_len,))
    #     mask = (src == pad_index)
    #     return mask

def convert_batches_to_numpy_with_mask(batch, tokenizer:Tokenizer, seq_len):
    assert isinstance(batch, list)

    batches = [list(line) for line in batch]
    batches = [tokenizer.pad_line(line, seq_len) for line in batches]
    batches = [tokenizer.to_index(line) for line in batches]

    source_mask = mask_out_pad_positions(np.array(batches), tokenizer.PAD_INDEX)
    source = np.array([tokenizer.encode(line) for line in batches])

    assert source.ndim == 3
    assert source.shape == (len(batch), seq_len, tokenizer.vocab_size)
    # assert source_mask.shape == (len(batch), seq_len, seq_len)
    # assert source_mask.shape == (len(batch), seq_len)
    return source, source_mask

def get_batches(lines, seq_len, batch_size):
    # seq_len2 = seq_len - 1 # leave room for the EOS token
    batch = []
    last_line = ""

    for char in lines:
        last_line += char
        if len(last_line) >= seq_len:
            batch.append(last_line)
            last_line = ""
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if len(last_line) > 0:
        batch.append(last_line)
    if len(batch) > 0:
        yield batch


    # batch = []
    # last_line = ""
    # for next_line in lines:
    #     last_line += next_line

    #     # split last_line into batches of seq_len
    #     line_batch = []
    #     while len(last_line) >= seq_len:
    #         line_batch.append(last_line[:seq_len])
    #         last_line = last_line[seq_len:]

    #     for line in line_batch:
    #         batch.append(list(line))

    #         if len(batch) >= batch_size:
    #             yield batch
    #             batch = []

    # if len(last_line) > 0:
    #     batch.append(list(last_line))

    # if len(batch) > 0:
    #     yield batch