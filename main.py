import os
import numpy as np

from typing import *
from pprint import pprint
import utils
import autograd2 as ag
from layers import *
from dataloader.shakespeare import ShakespeareDataLoader
from tokenizer import Tokenizer


class Encoder(Sequence):
    def __init__(self, num_layers, seq_len, embed_dims, num_heads=8):
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.embed_dims = embed_dims
        self.num_heads = num_heads

        layers = []
        for _ in range(num_layers):
            t = EncoderLayer(
                self.seq_len,
                self.embed_dims,
                num_heads=self.num_heads,
            )
            layers.append(t)
        layers.append(LayerNorm2((self.embed_dims,)))
        super().__init__(layers)
    
class Decoder(Module):
    def __init__(self, num_layers, seq_len, embed_dims, num_heads=8):
        super().__init__()
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.embed_dims = embed_dims
        self.num_heads = num_heads

        self.decoders = []
        for _ in range(num_layers):
            t = DecoderLayer(
                self.seq_len,
                self.embed_dims,
                num_heads=self.num_heads,
            )
            self.decoders.append(t)
        self.norm = LayerNorm2((self.embed_dims,))

        self.params = self.decoders + [self.norm]

    def forward(self, x, memory):
        for layer in self.decoders:
            x = layer.forward(x, memory)
        x = self.norm.forward(x)
        return x

class Transformer(Module):
    # TODO: Make this more general and allow users to pass in the encoder/decoder submodels
    # TODO: Make an abstract EncoderDecoder model which transform should sub-class from.
    def __init__(
            self, 
            seq_len, 
            source_vocab, 
            target_vocab=None,
            embed_dims=64,
            encoder_layers=2,
            decoder_layers=2,
            attention_heads=4,
        ):
        super().__init__()
        self.seq_len = seq_len
        self.source_vocab = source_vocab
        self.target_vocab = source_vocab if target_vocab is None else target_vocab
        self.embed_dims = embed_dims
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.attention_heads = attention_heads

        self.source_embedding = Sequence([
            Embedding(len(self.source_vocab), self.embed_dims),
            PositionalEncoding(seq_len, self.embed_dims),
        ])
        self.target_embedding = Sequence([
            Embedding(len(self.target_vocab), self.embed_dims),
            PositionalEncoding(seq_len, self.embed_dims),
        ])

        self.encoder = Encoder(
            self.encoder_layers, 
            seq_len=seq_len, 
            embed_dims=embed_dims,
            num_heads=self.attention_heads,
        )
        self.decoder = Decoder(
            self.decoder_layers, 
            seq_len=seq_len, 
            embed_dims=embed_dims,
            num_heads=self.attention_heads,
        )
        self.generator = Sequence([
            Linear(self.embed_dims, len(self.target_vocab)),
            # TODO: softmax must be done over all the inputs?
            # or just retrieve the last one?
            LogSoftmax()
        ])

    def forward(self, x):
        # embeddings
        src = self.source_embedding.forward(x)         # (b,n,d_embed)
        target = self.target_embedding.forward(x)      # (b,n,d_embed)

        # encoder/decoder
        memory = self.encoder.forward(src)             # (b,n,d_v)
        z = self.decoder.forward(target, memory)       # (b,n,d_v2)

        # Convert to logits
        z = self.generator.forward(z)                  # (b,d_vocab)
        return z
    
def get_batches(tokenizer: Tokenizer, lines, seq_len, batch_size):
    seq_len2 = seq_len - 1 # leave room for the EOS token

    batch = []
    last_line = ""
    for next_line in lines:
        last_line += next_line

        # split last_line into batches of seq_len
        line_batch = []
        while len(last_line) >= seq_len2:
            line_batch.append(last_line[:seq_len2])
            last_line = last_line[seq_len2:]

        for line in line_batch:
            line = list(line)
            line = tokenizer.pad_line(line, seq_len)
            x = tokenizer.encode(line)
            
            assert x.shape == (seq_len, len(tokenizer.vocab))
            batch.append(x)

            if len(batch) >= batch_size:
                yield np.array(batch)
                batch = []

    if len(last_line) > 0:
        last_line = list(last_line)
        last_line = tokenizer.pad_line(last_line, seq_len)
        x = tokenizer.encode(last_line)
        assert x.shape == (seq_len, len(tokenizer.vocab))
        batch.append(x)

    if len(batch) > 0:
        yield np.array(batch)
    
def main():
    dl = ShakespeareDataLoader("data/shakespeare.txt")
    dl.load_data()

    num_batches = 2
    seq_len = 10
    tok = Tokenizer(dl.vocab)
    batches = get_batches(tok , dl.x_train, seq_len, num_batches)
    b1 = next(batches)
    print(b1.shape)

    embed_dims = 32
    model = Transformer(seq_len, tok.vocab, embed_dims=embed_dims)
    y = model.forward(ag.Tensor(b1))
    y.backward()
    print(y.shape)
    ag.render_graphviz(y)

    # encoder = Encoder(2, seq_len, embed_dims)
    # decoder = Decoder(2, seq_len, embed_dims)
    # x = ag.Parameter(np.random.rand(1, seq_len, embed_dims))
    # y = encoder.forward(x)
    # y = decoder.forward(y)
    # y.backward()
    # print(y)
    # ag.render_graphviz(y)
    
    # print(dl.x[:100])

if __name__ == "__main__":
    main()
    