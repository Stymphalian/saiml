import numpy as np
import autograd2 as ag
from loss import *
from .module import Module
from .dense import Dense
from .sequence import Sequence


class Embedding(Module):
    def __init__(self, vocab_size, embed_dims):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dims = embed_dims
        self.w = ag.Tensor(np.random.normal(size=(vocab_size, embed_dims)), requires_grad=True)
        self.w.set_name("Embedding.w")
        self.params = [self.w]

    def forward(self, x):
        # b,n,t = x.shape
        return ag.batch_matmul(x, self.w)
    
class PositionalEncoding(Module):
    def __init__(self, seq_len, embed_size):
        super().__init__()
        self.seq_len = seq_len
        self.embed_size = embed_size

        w = np.random.normal(size=(seq_len, embed_size))
        for pos in range(seq_len):
            for k in range(embed_size):
                if k%2 == 0:
                    z1 = np.sin(pos / np.power(10000, (2*k) // self.embed_size))
                    w[pos][k] = z1 
                else:
                    z2 = np.cos(pos / np.power(10000, (2*k) // self.embed_size))
                    w[pos][k] = z2
        self.w = ag.Tensor(w, requires_grad=True)
        self.w.set_name("PositionalEncoding.w")

        self.params = [self.w]

    def forward(self, x):
        # b,n,k = x.shape
        return ag.add(x, self.w)
    
class LayerNorm(Module):
    EPSILON = ag.Tensor(1e-8)

    def __init__(self, features_shape):
        self.features_shape = features_shape
        self.w = ag.Tensor(np.ones(features_shape), requires_grad=True)
        self.b = ag.Tensor(np.zeros(features_shape), requires_grad=True)
        self.params = [self.w, self.b]
    
    def forward(self, x):
        batches = x.shape[0]
        reduced_shape = (batches,) + (1,) * (x.ndim - 1)
        axes = tuple(range(1, x.ndim))

        mean = ag.mean(x, axis=axes).reshape(reduced_shape)
        u = x - mean
        var = ag.mean(u*u, axis=axes)
        stddev = ag.sqrt(var).reshape(reduced_shape)

        z = u * self.w
        z = z / (stddev + self.EPSILON)
        z = z + self.b
        return z
    
class FeedForward(Module):
    def __init__(self, input_embed, output_embed):
        super().__init__()
        self.input_embed = input_embed
        self.output_embed = output_embed
        self.w1 = ag.Tensor(np.random.normal(size=(input_embed, output_embed)), requires_grad=True)
        self.b1 = ag.Tensor(np.random.normal(size=(1, 1)), requires_grad=True)
        self.w1.set_name("FeedForward.w1")
        self.b1.set_name("FeedForward.b1")
        self.params = [self.w1, self.b1]

    def forward(self, x):
        assert x.ndim == 3
        z1 = ag.einsum("Bni,io->Bno", x, self.w1)
        z2 = z1 + self.b1
        z3 = ag.relu(z2)
        return z3

class SelfAttention(Module):
    # TODO : multihead attention?

    def __init__(self, embed_dims, dim=None):
        super().__init__()
        self.emebedding_size = embed_dims
        if dim is None:
            dim = embed_dims
        self.dim = dim

        k = np.random.normal(size=(embed_dims, dim))
        kb = np.random.normal(size=(1,1))
        q = np.random.normal(size=(embed_dims, dim))
        qb = np.random.normal(size=(1,1))
        v = np.random.normal(size=(embed_dims, dim))
        vb = np.random.normal(size=(1,1))
        self.k = ag.Tensor(k, requires_grad=True, name="k")
        self.kb = ag.Tensor(kb, requires_grad=True, name="kb")
        self.q = ag.Tensor(q, requires_grad=True, name="q")
        self.qb = ag.Tensor(qb, requires_grad=True, name="qb")
        self.v = ag.Tensor(v, requires_grad=True, name="v")
        self.vb = ag.Tensor(vb, requires_grad=True, name="vb")
        self.params = [
            self.k, self.kb,
            self.q, self.qb,
            self.v, self.vb
        ]

    def forward(self, x):
        assert x.ndim == 3
        b, n, k = x.shape

        # (n,k)*(k,n)
        key = ag.batch_matmul(x, self.k) + self.kb
        key.set_name("key")
        query = ag.batch_matmul(x, self.q) + self.qb
        query = ag.transpose(query, axis=(0,2,1))
        query.set_name("query")
        value = ag.batch_matmul(x, self.v) + self.vb
        value.set_name("value")

        score = ag.batch_matmul(key, query)  # (n,n)
        w = score / np.sqrt(k)               # (n,n)
        w = ag.softmax(w)                   # (n,n)
        w.set_name("w")

        y = ag.batch_matmul(w, value) # (n,n)*(n,k) => (n,k)
        return y


class ResidualLayer(Module):
    def __init__(self, sub_layer: Module):
        super().__init__()
        self.sub_layer = sub_layer
        self.params = [sub_layer]

    def forward(self, x):
        y = self.sub_layer.forward(x)
        return y + x

class Transformer(Module):

    def __init__(self, seq_len, embed_dims):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dims = embed_dims
        self.attention = ResidualLayer(
            Sequence([
                LayerNorm((seq_len, embed_dims)),
                SelfAttention(embed_dims)
            ])
        )
        self.linear = ResidualLayer(
            Sequence([
                LayerNorm((seq_len, embed_dims)),
                FeedForward(embed_dims, embed_dims)
            ])
        )

        self.params = [
            self.attention,
            self.linear
            # self.layernorm1,
            # self.attention,
            # self.layernorm2,
            # self.linear
        ]

    def forward(self, x):
        # b, n, k = x.shape
        y = self.attention(x)  # b,n,k
        y = self.linear(x)     # b,n,k
        return y
                        

