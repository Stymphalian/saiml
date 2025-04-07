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
    def __init__(self, features_shape, eps=1e-8):
        self.features_shape = features_shape
        self.w = ag.Tensor(np.ones(features_shape), requires_grad=True)
        self.b = ag.Tensor(np.zeros(features_shape), requires_grad=True)
        self.params = [self.w, self.b]
        self.epsilon = ag.Tensor(eps)
    
    def forward(self, x):
        index = x.ndim - len(self.features_shape)
        reduced_shape = x.shape[:index] + (1,)*len(self.features_shape)
        axes = tuple(range(index, x.ndim))

        mean = ag.mean(x, axis=axes).reshape(reduced_shape)
        u = x - mean
        var = ag.mean(u*u, axis=axes)
        stddev = ag.sqrt(var).reshape(reduced_shape)

        z = u * self.w
        z = z / (stddev + self.epsilon)
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
        # B == Batch
        # n == sequence length
        # i == input_emebedding_size
        # o == output_embedding_size (typically the same as input_embedding_size)
        z1 = ag.einsum("Bni,io->Bno", x, self.w1)
        z2 = z1 + self.b1
        z3 = ag.relu(z2)
        return z3

class SimpleSelfAttention(Module):

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
        query = ag.batch_matmul(x, self.q) + self.qb
        query.set_name("query")
        key = ag.batch_matmul(x, self.k) + self.kb
        key = ag.transpose(key, axis=(0,2,1))
        key.set_name("key")
        value = ag.batch_matmul(x, self.v) + self.vb
        value.set_name("value")

        score = ag.batch_matmul(query, key)  # (n,n)
        w = score / np.sqrt(k)               # (n,n)
        w = ag.softmax(w)                    # (n,n)
        w.set_name("w")

        y = ag.batch_matmul(w, value) # (n,n)*(n,k) => (n,k)
        return y

class Linear(Module):
    def __init__(self, input_embed, output_embed):
        super().__init__()
        self.input_embed = input_embed
        self.output_embed = output_embed

        total_size = input_embed * output_embed
        w = np.random.normal(scale=(2.0/total_size), size=(input_embed, output_embed))
        b = np.random.normal(scale=(2.0/total_size), size=(1, 1))
        self.w = ag.Tensor(w, requires_grad=True)
        self.b = ag.Tensor(b, requires_grad=True)
        self.params = [self.w, self.b]

    def forward(self, x):
        return ag.batch_matmul(x, self.w) + self.b

# single pass of attention 
#  query: (*, seq_len, dim_keyquery)
#  key  : (*, dim_keyquery, seq_len)
#  value: (*, seq_len, dim_value)
def attention(query, key, value, mask=None):
    querykey_size = query.shape[-1]
    y = ag.batch_matmul(query, key)     # (*, seq_len, seq_len)
    y = y / np.sqrt(querykey_size)      # (*, seq_len, seq_len)
    y = ag.softmax(y, axis=(-2,-1))     # (*, seq_len, seq_len)
    y = ag.batch_matmul(y, value)       # (*, seq_len, dim_value)
    return y
    
class MultiHeadSelfAttention(Module):

    def __init__(self, embed_dims, dim_keyquery=None, dim_value=None, num_heads=1):
        super().__init__()
        if dim_keyquery is None:
            dim_keyquery = embed_dims
        if dim_value is None:
            dim_value = embed_dims
        
        self.embed_dims = embed_dims
        self.dim_query = dim_keyquery
        self.dim_key = dim_keyquery
        self.dim_value = dim_value
        self.num_heads = num_heads

        assert self.dim_query % self.num_heads == 0
        assert self.dim_key % self.num_heads == 0
        assert self.dim_value % self.num_heads == 0

        self.query = Linear(self.embed_dims, self.dim_query)
        self.key = Linear(self.embed_dims, self.dim_key)
        self.value = Linear(self.embed_dims, self.dim_value)
        self.linear = Linear(self.dim_value, self.dim_value)
        self.params = [self.query, self.key, self.value, self.linear]

    def forward(self, x):
        assert x.ndim == 3
        b, n, d = x.shape

        query = self.query.forward(x)   # (b,n,d_kq)
        key = self.key.forward(x)       # (b,n,d_kq)
        value = self.value.forward(x)   # (b,n,d_v)
        query.set_name("query")
        key.set_name("key")
        value.set_name("value")

        dim_query = self.dim_query // self.num_heads
        dim_key = self.dim_key // self.num_heads
        dim_value = self.dim_value // self.num_heads
        query_split = ag.reshape(query, (b, n, self.num_heads, dim_query))
        query_split.set_name("query_split")
        key_split   = ag.reshape(key,   (b, n, self.num_heads, dim_key))
        key_split.set_name("key_split")
        value_split = ag.reshape(value, (b, n, self.num_heads, dim_value))
        value_split.set_name("value_split")

        query_split = ag.transpose(query_split, axis=(0,2,1,3)) # (b,h,n,d_kq/h)
        query_split.set_name("query_split_transposed")
        key_split   = ag.transpose(key_split,   axis=(0,2,3,1)) # (b,h,d_kq/h,n)
        key_split.set_name("key_split_transposed")
        value_split = ag.transpose(value_split, axis=(0,2,1,3)) # (b,h,n,d_v/h)
        value_split.set_name("value_split_transposed")

        z = attention(query_split, key_split, value_split)      # (b,h,n,dv/h)
        z.set_name("score")
        z = ag.transpose(z, axis=(0,2,1,3))                     # (b,n,h,dv/h)
        z.set_name("score_transposed")
        z = ag.reshape(z, (b, n, self.dim_value))               # (b,n,dv)
        z.set_name("score_concatenated")
        z = self.linear(z)                                      # (b,n,dv)
        z.set_name("MultiHeadSelfAttention.z")

        return z


class ResidualLayer(Module):
    def __init__(self, sub_layer: Module):
        super().__init__()
        self.sub_layer = sub_layer
        self.params = [sub_layer]

    def forward(self, x):
        y = self.sub_layer.forward(x)
        y = y + x
        return y

class Transformer(Module):

    def __init__(self, seq_len, embed_dims, num_heads=8):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dims = embed_dims
        self.num_heads = num_heads

        # TODO: The order of the norm and sublayer doesn't seem right to me,
        # the diagram in the paper clearly shows "add + norm" happens AFTER
        # the sublayer operation.
        self.attention = ResidualLayer(
            Sequence([
                LayerNorm((embed_dims,)),
                MultiHeadSelfAttention(embed_dims, num_heads=self.num_heads)
            ])
        )
        self.linear = ResidualLayer(
            Sequence([
                LayerNorm((embed_dims,)),
                FeedForward(embed_dims, embed_dims)
            ])
        )

        self.params = [
            self.attention,
            self.linear
        ]

    def forward(self, x):
        # b, n, k = x.shape
        y = self.attention(x)  # b,n,k
        y = self.linear(y)     # b,n,k
        return y
                        

