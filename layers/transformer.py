import numpy
import autograd2 as ag
from loss import *
from .module import Module


def np_normal(shape):
    return xp.random.normal(scale=(2.0/numpy.prod(shape)), size=shape)

class Embedding(Module):
    def __init__(self, vocab_size, embed_dims):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dims = embed_dims
        self.w = ag.Tensor(np_normal((vocab_size, embed_dims)), requires_grad=True)
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

        w = np_normal((seq_len, embed_size))
        for pos in range(seq_len):
            for k in range(embed_size):
                if k%2 == 0:
                    z1 = xp.sin(pos / xp.power(10000, (2*k) // self.embed_size))
                    w[pos][k] = z1 
                else:
                    z2 = xp.cos(pos / xp.power(10000, (2*k) // self.embed_size))
                    w[pos][k] = z2
        self.w = ag.Tensor(w, requires_grad=True)
        self.w.set_name("PositionalEncoding.w")

        self.params = [self.w]

    def forward(self, x):
        # b,n,k = x.shape
        return ag.add(x, self.w)
    
class LayerNorm2(Module):
    def __init__(self, features_shape, eps=1e-8):
        self.features_shape = features_shape
        self.w = ag.Tensor(xp.ones(features_shape), requires_grad=True)
        self.b = ag.Tensor(xp.zeros(features_shape), requires_grad=True)
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
    

class Linear(Module):
    def __init__(self, input_embed, output_embed):
        super().__init__()
        self.input_embed = input_embed
        self.output_embed = output_embed

        total_size = input_embed * output_embed
        w = xp.random.normal(scale=(2.0/total_size), size=(input_embed, output_embed))
        b = xp.random.normal(scale=(2.0/total_size), size=(1, 1))
        self.w = ag.Tensor(w, requires_grad=True)
        self.b = ag.Tensor(b, requires_grad=True)
        self.params = [self.w, self.b]

    def forward(self, x):
        return ag.batch_matmul(x, self.w) + self.b
    
class FeedForward(Module):
    def __init__(self, input_embed, inner_embed, output_embed):
        super().__init__()
        self.input_embed = input_embed
        self.inner_embed = inner_embed
        self.output_embed = output_embed

        self.dense1 = Linear(input_embed, inner_embed)
        self.dense2 = Linear(inner_embed, output_embed)
        self.params = [self.dense1, self.dense2]

    def forward(self, x):
        assert x.ndim == 3
        y = self.dense1(x)
        y = ag.relu(y)
        y = self.dense2(y)
        return y

class SimpleSelfAttention(Module):

    def __init__(self, embed_dims, dim=None):
        super().__init__()
        self.emebedding_size = embed_dims
        if dim is None:
            dim = embed_dims
        self.dim = dim

        k = np_normal((embed_dims, dim))
        kb = np_normal((1,1))
        q = np_normal((embed_dims, dim))
        qb = np_normal((1,1))
        v = np_normal((embed_dims, dim))
        vb = np_normal((1,1))
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
        w = score / xp.sqrt(k)               # (n,n)
        w = ag.softmax(w)                    # (n,n)
        w.set_name("w")

        y = ag.batch_matmul(w, value) # (n,n)*(n,k) => (n,k)
        return y

# Creates a mask which masks out the future token positions
# (seq_len, seq_len) returns a top-right diagonal matrix of 1's
def create_mask_of_future_positions(seq_len):
    mask = xp.ones((seq_len, seq_len))
    mask = xp.tril(mask) == 0
    return mask
    # mask = xp.ones((seq_len, seq_len))
    # mask = xp.triu(mask) > 0
    # return mask

def extend_mask(in_mask, seq_len):
    # b,n,..., seq_len
    mask = xp.reshape(in_mask, (-1, 1, seq_len))
    mask = xp.broadcast_to(mask, in_mask.shape + (seq_len,))
    return mask

# single pass of attention 
#  query: (*, seq_len, dim_keyquery)
#  key  : (*, dim_keyquery, seq_len)
#  value: (*, seq_len, dim_value)
#  mask : (seq_len, seq_len)
def attention(query, key, value, mask=None):
    querykey_size = query.shape[-1]
    y = ag.batch_matmul(query, key)         # (*, seq_len, seq_len)
    y = y / xp.sqrt(querykey_size)          # (*, seq_len, seq_len)
    if mask is not None:
        y = ag.mask_fill(y, mask, -xp.inf)  # (*, seq_len, seq_len)
    y = ag.softmax(y, axis=(-2,-1))         # (*, seq_len, seq_len)
    y = ag.batch_matmul(y, value)           # (*, seq_len, dim_value)
    return y
    
class MultiHeadSelfAttention(Module):

    def __init__(self, embed_dims, dim_keyquery=None, dim_value=None, num_heads=1, mask=None):
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
        self.mask = mask

        assert self.dim_query % self.num_heads == 0
        assert self.dim_key % self.num_heads == 0
        assert self.dim_value % self.num_heads == 0

        self.query = Linear(self.embed_dims, self.dim_query)
        self.key = Linear(self.embed_dims, self.dim_key)
        self.value = Linear(self.embed_dims, self.dim_value)
        self.linear = Linear(self.dim_value, self.dim_value)
        self.params = [self.query, self.key, self.value, self.linear]

    def forward(self, x_query, x_key, x_value, x_mask:ag.Tensor=None):
        assert x_query.ndim == 3
        assert x_key.ndim == 3
        assert x_value.ndim == 3
        b, n, _ = x_query.shape

        query = self.query.forward(x_query)   # (b,n,d_kq)
        key = self.key.forward(x_key)         # (b,n,d_kq)
        value = self.value.forward(x_value)   # (b,n,d_v)
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

        # merge the masks
        mask = self.mask                                        # (n,n)
        if x_mask is not None:
            assert isinstance(x_mask, ag.Tensor)
            x_mask = x_mask.value()                             # (b,n)
            x_mask = extend_mask(x_mask, n)                     # (b,n,n)
            assert x_mask.shape == (b,n,n)
            mask = mask.value() | x_mask                        # (b,n,n)
            mask = xp.reshape(mask, (b,1,n,n))                  # (b,h,n,n)
            mask = ag.Tensor(mask)

        z = attention(
            query_split, 
            key_split, 
            value_split, 
            mask=mask
        )                                                       # (b,h,n,dv/h)                                          
        z.set_name("score")
        z = ag.transpose(z, axis=(0,2,1,3))                     # (b,n,h,dv/h)
        z.set_name("score_transposed")
        z = ag.reshape(z, (b, n, self.dim_value))               # (b,n,dv)
        z.set_name("score_concatenated")
        z = self.linear(z)                                      # (b,n,dv)
        z.set_name("MultiHeadSelfAttention.z")

        return z


# TODO: Re-use the ResidualLayer class in the Encoder/Decoder Layers
class ResidualLayer(Module):
    def __init__(self, sub_layer: Module):
        super().__init__()
        self.sub_layer = sub_layer
        self.params = [sub_layer]

    def forward(self, x):
        y = self.sub_layer.forward(x)
        y = y + x
        return y


# TODO: Allow creating the transformer with different 'value' dimensions
# This will requires changes to the Encoder/Decoder classes so that
# the cross attention in the DecoderTransformer will work
# will also require updating how the residual connections will work due to 
# mismatching dimensions
class EncoderBlock(Module):
    def __init__(
        self,
        seq_len,
        embed_dims,
        num_heads=8,
        feedforward_dims=64
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dims = embed_dims
        self.feedforward_dims = feedforward_dims
        self.num_heads = num_heads

        self.attention = MultiHeadSelfAttention(
            embed_dims, 
            dim_keyquery=embed_dims,
            dim_value=embed_dims,
            num_heads=num_heads,
        )
        self.norm1 = LayerNorm2((embed_dims,))
        self.feedforward = FeedForward(
            self.embed_dims, 
            self.feedforward_dims,
            self.embed_dims)
        self.norm2 = LayerNorm2((embed_dims,))
        
        self.params = [
            self.norm1,
            self.attention,
            self.norm2,
            self.feedforward,
        ]

    def forward(self, x):
        # b, n, k = x.shape
        r = x
        y = self.norm1.forward(x)            # (b,n,d_v)
        y = self.attention.forward(y, y, y)  # (b,n,d_v)
        y = y + r                            # (b,n,d_v)

        r = y
        y = self.norm2.forward(y)            # (b,n,d_v)
        y = self.feedforward.forward(y)      # (b,n,d_v)
        y = y + r
        return y
                        

class DecoderBlock(Module):
    # TODO: Add params for dim_keyquery and dim_value

    def __init__(
            self,
            seq_len,
            embed_dims,
            num_heads=8,
            ff_dims=None,
            include_encoder_attention=False
        ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.feedforward_dims = ff_dims if ff_dims is not None else 4*embed_dims
        self.include_encoder_attention = include_encoder_attention
        self.mask = ag.Tensor(create_mask_of_future_positions(self.seq_len))

        self.norm1 = LayerNorm2((embed_dims,))
        self.masked_attention = MultiHeadSelfAttention(
            embed_dims, 
            dim_keyquery=embed_dims,
            dim_value=embed_dims,
            num_heads=num_heads,
            mask=self.mask
        )

        self.norm2 = None
        self.encoder_attention = None
        
        self.norm3 = LayerNorm2((embed_dims,))
        self.feedforward = FeedForward(
            self.embed_dims, 
            self.feedforward_dims,
            self.embed_dims)

        self.params = [
            self.norm1,
            self.masked_attention,
            self.norm3,
            self.feedforward,
        ]

        if self.include_encoder_attention:
            self.norm2 = LayerNorm2((embed_dims,))
            self.encoder_attention = MultiHeadSelfAttention(
                embed_dims, 
                dim_keyquery=embed_dims,
                dim_value=embed_dims,
                num_heads=num_heads
            )
            self.params += [
                self.norm2,
                self.encoder_attention,
            ]        

    def forward(self, x, memory=None, x_mask=None):
        # b, n, k = x.shape

        # masked multi-attention with residual
        r = x                                                     # (b,n,k)
        y = self.norm1.forward(x)                                 # (b,n,k)
        y = self.masked_attention.forward(y,y,y, x_mask=x_mask)   # (b,n,d_v)
        y = y + r                                                 # (b,n,d_v)

        # multi-attention with residual, with key,value from encoder
        if self.include_encoder_attention:
            assert memory is not None
            r = y                                                 # (b,n,d_v)
            y = self.norm2.forward(y)                             # (b,n,d_v)
            y = self.encoder_attention.forward(y, memory, memory) # (b,n,d_v)
            y = y + r                                             # (b,n,d_v)

        # feed-forward and norm with residual
        r = y                                                     # (b,n,d_v)
        y = self.norm3.forward(y)                                 # (b,n,d_v)
        y = self.feedforward.forward(y)                           # (b,n,d_v)
        y = y + r                                                 # (b,n,d_v)

        return y                                                  # (b,n,d_v)
