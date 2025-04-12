import unittest
from devices import xp
from .transformer import *
from tokenizer import Tokenizer
from layers import Sequence
import base_gradient_test

class TestTransformers(base_gradient_test.NumericalGradientTest):

    def test_embedding(self):
        vocab = list("abcdefghijklmnopqrstuvwxyz")
        embed_dims = 10
        tokenizer = Tokenizer(vocab)

        x1 = tokenizer.encode("abc")
        x2 = tokenizer.encode("def")
        x = ag.Parameter(xp.array([x1, x2]))
        
        layer = Embedding(tokenizer.vocab_size, embed_dims)
        y = layer.forward(x)
        self.assertEqual(y.shape, (2,3,embed_dims))

    def test_embedding_gradient(self):
        b,n,vocab_size = 2,3,4
        embed_dims = 10
        layer = Embedding(vocab_size,embed_dims)
        x = ag.Parameter(xp.random.normal(size=(b,n,vocab_size)))
        def do():
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        def forward(params):
            self.unravel_params(params, x)
            return do()
        self.numeric_check(forward, x)

    def test_positional_encoding(self):
        xp.random.seed(1)
        batches = 2
        seq_len = 3
        embed_dims = 4
        x = ag.Tensor([
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ],
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ],
        ])
        layer = PositionalEncoding(seq_len, embed_dims)
        y = layer.forward(x)
        y.backward()
        self.assertEqual(y.shape, x.shape)

    def test_positional_encoding_gradient(self):
        batches = 2
        seq_len = 10
        embed_dims = 5
        layer = PositionalEncoding(seq_len, embed_dims)
        x = ag.Parameter(xp.random.normal(size=(batches, seq_len, embed_dims)))
        def do():
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        def forward(params):
            self.unravel_params(params, x)
            return do()
        self.numeric_check(forward, x)

    def test_layer_norm(self):
        xp.random.seed(1)
        batches = 2
        x = ag.arange((batches, 2,3,4,5,6)) + 1.0
        layer = LayerNorm2((x.shape[1:]))
        y = layer.forward(x)
        y.backward()
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(xp.allclose(xp.mean(y.value()[0]), 0.0))
        self.assertTrue(xp.allclose(xp.std(y.value()[0]), 1.0))

    @unittest.skip("Numerically unstable for numeric gradient checking")
    def test_layer_norm_gradient(self):
        batches = 2
        input_shape = (4,3,2)
        layer = LayerNorm2(input_shape)
        x = ag.Parameter(xp.random.rand(batches, *input_shape))
        def do():
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        def forward(params):
            self.unravel_params(params, x)
            return do()
        self.numeric_check(forward, x)

    def test_feed_forward(self):
        xp.random.seed(1)
        batches = 2
        seq_len = 3
        embed_dims = 4
        ff_dims = 10
        output_embed_dims = 5
        x = ag.arange((batches, seq_len, embed_dims)) + 1.0
        layer = FeedForward(embed_dims, ff_dims, output_embed_dims)
        y = layer.forward(x)
        y.backward()
        self.assertEqual(y.shape, (batches, seq_len, output_embed_dims))

    def test_feed_forward_gradient(self):
        xp.random.seed(1)
        batches = 2
        seq_len = 3
        input_dims = 4
        ff_dims = 10
        output_dims = 4
        x = ag.Parameter(xp.random.rand(batches, seq_len, input_dims))
        layer = FeedForward(input_dims, ff_dims, output_dims)
        def do():
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        def forward(params):
            self.unravel_params(params, x)
            return do()
        self.numeric_check(forward, x)

    def test_batch_mult_for_multihead_attention(self):
        b = 1
        n = 2
        h = 3
        k = 4
        x = xp.arange(b*h*n*k, dtype=xp.float64).reshape((b,h,n,k)) + 1.0
        q = xp.round(xp.random.rand(h,k,k), 2)
        q = xp.reshape(q, (b, h, k, k))
        y = xp.einsum("Bhnk,Bhkj->Bhnj", x, q)

        zs = []
        for head in range(h):
            z1 = xp.matmul(x[0,head], q[0,head])
            zs.append(z1)
        z = xp.stack(zs)

        self.assertTrue(xp.allclose(y, z))


    def test_self_attention(self):
        xp.random.seed(1)
        x = ag.arange((2,3,4)) + 1.0
        layer = SimpleSelfAttention(x.shape[2])
        y = layer.forward(x)
        y.backward()
        self.assertEqual(y.shape, x.shape)

    def test_self_attention_gradient(self):
        xp.random.seed(1)
        batches = 2
        seq_len = 3
        embed_dims = 5
        x = ag.Parameter(xp.random.rand(batches, seq_len, embed_dims))
        layer = SimpleSelfAttention(embed_dims)
        def forward(params):
            self.unravel_params(params, x)
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        self.numeric_check(forward, x)

    def test_linear(self):
        layer = Linear(4,3)
        # TODO: Bug with +1.0 on arange, the resulting Tensor doesn't share the requires_grad
        x0 = xp.arange(2*3*4).reshape(2,3,4) + 1.0
        x = ag.Parameter(x0)
        y = layer.forward(x)
        y.backward()
        self.assertEqual(y.shape, (2,3,3))

        want1 = xp.matmul(x0[0], layer.w.value()) + layer.b.value()
        want2 = xp.matmul(x0[1], layer.w.value()) + layer.b.value()
        self.assertTrue(xp.allclose(y.value()[0], want1))
        self.assertTrue(xp.allclose(y.value()[1], want2))

        def forward(params):
            self.unravel_params(params, x)
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        self.numeric_check(forward, x)


    def test_dotproduct_self_attention(self):
        b = 1
        n = 2
        h = 3
        dq = 4
        dv = 5
        
        query = ag.Parameter(xp.random.rand(b,h,n,dq))
        key = ag.Parameter(xp.random.rand(b,h,dq,n))
        value = ag.Parameter(xp.random.rand(b,h,n,dv))
        mask = ag.Tensor(create_mask_of_future_positions(n))
        y = attention(query, key, value, mask=mask)
        y.backward()
        self.assertEqual(y.shape, (b,h,n,dv))

    def test_dotproduct_self_attention_gradient(self):
        b = 1
        n = 2
        h = 3
        dq = 4
        dv = 5
        
        query = ag.Parameter(xp.random.rand(b,h,n,dq))
        key = ag.Parameter(xp.random.rand(b,h,dq,n))
        value = ag.Parameter(xp.random.rand(b,h,n,dv))
        mask = ag.Tensor(create_mask_of_future_positions(n))
        def forward(params):
            self.unravel_params(params, query, key, value)
            got = attention(query, key, value, mask=mask)
            loss = ag.mean(got)
            return loss
        self.numeric_check(forward, query, key, value)

    def test_multihead_self_attention(self):
        b,n,d = 3,5,6
        h,d_kq,d_v = 1,7,8
        future_mask = ag.Tensor(create_mask_of_future_positions(n))
        layer = MultiHeadSelfAttention(
            d, dim_keyquery=d_kq, dim_value=d_v, num_heads=h, mask=future_mask)
        x_mask = xp.array([
            [0,0,0,0,0],
            [0,1,1,1,1],
            [0,0,0,1,1]
        ]) == 1
        x_mask = ag.Tensor(x_mask)
        x = ag.Parameter(xp.random.rand(b,n,d))
        y = layer.forward(x,x,x, x_mask=x_mask)

        y.backward()
        self.assertEquals(y.shape, (b,n,d_v))

        def forward(params):
            self.unravel_params(params, x)
            got = layer.forward(x, x, x, x_mask=x_mask)  
            loss = ag.mean(got)
            return loss
        self.numeric_check(forward, x)
        
    def test_residual_layer(self):
        batches = 2
        seq_len = 3
        embed_dims = 4
        x = ag.Parameter(xp.random.rand(batches, seq_len, embed_dims))
        layer1 = Sequence([LayerNorm2((seq_len, embed_dims)), SimpleSelfAttention(embed_dims)])
        layer = ResidualLayer(layer1)
        y = layer.forward(x)
        y.backward()

    def test_residual_layer_gradient(self):
        batches = 2
        seq_len = 3
        embed_dims = 4
        x = ag.Parameter(xp.random.rand(batches, seq_len, embed_dims))
        layer1 = Sequence([LayerNorm2((seq_len, embed_dims)), SimpleSelfAttention(embed_dims)])
        layer = ResidualLayer(layer1)
        def forward(params):
            self.unravel_params(params, x)
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        self.numeric_check(forward, x)

    def test_encoder(self):
        xp.random.seed(1)
        batches = 2
        seq_len = 3
        embed_dims = 6
        num_heads = 2

        x = xp.arange(batches*seq_len*embed_dims, dtype=xp.float64)
        x = x.reshape((batches, seq_len, embed_dims)) + 1.0
        x = ag.Parameter(x)
        layer = EncoderBlock(seq_len, embed_dims, num_heads=num_heads)
        y = layer.forward(x)
        y.backward()

    def test_encoder_gradient(self):
        xp.random.seed(1)
        batches = 2
        seq_len = 3
        embed_dims = 6
        num_heads = 2

        x = xp.arange(batches*seq_len*embed_dims, dtype=xp.float64)
        x = x.reshape((batches, seq_len, embed_dims)) + 1.0
        x = ag.Parameter(x)
        layer = EncoderBlock(seq_len, embed_dims, num_heads=num_heads)
        def forward(params):
            self.unravel_params(params, x)
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        self.numeric_check(forward, x)

        # dot = ag.generate_graphviz(y)
        # dot.render("graphviz", view=True, format="svg")

    def test_decoder_layer(self):
        xp.random.seed(1)
        batches = 1
        seq_len = 3
        embed_dims = 6
        num_heads = 2
        x_shape = (batches, seq_len, embed_dims)

        x = xp.arange(batches*seq_len*embed_dims, dtype=xp.float64)
        x = ag.Parameter(x.reshape(x_shape) + 1.0)
        memory = ag.Tensor(xp.arange(batches*seq_len*embed_dims).reshape(x_shape) + 1.0)

        layer = DecoderBlock(seq_len, embed_dims, num_heads=num_heads)
        y = layer.forward(x, memory)
        y.backward()
        
    

if __name__ == '__main__':
    unittest.main()

