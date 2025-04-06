import unittest
import numpy as np
import utils
from .transformer import *
from tokenizer import Tokenizer
import base_gradient_test

class TestAttention(base_gradient_test.NumericalGradientTest):

    def test_embedding(self):
        vocab = list("abcdefghijklmnopqrstuvwxyz")
        embed_dims = 10
        tokenizer = Tokenizer(vocab)

        x1 = tokenizer.encode("abc")
        x2 = tokenizer.encode("def")
        x = ag.Parameter(np.array([x1, x2]))
        
        layer = Embedding(tokenizer.vocab_size, embed_dims)
        y = layer.forward(x)
        self.assertEqual(y.shape, (2,3,embed_dims))

    def test_embedding_gradient(self):
        b,n,vocab_size = 2,3,4
        embed_dims = 10
        layer = Embedding(vocab_size,embed_dims)
        x = ag.Parameter(np.random.normal(size=(b,n,vocab_size)))
        def do():
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        def forward(params):
            self.unravel_params(params, x)
            return do()
        self.numeric_check(forward, x)

    def test_positional_encoding(self):
        np.random.seed(1)
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
        x = ag.Parameter(np.random.normal(size=(batches, seq_len, embed_dims)))
        def do():
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        def forward(params):
            self.unravel_params(params, x)
            return do()
        self.numeric_check(forward, x)

    def test_batch_layer_norm(self):
        np.random.seed(1)
        batches = 2
        x = ag.arange((batches, 2,3,4,5,6)) + 1.0
        layer = LayerNorm(x.shape[1:])
        y = layer.forward(x)
        y.backward()
        self.assertEqual(y.shape, x.shape)
        self.assertAlmostEqual(np.mean(y.value()[0]), 0.0)
        self.assertAlmostEqual(np.std(y.value()[0]), 1.0)

    @unittest.skip("Numerically unstable for numeric gradient checking")
    def test_batch_layer_norm_gradient(self):
        batches = 2
        input_shape = (4,3,2)
        layer = LayerNorm(input_shape)
        x = ag.Parameter(np.random.rand(batches, *input_shape))
        def do():
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        def forward(params):
            self.unravel_params(params, x)
            return do()
        self.numeric_check(forward, x)

    def test_feed_forward(self):
        np.random.seed(1)
        batches = 2
        seq_len = 3
        embed_dims = 4
        output_embed_dims = 5
        x = ag.arange((batches, seq_len, embed_dims)) + 1.0
        layer = FeedForward(embed_dims, output_embed_dims)
        y = layer.forward(x)
        y.backward()
        self.assertEqual(y.shape, (batches, seq_len, output_embed_dims))

    def test_feed_forward_gradient(self):
        np.random.seed(1)
        batches = 2
        seq_len = 3
        input_dims = 4
        output_dims = 4
        x = ag.Parameter(np.random.rand(batches, seq_len, input_dims))
        layer = FeedForward(input_dims, output_dims)
        def do():
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        def forward(params):
            self.unravel_params(params, x)
            return do()
        self.numeric_check(forward, x)

    def test_self_attention(self):
        np.random.seed(1)
        x = ag.arange((2,3,4)) + 1.0
        layer = SelfAttention(x.shape[2])
        y = layer.forward(x)
        y.backward()
        self.assertEqual(y.shape, x.shape)

    def test_self_attention_gradient(self):
        np.random.seed(1)
        batches = 2
        seq_len = 3
        embed_dims = 5
        x = ag.Parameter(np.random.rand(batches, seq_len, embed_dims))
        layer = SelfAttention(embed_dims)
        def forward(params):
            self.unravel_params(params, x)
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        self.numeric_check(forward, x)

    def test_residual_layer(self):
        batches = 2
        seq_len = 3
        embed_dims = 4
        x = ag.Parameter(np.random.rand(batches, seq_len, embed_dims))
        layer1 = Sequence([LayerNorm((seq_len, embed_dims)), SelfAttention(embed_dims)])
        layer = ResidualLayer(layer1)
        y = layer.forward(x)
        y.backward()

    def test_residual_layer_gradient(self):
        batches = 2
        seq_len = 3
        embed_dims = 4
        x = ag.Parameter(np.random.rand(batches, seq_len, embed_dims))
        layer1 = Sequence([LayerNorm((seq_len, embed_dims)), SelfAttention(embed_dims)])
        layer = ResidualLayer(layer1)
        def forward(params):
            self.unravel_params(params, x)
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        self.numeric_check(forward, x)

    def test_transformer(self):
        np.random.seed(1)
        batches = 2
        seq_len = 3
        embed_dims = 5

        x = np.arange(batches*seq_len*embed_dims, dtype=np.float64)
        x = x.reshape((batches, seq_len, embed_dims)) + 1.0
        x = ag.Parameter(x)
        layer = Transformer(seq_len, embed_dims)
        y = layer.forward(x)
        y.backward()

    def test_transformer_gradient(self):
        np.random.seed(1)
        batches = 2
        seq_len = 3
        embed_dims = 5

        x = np.arange(batches*seq_len*embed_dims, dtype=np.float64)
        x = x.reshape((batches, seq_len, embed_dims)) + 1.0
        x = ag.Parameter(x)
        layer = Transformer(seq_len, embed_dims)
        def forward(params):
            self.unravel_params(params, x)
            got = layer.forward(x)  
            loss = ag.mean(got)
            return loss
        self.numeric_check(forward, x)

        # dot = ag.generate_graphviz(y)
        # dot.render("graphviz", view=True, format="svg")
        
    

if __name__ == '__main__':
    unittest.main()

