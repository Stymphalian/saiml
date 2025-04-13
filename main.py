import os
import timeit
from typing import *
from pprint import pprint

import autograd2 as ag
from devices import xp
from layers import *
from dataloader.shakespeare import ShakespeareDataLoader
import tokenizer
import optimizer
import cupy as cp
import numpy as np
import devices

class CharGPT(Module):
    # TODO: Make this more general and allow users to pass in the encoder/decoder submodels
    # TODO: Make an abstract EncoderDecoder model which transform should sub-class from.
    def __init__(
            self, 
            seq_len, 
            vocab_len, 
            embed_dims=64,
            decoder_layers=2,
            attention_heads=4,
        ):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_len = vocab_len
        self.embed_dims = embed_dims
        self.decoder_layers = decoder_layers
        self.attention_heads = attention_heads

        self.source_embedding = Sequence([
            Embedding(self.vocab_len, self.embed_dims),
            PositionalEncoding(seq_len, self.embed_dims),
        ])

        self.blocks = [
            DecoderBlock(
                seq_len,
                self.embed_dims,
                num_heads=self.attention_heads,
                include_encoder_attention=False
            ) for _ in range(self.decoder_layers)
        ]
        self.norm = LayerNorm2((self.embed_dims,))

        self.logits = Sequence([
            Linear(self.embed_dims, self.vocab_len),
            LogSoftmax()
        ])

        self.params = [self.source_embedding] 
        self.params.extend(self.blocks)
        self.params += [self.norm, self.logits]


    def forward(self, x, x_mask=None):
        # embeddings
        y = self.source_embedding.forward(x)         # (b,n,d_embed)

        # encoder
        for block in self.blocks:
            y = block.forward(y, x_mask)             # (b,n,d_embed)
        y = self.norm.forward(y)                     # (b,n,d_embed)

        # Convert to logits
        y = self.logits.forward(y)                   # (b,d_vocab)
        return y
    
    def loss(self, y_pred, y_true):
        b, n, d_value = y_pred.shape
        loss = ag.cross_entropy_loss(y_pred, y_true, axis=(2,))
        loss = ag.mean(loss)
        return loss
    

def train(
        model: CharGPT,
        x_train_batches,
        y_train_batches,
        number_epochs=2, 
        learning_rate=5e-4):
    

    mempool = cp.get_default_memory_pool()
    KB = 1024
    MB = 1024 * KB
    def print_memory(title):
        print(title)
        print("  Memory Used: {:.2f}".format(mempool.used_bytes() / MB))
        print("  Memory Total: {:.2f}".format(mempool.total_bytes() / MB))
        print("  Memory Free: {:.2f}".format(mempool.free_bytes() / MB))

    print_memory("Before Training:")

    # context = {"learning_rate": learning_rate}
    # context = {"optimizer": optimizer.SGDMomentum(lr=learning_rate, momentum=0.9)}
    context = {"optimizer": optimizer.RMSProp(lr=learning_rate)}
    # context = {"optimizer": optimizer.Adam(lr=learning_rate)}

    for epoch in range(number_epochs):
        avg_batch_err = 0
        context["optimizer"].iteration = 1
        # context["optimizer"].learning_rate = (1 / (1 + lr_decay * epoch)) * learning_rate
        print("Learning Rate: {}".format(context["optimizer"].learning_rate))

        for batch_num, ((x, x_mask), (y, y_mask)) in enumerate(zip(x_train_batches, y_train_batches)):

            start = timeit.default_timer()
            pred = model.forward(x, x_mask=x_mask)
            loss = model.loss(pred, y)
            print_memory("Forward:")
            loss.backward()
            print_memory("Backward:")
            model.backward(context)
            print_memory("Model Backward:")
            end = timeit.default_timer()
            context["optimizer"].iteration += 1

            avg_batch_err += float(loss.value())
            time_taken = end - start
            print("Batch Number {:3d}: error {}, time taken: {:.4f}".format(batch_num, loss.value(), time_taken))
        avg_batch_err /= len(x_train_batches)

        print(f"Epoch {epoch+1}/{number_epochs} - Average Batch Error: {avg_batch_err}")

def main():
    xp.random.seed(0)
    dl = ShakespeareDataLoader("data/shakespeare.txt")
    dl.load_data()
    tok = tokenizer.Tokenizer(dl.vocab)

    seq_len = 128
    embed_dims = 256
    batch_size = 64
    model = CharGPT(
        seq_len, 
        len(tok.vocab), 
        embed_dims=embed_dims,
        decoder_layers=6,
        attention_heads=8
    )
    print("Model Bytes: {:.2f} MB".format(model.total_bytes() / (1024*1024)))
    # print(model.total_bytes() / (1024*1024), "MB")

    x_train_batches = tokenizer.get_batches(dl.x_train, seq_len-1, batch_size)
    y_train_batches = tokenizer.get_batches(dl.y_train, seq_len-1, batch_size)
    x_train_batches = [
        tokenizer.convert_batches_to_numpy_with_mask(b1, tok, seq_len)
        for b1 in x_train_batches
    ]
    y_train_batches = [
        tokenizer.convert_batches_to_numpy_with_mask(b1, tok, seq_len)
        for b1 in y_train_batches
    ]
    x_train_batches = [(ag.Tensor(x), ag.Tensor(x_mask)) for x, x_mask in x_train_batches]
    y_train_batches = [(ag.Tensor(y), ag.Tensor(y_mask)) for y, y_mask in y_train_batches]

    total_batches_bytes = 0
    for b1, b1_mask in x_train_batches:
        total_batches_bytes += b1.value().nbytes + b1_mask.value().nbytes
    for b1, b1_mask in y_train_batches:
        total_batches_bytes += b1.value().nbytes + b1_mask.value().nbytes
    print("Total Batches Bytes: {:.2f} MB".format(total_batches_bytes / (1024*1024)))

    train(model, x_train_batches, y_train_batches)

    # y = model.forward(ag.Tensor(b1), ag.Tensor(b1_mask))
    # y.backward()
    # print(y.shape)
    # ag.render_graphviz(y)
if __name__ == "__main__":
    main()
    