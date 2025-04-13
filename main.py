import os
import timeit
import time
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
            BatchSoftmax()
        ])

        self.params = [self.source_embedding] 
        self.params.extend(self.blocks)
        self.params += [self.norm, self.logits]

    def onehot_encode(self, x, pad_index=0):
        #TODO: pad_index should come from the tokenizer
        #TODO: Make this more efficient
        b, n = x.shape
        assert n <= self.seq_len
        n = self.seq_len
        x_pad = xp.pad(
            x.value(),
            ((0,0), (0, n - x.shape[1])),
            mode='constant',
            constant_values=pad_index)
        
        eye = xp.eye(self.vocab_len)
        y = eye[x_pad]
        return ag.Tensor(y)

    def forward(self, x, x_mask=None):
        b, n = x.shape                               # (b,n)
        y = self.onehot_encode(x)                    # (b,n,d_vocab)

        # embeddings
        y = self.source_embedding.forward(y)         # (b,n,d_embed)

        # encoder
        for block in self.blocks:
            y = block.forward(y, x_mask)             # (b,n,d_embed)
        y = self.norm.forward(y)                     # (b,n,d_embed)

        # Convert to logits
        y = self.logits.forward(y)                   # (b,d_vocab)
        return y
    
    def loss(self, y_pred, y_true):
        b, n, d_value = y_pred.shape
        y_true = self.onehot_encode(y_true)          # (b,n,d_vocab)
        loss = ag.cross_entropy_loss(y_pred, y_true, axis=(2,))
        loss = ag.mean(loss)
        return loss

    def generate(self, context, max_new_tokens):
        assert context.ndim == 2
        b, n = context.shape

        cumulative = context.value()
        for _ in range(max_new_tokens):
            # trim back down to seq_len
            if cumulative.shape[1] > self.seq_len:
                input_context = cumulative[:, -self.seq_len:]
            else:
                input_context = cumulative
            
            # run the model
            input_context = ag.Tensor(input_context)
            logits = self.forward(input_context).value()            # (b, n, d_vocab)
            logits = logits[:, -1, :]                               # (b, d_vocab)

            # sample
            next_idxs = [xp.argmax(xp.random.multinomial(1, logits[b])) for b in range(b)]
            next_idxs = xp.array(next_idxs).reshape((b,1))                        # (b, n)
            cumulative = xp.concatenate((cumulative, next_idxs), axis=1)          # (b, n+1)

        return ag.Tensor(cumulative)
    

def train(
        model: CharGPT,
        get_next_batch_fn,
        number_batches=100,
        number_epochs=2, 
        learning_rate=5e-4):
    # devices.print_memory("Forward:")
    # devices.print_memory("Backward:")
    # devices.print_memory("Model Backward:")
    context = {"optimizer": optimizer.RMSProp(lr=learning_rate)}

    for epoch in range(number_epochs):
        avg_batch_err = 0
        context["optimizer"].batch_start(epoch=epoch)
        print("Learning Rate: {}".format(context["optimizer"].learning_rate))

        for batch_num in range(number_batches):
            x, y = get_next_batch_fn()
            x, y = ag.Tensor(x), ag.Tensor(y)

            start = timeit.default_timer()
            pred = model.forward(x)
            loss = model.loss(pred, y)
            loss.backward()
            model.backward(context)
            end = timeit.default_timer()

            context["optimizer"].batch_step()

            avg_batch_err += float(loss.value())
            time_taken = end - start
            if batch_num % (number_batches // 10) == 0:
                print("Batch Number {:3d}: error {}, time taken: {:.4f}".format(batch_num, loss.value(), time_taken))
            

        model.checkpoint(f"checkpoint_{epoch}.npy")
        avg_batch_err /= number_batches
        context["optimizer"].batch_end()
        print(f"Epoch {epoch+1}/{number_epochs} - Average Batch Error: {avg_batch_err}")

    timestr = time.strftime("%Y%m%d")
    model.checkpoint(f"checkpoint_{timestr}.npy")


def generate_text(model, tok, num_tokens, seed=""):
    x = tok.encode(seed).reshape(1, -1)
    x = ag.Tensor(x)
    y = model.generate(x, num_tokens).numpy()
    generated = "".join(tok.decode(y[0]))
    return generated
                         

def main():
    np.random.seed(1337)
    cp.random.seed(1337)
    xp.random.seed(1337)

    seq_len = 64
    embed_dims = 32
    batch_size = 32

    dl = ShakespeareDataLoader("data/shakespeare.txt").load_data()
    tok = tokenizer.Tokenizer(dl.vocab)

    encoded_train = tok.encode(dl.train)
    # encoded_valid = tok.encode(dl.valid)
    model = CharGPT(
        seq_len, 
        len(tok.vocab), 
        embed_dims=embed_dims,
        decoder_layers=3,
        attention_heads=8
    )
    # model.load_checkpoint("checkpoint_20250413.npy")

    # x, y = tokenizer.get_batch(encoded_train, seq_len, batch_size)
    # for xi in x:
    #     print(tok.decode(xi))

    train(
        model,
        lambda: tokenizer.get_batch(encoded_train, seq_len, batch_size),
        number_batches=1000,
        number_epochs=10,
        learning_rate=1e-3
    )
    print(generate_text(model, tok, 500, seed="LUCIUS:\nHe said"))

if __name__ == "__main__":
    main()
    