import os
import timeit
import time
import math
from typing import *
from pprint import pprint

import autograd2 as ag
from devices import xp
from layers import *
from dataloader.mnist import MnistDataloader
import cupy as cp
import numpy as np
from devices import xp
from matplotlib import pyplot as plt
import optimizer
import trainer
import utils

def preprocess_data(x, y, limit=None):
    # reshape and normalize input data
    x = xp.array(x)
    # x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float64") / 255

    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = xp.array(y)
    if limit:
        return x[:limit], y[:limit]
    else:
        return x, y
    
class GANDiscriminator(Module):
    def __init__(self, input_size=784, is_training=False):
        super().__init__()
        self.input_size = input_size
        self.img_size = math.sqrt(input_size)  # 1, 28,28

        self.blocks = Sequence([
            Reshape((-1, 1, 28, 28)),
            Conv2d((1, 28, 28), num_kernels=16, kernel_size=6, stride=2, padding=0),
            ReLU(),
            LayerNorm2((16, 12, 12)),

            Conv2d((16, 12, 12), num_kernels=32, kernel_size=6, stride=2, padding=0),
            ReLU(),
            LayerNorm2((32, 4, 4)),

            Conv2d((32, 4, 4), num_kernels=64, kernel_size=4, stride=1, padding=0),
            ReLU(),
            LayerNorm2((64, 1, 1)),
        ])
        self.logits = Sequence([
            Reshape((-1, 64)),
            Linear(64, 2),
            BatchSoftmax()
        ])
        self.params = [self.blocks, self.logits]

    def forward(self, x):
        b, n = x.shape               # (b,784)
        y = self.blocks.forward(x)   # (b,64)
        y = self.logits.forward(y)   # (b,2)
        return y
    
class GANGenerator(Sequence):
    def __init__(self, input_size=64, output_size=784, is_training=False):
        self.input_size = input_size
        self.output_size = output_size

        super().__init__([
            Reshape((-1, 64, 1, 1)),
            Conv2dTranspose((64, 1, 1), num_kernels=32, kernel_size=4, stride=1, padding=0),
            ReLU(),
            LayerNorm2((32, 4, 4)),

            Conv2dTranspose((32, 4, 4), num_kernels=16, kernel_size=6, stride=2, padding=0),
            ReLU(),
            LayerNorm2((16, 12, 12)),

            Conv2dTranspose((16, 12, 12), num_kernels=1, kernel_size=6, stride=2, padding=0),
            ReLU(),
            LayerNorm2((1, 28, 28)),

            Reshape((-1, 784)),
            Linear(output_size, output_size),
            Sigmoid(),
        ])

class GANMnist(Module):
    def __init__(self, input_size=128, image_size=784, is_training=False):
        super().__init__()
        self.input_size = input_size
        self.image_size = image_size
        self.desc = GANDiscriminator(image_size, is_training)
        self.gen = GANGenerator(input_size, image_size, is_training)
        self.params = [self.desc, self.gen]

    def forward(self, x):
        pass
    def loss(self, pred, label):
        loss = ag.cross_entropy_loss(pred, label, axis=-1)
        loss = ag.mean(loss)
        return loss
    
    def descriminate(self, x):
        b, n = x.shape
        pred = self.desc.forward(x)          # (b,2)
        return pred

    def generate(self, b):
        random_latents = ag.random((b, self.input_size))
        gen = self.gen.forward(random_latents)    # (b,784)
        return gen

def train_gan(
        model: Module,
        batcher,
        num_iters=100,
        minibatch_size=10,
        learning_rate=5e-4):
    context = {"optimizer": optimizer.RMSProp(lr=learning_rate)}
    print("Learning Rate: {}".format(context["optimizer"].learning_rate))

    valid = xp.array([1.0, 0.0])
    fake = xp.array([0.0, 1.0])

    for iter in range(num_iters):
        start = timeit.default_timer()

        # setup the data
        reals,_ = batcher.sample(minibatch_size)
        fakes = model.generate(minibatch_size)
        reals_labels = xp.broadcast_to(valid, (minibatch_size, 2))
        fakes_labels = xp.broadcast_to(fake, (minibatch_size, 2))
        xs = xp.concatenate((reals, fakes.value()), axis=0)
        ys = xp.concatenate((reals_labels, fakes_labels), axis=0)
        idx = xp.random.choice(xs.shape[0], xs.shape[0], replace=False)
        xs = xs[idx]
        ys = ys[idx]

        # figure out the losses
        pred = model.descriminate(ag.Tensor(xs))
        fake_pred = model.descriminate(fakes)
        desc_loss = model.loss(pred, ag.Tensor(ys))
        gen_loss = model.loss(fake_pred, ag.Tensor(reals_labels))
        
        # update the gradients
        desc_loss.backward()
        model.desc.backward(context)
        gen_loss.backward()
        model.gen.backward(context)
        
        end = timeit.default_timer()
        d_loss = float(desc_loss.value())
        g_loss = float(gen_loss.value())
        time_taken = end - start
        if (num_iters//10) > 0 and iter % (num_iters//10) == 0:
            print(f"[{iter}] D_err: {d_loss}, G_err: {g_loss} - time {time_taken:.4f}")
            timestr = time.strftime("%Y%m%d")
            model.checkpoint(f"checkpoints/checkpoint_{timestr}.npy")
            
    timestr = time.strftime("%Y%m%d")
    model.checkpoint(f"checkpoints/checkpoint_{timestr}.npy")


def test_conv_seq():
    utils.conv_utils.validate_conv2d_sequence([
        ((1, 28, 28), (16, 6, 6), 2, 0),
        ((16, 12, 12), (32, 6, 6), 2, 0),
        ((32, 4, 4), (64, 4, 4), 1, 0),
        # ((64, 1, 1), _, _, _)
    ])
    print("transpose")
    utils.conv_utils.validate_conv2d_transpose_sequence([
        ((64,1,1), (32, 4, 4), 1, 0),
        ((32,4,4), (16, 6, 6), 2, 0),
        ((16,12,12), (1, 6, 6), 2, 0),
        # ((1, 28, 28), _, _, _)
    ])

def main():
    np.random.seed(1337)
    cp.random.seed(1337)
    xp.random.seed(1337)

    input_path = 'data'
    training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    # test_conv_seq()
    # return

    mnist_dataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # model = GANMnist(is_training=True)
    image_size=784
    latent_size=64

    model = GANMnist(latent_size, image_size, is_training=True)
    # model.load_checkpoint("checkpoints/checkpoint_20250418.npy")
    # train_gan(
    #     model,
    #     trainer.BatchLoader().from_arrays(x_train, y_train, num_batches=10),
    #     num_iters=2000,
    #     minibatch_size=50,
    #     learning_rate=1e-4
    # )
    model.load_checkpoint("checkpoints/checkpoint_20250420.npy")
    y = model.generate(10)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(y[i].value().get().reshape(28,28), cmap='gray')
    plt.show(block=True)
   

if __name__ == "__main__":
    main()
    