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
import trainer

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
    def __init__(self, input_size=784,is_training=False):
        super().__init__()
        self.input_size = input_size

        self.blocks = Sequence([
            Linear(input_size, input_size),
            Dropout(is_training),
            ReLU(),
            LayerNorm2((input_size,)),

            Linear(input_size, input_size//2),
            Dropout(is_training),
            ReLU(),
            LayerNorm2((input_size//2,)),

            Linear(input_size//2, input_size//4),
            Dropout(is_training),
            ReLU(),
            LayerNorm2((input_size//4,)),
        ])
        self.logits = Sequence([
            Linear(input_size//4, 2),
            BatchSoftmax()
        ])
        self.params = [self.blocks, self.logits]

    def forward(self, x):
        b, n = x.shape               # (b,784)
        y = self.blocks.forward(x)   # (b,256)
        y = self.logits.forward(y)   # (b,2)
        return y
    
class GANGenerator(Sequence):
    def __init__(self, input_size=128, output_size=784, is_training=False):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__([
            Linear(input_size, input_size*2),
            ReLU(),
            Dropout(is_training),
            LayerNorm2((input_size*2,)),

            Linear(input_size*2, output_size//2),
            ReLU(),
            Dropout(is_training),
            LayerNorm2((output_size//2,)),

            Linear(output_size//2, output_size),
            ReLU(),
            Dropout(is_training),
            LayerNorm2((output_size,)),

            Linear(output_size, output_size),
            ReLU(),
        ])


def ag_onehot(x, num_classes=10):
    eye = xp.eye(num_classes)
    y = eye[x.value()]
    y = ag.Tensor(y)
    return y

class GANMnist(Module):
    def __init__(self, input_size=128, image_size=784, is_training=False):
        super().__init__()
        self.input_size = input_size
        self.image_size = image_size
        self.desc = GANDiscriminator(image_size, is_training)
        self.gen = GANGenerator(input_size, image_size, is_training)
        self.params = [self.desc, self.gen]

    def forward(self, x):
        b, n = x.shape
        random_latents = ag.random((b, self.input_size))

        gen = self.gen.forward(random_latents)    # (b,784)
        pred_gen = self.desc.forward(gen)         # (b,2)
        pred_true = self.desc.forward(x)          # (b,2)
        return pred_gen, pred_true
    
    def descriminate(self, x):
        b, n = x.shape
        random_latents = ag.random((b, self.input_size))

        gen = self.gen.forward(random_latents)    # (b,784)
        pred_gen = self.desc.forward(gen)         # (b,2)
        pred_true = self.desc.forward(x)          # (b,2)
        return pred_gen, pred_true

    def generate(self, b):
        random_latents = ag.random((b, self.input_size))
        gen = self.gen.forward(random_latents)    # (b,784)
        pred_gen = self.desc.forward(gen)         # (b,2)
        return pred_gen
    
    def descriminate_loss(self, preds):
        pred_gen, pred_true = preds
        b, n = pred_gen.shape                           # (b,2)
        b, n = pred_true.shape                          # (b,2)

        # calculate the loss
        onehot_one = ag_onehot(ag.zeros((b,), dtype=xp.int32), 2)
        onehot_zero = ag_onehot(ag.ones((b,), dtype=xp.int32), 2)
        desc_loss1 = ag.cross_entropy_loss(pred_gen, onehot_one, axis=(1,))
        desc_loss2 =  ag.cross_entropy_loss(pred_gen, onehot_zero, axis=(1,))
        desc_loss = desc_loss1 + desc_loss2
        desc_loss = ag.mean(desc_loss)

        return desc_loss
    
    def generator_loss(self, pred):
        b, n = pred.shape
        onehot_one = ag_onehot(ag.ones((b,), dtype=xp.int32), 2)
        gen_loss = ag.cross_entropy_loss(pred, onehot_one, axis=(1,))
        gen_loss = ag.mean(gen_loss)
        gen_loss = -gen_loss
        return gen_loss

    
def random_images(num_batches):
    return xp.random.rand(num_batches, 784, dtype=xp.float64)

def main():
    np.random.seed(1337)
    cp.random.seed(1337)
    xp.random.seed(1337)

    input_path = 'data'
    training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

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
    latent_size=128
    # desc = GANDiscriminator(image_size, is_training=True)
    # gen = GANGenerator(latent_size, image_size, is_training=True)
    # # pred = desc.forward(ag.Tensor(x_train))
    # image = gen.forward(ag.random((1,latent_size)))
    # plt.imshow(image.value()[0].get().reshape(28,28), cmap='gray')
    # plt.show()

    model = GANMnist(latent_size, image_size, is_training=True)
    trainer.train_gan(
        model,
        trainer.BatchLoader().from_arrays(x_train, y_train, num_batches=10),
        num_iters=1000,
        descrimiator_iters=1,
        generator_iters=1,
        minibatch_size=50,
        learning_rate=5e-5
    )
    # model.load_checkpoint("checkpoints/checkpoint_20250415.npy")
   

if __name__ == "__main__":
    main()
    