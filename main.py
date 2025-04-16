import os
import timeit
import time
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

class SimpleMnist(Module):
    def __init__(self, num_digits=10, is_training=False):
        super().__init__()
        self.num_digits = num_digits

        self.blocks = Sequence([
            Linear(784, 1024),
            Dropout(is_training),
            ReLU(),
            LayerNorm2((1024,)),

            Linear(1024, 1024),
            Dropout(is_training),
            ReLU(),
            LayerNorm2((1024,)),

            Linear(1024, 256),
            Dropout(is_training),
            ReLU(),
            LayerNorm2((256,)),
        ])
        self.logits = Sequence([
            Linear(256, self.num_digits),
            BatchSoftmax()
        ])

        self.params = [
            self.blocks,
            self.logits
        ]

    def onehot_encode(self, x, pad_index=0):
        eye = xp.eye(self.num_digits)
        y = eye[x.value()]
        return ag.Tensor(y)

    def forward(self, x):
        b, n = x.shape               # (b,784)
        y = self.blocks.forward(x)   # (b,256)
        y = self.logits.forward(y)   # (b,10)
        return y
    
    def loss(self, y_pred, y_true):
        b, n = y_pred.shape                          # (b,10)
        y_true = self.onehot_encode(y_true)          # (b,10)
        loss = ag.cross_entropy_loss(y_pred, y_true, axis=(1,))
        loss = ag.mean(loss)
        return loss    

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

    model = SimpleMnist(10, is_training=False)
    # trainer.train(
    #     model,
    #     trainer.BatchLoader().from_arrays(x_train, y_train, num_batches=2000),
    #     number_epochs=10,
    #     learning_rate=5e-5
    # )

    model.load_checkpoint("checkpoints/checkpoint_20250415.npy")
    # Evaluate the test set and visualize the worse performing cases
    y_pred = model(ag.Tensor(x_test))
    y_arg = xp.argmax(y_pred.value(), axis=1)
    y_true_onehot = model.onehot_encode(ag.Tensor(y_test))
    loss = ag.cross_entropy_loss(y_pred, y_true_onehot, axis=(1,))

    count = 0 
    indices = []
    for ix, (true, pred) in enumerate(zip(y_test, y_arg)):
        if true != pred:
            count += 1
            indices.append((loss[ix].value().get(), ix))
    print("Test Error Rate: ", count / x_test.shape[0])
    sorted_indices = sorted(indices, reverse=True)
    indices = [s[1] for s in sorted_indices]

    fig, axes = plt.subplots(5, 20, figsize=(15, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.set_title(f"{y_arg[indices][i]}")
        ax.imshow(xp.asnumpy(x_test[indices][i]).reshape(28, 28), cmap='gray')
    plt.show(block=True)

if __name__ == "__main__":
    main()
    