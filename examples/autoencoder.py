import os
import numpy as np

from typing import *
from pprint import pprint

import utils
import autograd2 as ag
from layers import *
from dataloader.mnist import MnistDataloader
      
def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = np.array(x)
    # x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255

    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    new_y = []
    for index in y:
        new_y.append(utils.onehot_encode(index, 10))
    y = np.array(new_y)
    return x[:limit], y[:limit]

class AutoEncoder(Module):
    def __init__(self):
        self.encoder = Sequence([
            Conv2d((1,28,28), num_kernels=32, kernel_size=3, stride=2, padding=1),
            Sigmoid(),
            LayerNorm(),
            Conv2d((32,14,14), num_kernels=64, kernel_size=3, stride=2, padding=0),
            Sigmoid(),
            LayerNorm(),
            Flatten()
        ])
        self.z_mean = Dense(64*6*6, 2)
        self.z_log_var = Dense(64*6*6, 2)
        self.sampling = Sampling() # 2
        self.decoder = Sequence([
            Dense(2, 64*6*6),
            Reshape((64,6,6)),
            Conv2dTranspose((64,6,6), num_kernels=32, kernel_size=3, stride=2, padding=0),
            Sigmoid(),
            LayerNorm(),
            Conv2dTranspose((32,14,14), num_kernels=1, kernel_size=3, stride=2, padding=1),
            Sigmoid(),
        ])

        self.params = [
            self.encoder,
            self.z_mean, 
            self.z_log_var,
            self.sampling,
            self.decoder
        ]

    def forward(self, x):
        z = self.encoder.forward(x).set_name("z")
        z_mean = self.z_mean.forward(z).set_name("z_mean")
        z_var = self.z_log_var.forward(z).set_name("z_var")
        sample = self.sampling.forward(z_mean, z_var).set_name("sample")
        reconstruction = self.decoder.forward(sample).set_name("reconstruction")
        return reconstruction, z_mean, z_var
    
    def backward(self, context):
        self.decoder.backward(context)
        self.sampling.backward(context)
        self.z_log_var.backward(context)
        self.z_mean.backward(context)
        self.encoder.backward(context)

def main():
    # input_shape = (14, 14)
    # for stride in range(1,input_shape[0]):
    #     for padding in range(3):
    #         valid, out_shape = get_hw(input_shape, 3, stride, padding)
    #         if valid:
    #             print(input_shape, " Stride = ", stride, "Padding = ", padding, "Output shape = ", out_shape)
    # return

    # np.seterr(all='raise')
    x_train = np.array([
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        ], dtype=np.float64)
    y_train = np.array([0, 1.0], dtype=np.float64)
    x_train = ag.Parameter(x_train.reshape(1, 28, 28), requires_grad=True)
    # y_train = ag.Parameter(y_train.reshape(2, 1))
    y_train = ag.Parameter(np.identity(10, dtype=np.float64)[2].reshape(10, 1))

    np.random.seed(1)
    encoder = AutoEncoder()

    # def calculate_loss(x, y):
    #     return ag.cross_entropy_loss(x, y)
    # y, _, _ = encoder.forward(x_train)
    # loss = calculate_loss(y, y_train)
    # loss.backward()

    # for learning_rate, num_iterations in [(0.1, 1000), (0.01, 1000), (0.0001, 3000)]:
    #     context = {"learning_rate": learning_rate}
    #     print("Learning rate: ", learning_rate)
    #     for epoc in range(num_iterations):
    #         y, _, _ = encoder.forward(x_train)
    #         loss = calculate_loss(y, y_train)
    #         loss.backward()
    #         encoder.backward(context)
    #         if epoc % 1000 == 0:
    #             print(y)
    #             print(loss)
    
    def calculate_loss(x, z_log_var, z_mean, y):
        # reconstruction loss
        reconstruction_loss = ag.mean_square_error(x, y)
        reconstruction_loss.set_name("reconstruction_loss")

        # kl-divergence loss (probability distribution loss)
        kl_loss = -0.5 * (1 + z_log_var - ag.power(z_mean, 2) - ag.exp(z_log_var))
        kl_loss.set_name("kl_loss")
        kl_loss = ag.mean(kl_loss)
        kl_loss.set_name("kl_loss_mean")

        total_loss = reconstruction_loss + kl_loss
        total_loss.set_name("total_loss")
        return total_loss
    
    x, z_log_var, z_mean = encoder.forward(x_train)
    loss = calculate_loss(x, z_log_var, z_mean, x_train)
    loss.backward()

    # params, predGrads = encoder.get_params_and_grads()
    # def forward(params):
    #     encoder.set_params(params)
    #     x, z_log_var, z_mean = encoder.forward(x_train)
    #     loss = calculate_loss(x, z_log_var, z_mean, x_train)
    #     return loss.value()
    # grad, diffs = ag.utils.numeric_gradient_check(forward, params, predGrads, print_progress=True)
    # print(grad)
    # print(predGrads)
    # print(diffs)
    # print(ag.Node._NODE_AUTO_ID)

    # print(loss)

    for learning_rate, num_iterations in [(0.001, 5000)]:
        context = {"learning_rate": learning_rate}
        print("Learning rate: ", learning_rate)
        for epoc in range(num_iterations):
            x, z_log_var, z_mean = encoder.forward(x_train)
            loss = calculate_loss(x, z_log_var, z_mean, x_train)
            loss.backward()
            encoder.backward(context)
            # print(loss)
            if epoc % 100 == 0:
                dot = ag.generate_graphviz(loss)
                dot.render("graphviz", view=True, format="svg")
                print(loss, loss.inputs[0], loss.inputs[1])
    
    import matplotlib.pyplot as plt
    plt.imshow(x.value().reshape(28, 28), cmap='gray')
    plt.show(block=True)

    # loss = ag.cross_entropy_loss(x, y_train)
    

    # model = Sequence([
    #     Flatten(),
    #     Dense(28*28, 10),
    #     Sigmoid(),
    #     Softmax()
    # ])
    # pred = model.forward(x_train)
    # y_train = ag.Tensor(utils.onehot_encode(0, 10).reshape(10, 1))
    # loss = ag.cross_entropy_loss(pred, y_train)
    # loss.backward()
    
    # params, predGrads = model.get_params_and_grads()
    # def forward(params):
    #     model.set_params(params)
    #     pred = model.forward(x_train)
    #     loss = ag.cross_entropy_loss(pred, y_train)
    #     return loss.value()
    # grad, diffs = ag.utils.numeric_gradient_check(forward, params, predGrads, print_progress=True)
    # print(grad)
    # print(predGrads)
    # print(diffs)
    # print(ag.Node._NODE_AUTO_ID)


def main2():
    pass
    # input_path = 'data'
    # training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    # training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    # test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    # test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    # mnist_dataloader = MnistDataloader(
    #     training_images_filepath,
    #     training_labels_filepath,
    #     test_images_filepath,
    #     test_labels_filepath)
    # (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    # x_train, y_train = preprocess_data(x_train, y_train, 5)
    # x_test, y_test = preprocess_data(x_test, y_test, 5)

if __name__ == "__main__":
    main()
    

