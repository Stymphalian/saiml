import numpy as np
import os
from pprint import pprint

import utils
from layers import *
from mnist import MnistDataloader

class Model:
    def __init__(self):
        self.layers = [
            # FlattenLayer(),
            # DenseLayer(3, 3),
            # FlattenLayer(),
            # SoftmaxLayer(3, frozen=True),

            # Conv2DLayer((1,28,28), num_kernels=8, kernel_size=5, stride=1, padding=2),
            # # Conv2DLayerReference((1,28,28), num_kernels=8, kernel_size=5, stride=1, padding=0),
            # FlattenLayer(),
            # DenseLayer(6272, 10, frozen=True),
            # SoftmaxLayer(10, frozen=True),
            # FlattenLayer(),

            DenseLayer(784, 10),
            SigmoidLayer(10),
            SoftmaxLayer(10)
            # SigmoidLayer(784),
            # DenseLayer(784, 10),
        ]
        self.contexts = []
        self.gradients = []
        self.learning_rate = 0.01

    def forward(self, X):
        self.contexts = []
        for layer in self.layers:
            context = {}
            X = layer.forward(context, X)
            self.contexts.append(context)
        return X

    def loss(self, y_pred, y_true):
        # loss = mean_square_error(y_true, y_pred)
        # grads = mean_square_error_derivative(y_true, y_pred)
        # return (
        #     np.sum(y_pred.reshape(-1)),
        #     np.ones(y_pred.shape)
        # )
        loss = loss_cross_entropy_loss(y_true, y_pred)
        grads = loss_cross_entropy_loss_derivative(y_true, y_pred)

        # loss = mean_square_error(y_true, y_pred)
        # grads = mean_square_error_derivative(y_true, y_pred)
        return loss, grads

    def backward(self, gradients):
        for context, layer in zip(reversed(self.contexts), reversed(self.layers)):
            context["learning_rate"] = self.learning_rate
            gradients = layer.backward(context, gradients)
        return gradients
    
    def concatenateParameters(self):
        inputs = []
        for layer in self.layers:
            inputs.append(layer.getParameters())

        # unroll the inputs to a single vector of parameters
        parameters = None
        for params in inputs:
            for param in params:
                x = np.reshape(param, (-1, 1))
                if parameters is None:
                    parameters = x
                else:
                    parameters = np.concatenate((parameters, x), axis=0)
        return parameters
    
    def concatenateGradients(self):
        inputs = []
        for layer in self.layers:
            inputs.append(layer.getGradients())

        # unroll the inputs to a single vector of gradients
        parameters = None
        for params in inputs:
            for param in params:
                x = np.reshape(param, (-1, 1))
                if parameters is None:
                    parameters = x
                else:
                    parameters = np.concatenate((parameters, x), axis=0)
        return parameters
    
    def unravelParameters(self, parameters):
        inputs = []
        for layer in self.layers:
            inputs.append(layer.getParameters())

        output = []
        count = 0
        for paramsList in inputs:
            newList = []
            for params in paramsList:
                param_count = np.size(params)
                x = np.reshape(parameters[count:(count + param_count)], params.shape)
                newList.append(x)
                count += param_count
            output.append(newList)
        return output
    
    def gradientCheck(self, X, Y):
        # Run once to get the gradients
        pred = self.forward(X)
        parameters = self.concatenateParameters()
        _, grads = self.loss(pred, Y)
        self.backward(grads)
        predictedGradients = self.concatenateGradients()

        def compute(parameters):
            unrolled_params = self.unravelParameters(parameters)
            for layerIndex, layer in enumerate(self.layers):
                layer.setParameters(unrolled_params[layerIndex])
            pred = self.forward(X)
            loss, _ = self.loss(pred, Y)
            return loss

        return utils.numericalGradientCheck(compute, parameters, predictedGradients)
      

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

import autograd2 as ag
from typing import *

class Module:
    def __init__(self):
        self.params = []

    def forward(self, x):
        """Inference"""
        raise NotImplementedError()

    def backward(self, context):
        """Run backwards to update the gradients of all the parameters"""
        learning_rate = context["learning_rate"]
        for param in self.params:
            param._data -= learning_rate * param.grad

    def get_params_grads_size(self):
        params_size = sum(x.size for x in self.params)
        grads_size = sum(x.size for x in self.params)
        return (params_size, grads_size)
    
    def get_params_grads(self):
        params = [x._data.reshape(-1) for x in self.params]
        grads = [x.grad.reshape(-1) for x in self.params]
        if len(params) > 0:
            params = np.concatenate(params)
        else:
            params = np.array([])
        if len(grads) > 0:
            grads = np.concatenate(grads)
        else:
            grads = np.array([])
        return (params, grads)
    
    def set_params(self, params):
        count = 0
        for pi in range(len(self.params)):
            data_size = self.params[pi].size
            data_shape = self.params[pi].shape
            data = params[count:count+data_size].reshape(data_shape)
            self.params[pi]._data = data
            count += data_size

class Dense(Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.W = ag.Tensor(np.random.normal(size=(output_size, input_size)), requires_grad=True)
        self.b = ag.Tensor(np.random.normal(size=(output_size, 1)), requires_grad=True)
        self.params = [self.W, self.b]

    def forward(self, X):
        return ag.matmul(self.W, X) + self.b

class Sigmoid(Module):
    def forward(self, x):
        return ag.sigmoid(x)

class Softmax(Module):
    def forward(self, X):
        return ag.softmax(X)
        
class Conv2d(Module):

    def __init__(self, input_shape, num_kernels, kernel_size=3, stride=1, padding=0):
        self.input_shape = input_shape
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kernels = [np.random.normal(size=(self.input_channels, kernel_size, kernel_size)) for x in range(self.num_kernels)]
        bias = [np.random.normal(size=(1,1)) for x in range(self.num_kernels)]
        self.W = [ag.Tensor(d, requires_grad=True) for d in kernels]
        self.b = [ag.Tensor(d, requires_grad=True) for d in bias]
        self.params = self.W + self.b
        # self.params = [self.W, self.b]

    def forward(self, x):
        assert x.shape == self.input_shape

        output = []
        for k in range(self.num_kernels):
            kernel = self.W[k]
            bias = self.b[k]
            z1 = ag.convolve2d(x, kernel, stride=self.stride, padding=self.padding)
            z2 = ag.add(z1, bias)
            output.append(z2)
        return ag.vstack(*output)
    
class Conv2dTranspose(Module):
    def __init__(self, input_shape, num_kernels, kernel_size=3, stride=1, padding=0, outer_padding=0):
        self.input_shape = input_shape
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.outer_padding = outer_padding

        kernels = [np.random.normal(size=(self.input_channels, kernel_size, kernel_size)) for x in range(self.num_kernels)]
        bias = [np.random.normal(size=(1,1)) for x in range(self.num_kernels)]
        self.W = [ag.Tensor(d, requires_grad=True) for d in kernels]
        self.b = [ag.Tensor(d, requires_grad=True) for d in bias]
        self.params = self.W + self.b

    def forward(self, x):
        assert x.shape == self.input_shape

        output = []
        for k in range(self.num_kernels):
            kernel = self.W[k]
            bias = self.b[k]
            z1 = ag.convolve2d_transpose(
                x, kernel, stride=self.stride, padding=self.padding,
                outer_padding=self.outer_padding)
            z2 = ag.add(z1, bias)
            output.append(z2)
        return ag.vstack(*output)

class Flatten(Module):
    def forward(self, x):
        return ag.reshape(x, (x.size, 1))
    
class Reshape(Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return ag.reshape(x, self.shape)
    
class Sampling(Module):
    def __init__(self, seed=1):
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def forward(self, z_mean, z_var):
        assert z_mean.shape == z_var.shape
        dimensions = z_mean.size
        epsilon = self.rng.normal(size=(dimensions, 1))
        return z_mean + ag.exp(0.5 * z_var) * epsilon

class Sequence(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def get_params_grads_size(self):
        params, grads = self.get_params_grads()
        return (params.size, grads.size)

    def get_params_grads(self):
        params = []
        grads = []
        for layer in self.layers:
            p, g = layer.get_params_grads()
            assert p.size == g.size
            params.append(p)
            grads.append(g)
        params = np.concatenate(params)
        grads = np.concatenate(grads)
        return (params, grads)

    def set_params(self, params):
        count = 0
        for layer in self.layers:
            params_size, _ = layer.get_params_grads_size()
            sub_params = params[count:count+params_size]
            layer.set_params(sub_params)
            count += params_size

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, context):
        for layer in reversed(self.layers):
            layer.backward(context)

class AutoEncoder(Module):
    def __init__(self):
        self.encoder = Sequence([
            Conv2d((1,28,28), num_kernels=3, kernel_size=3, stride=2, padding=1),
            Sigmoid(),
            Conv2d((3,14,14), num_kernels=6, kernel_size=3, stride=2, padding=0),
            Sigmoid(),
            Flatten(),
        ])
        self.z_mean = Dense(6*6*6, 2)
        self.z_log_var = Dense(6*6*6, 2)
        self.sampling = Sampling() # 2
        self.decoder = Sequence([
            Dense(2, 6*6*6),
            Reshape((6,6,6)),
            Conv2dTranspose((6,6,6), num_kernels=3, kernel_size=3, stride=2, padding=0),
            Sigmoid(),
            Conv2dTranspose((3,14,14), num_kernels=1, kernel_size=3, stride=2, padding=1),
            Sigmoid()
        ])

    def get_params_grads(self):
        p1, g1 = self.encoder.get_params_grads()
        p2, g2 = self.z_mean.get_params_grads()
        p3, g3 = self.z_log_var.get_params_grads()
        p4, g4 = self.sampling.get_params_grads()
        p5, g5 = self.decoder.get_params_grads()
        return (np.concatenate([p1, p2, p3, p4, p5]), np.concatenate([g1, g2, g3, g4, g5]))

    def set_params(self, params):
        count = 0
        for layer in [self.encoder, self.z_mean, self.z_log_var, self.sampling, self.decoder]:
            params_size, _ = layer.get_params_grads_size()
            sub_params = params[count:count+params_size]
            layer.set_params(sub_params)
            count += params_size

    def forward(self, x):
        z = self.encoder.forward(x)
        z_mean = self.z_mean.forward(z)
        z_var = self.z_log_var.forward(z)
        sample = self.sampling.forward(z_mean, z_var)
        reconstruction = self.decoder.forward(sample)
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
    x_train = ag.Tensor(x_train.reshape(1, 28, 28), requires_grad=True)
    y_train = ag.Tensor(y_train.reshape(2, 1))

    np.random.seed(5)
    encoder = AutoEncoder()

    def calculate_loss(x, z_log_var, z_mean, y):
        # reconstruction loss
        reconstruction_loss = ag.mean_square_error(x, y)

        # kl-divergence loss (probability distribution loss)
        kl_loss = -0.5 * (1 + z_log_var - ag.power(z_mean, 2) - ag.exp(z_log_var))
        kl_loss = ag.mean(kl_loss)

        total_loss = reconstruction_loss + kl_loss
        return total_loss

    for learning_rate, num_iterations in [(0.1, 1000), (0.01, 1000), (0.0001, 3000)]:
        context = {"learning_rate": learning_rate}
        print("Learning rate: ", learning_rate)
        for epoc in range(num_iterations):
            x, z_log_var, z_mean = encoder.forward(x_train)
            loss = calculate_loss(x, z_log_var, z_mean, x_train)
            loss.backward()
            encoder.backward(context)
            if epoc % 1000 == 0:
                print(loss)
    
    import matplotlib.pyplot as plt
    plt.imshow(x.value().reshape(28, 28), cmap='gray')
    plt.show()

    # loss = ag.cross_entropy_loss(x, y_train)
    
    # params, predGrads = encoder.get_params_grads()
    # def forward(params):
    #     encoder.set_params(params)
    #     pred = encoder.forward(x_train)
    #     loss = ag.cross_entropy_loss(pred, y_train)
    #     return loss.value()
    # grad, diffs = ag.utils.numeric_gradient_check(forward, params, predGrads, print_progress=True)
    # print(grad)
    # print(predGrads)
    # print(diffs)
    # print(ag.Node._NODE_AUTO_ID)


def main2():
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
    y_train = np.array([1,0,0,0,0,0,0,0,0,0], dtype=np.float64)
    x_train = x_train.reshape(1, 784, 1)
    y_train = y_train.reshape(1, 10, 1)

    # x_train = np.reshape(x_train, (-1, 784, 1))
    # np.random.seed(1)
    # x_train = np.arange(3, dtype=np.float64).reshape(1,3, 1) + 1
    # y_train = np.zeros(3, dtype=np.float64).reshape(1,3, 1)

    np.random.seed(6)
    model = Model()
    # output = model.forward(x_train)
    # print(output.shape)

    grads, diff = model.gradientCheck(x_train, y_train)
    pprint(grads)
    pprint(diff)
    ## 2.051177863180922e-08
    return

if __name__ == "__main__":
    main()
    

