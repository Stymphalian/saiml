import numpy as np
import os
from pprint import pprint

import utils
from layers import *
from loss import *
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
            # SigmoidLayer(10),
            # SigmoidLayer(784),
            # DenseLayer(784, 10),
            SoftmaxLayer(10, frozen=True)
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
        # loss = cross_entropy_loss(y_true, y_pred)
        # grads = cross_entropy_loss_derivative(y_true, y_pred)

        loss = mean_square_error(y_true, y_pred)
        grads = mean_square_error_derivative(y_true, y_pred)
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

def cross_entropy_loss(y_pred, y_true):
    # return ag.sum(y_pred)
    y_pred += ag.Tensor(1e-8)
    return -ag.sum(y_true * ag.log(y_pred))

def softmax(x):
    shiftx = x - ag.max(x)
    exps = ag.exp(shiftx)
    return exps / ag.sum(exps)

class Module:
    params: List[ag.Tensor]
    def __init__(self):
        self.params = []
    def forward(self, x):
        """Inference"""
        pass
    def backward(self, context):
        """Run backwards to update the gradients of all the parameters"""
        pass

    def get_params_grads_size(self):
        params_size = sum(x.size for x in self.params)
        grads_size = sum(x.size for x in self.params)
        return (params_size, grads_size)
    
    def get_params_grads(self):
        params = [x.data.reshape(-1) for x in self.params]
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
            self.params[pi].data = data
            count += self.params[pi].size

class Dense(Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.W = ag.Tensor(np.random.normal(size=(output_size, input_size)), requires_grad=True)
        self.b = ag.Tensor(np.random.normal(size=(output_size, 1)), requires_grad=True)
        self.params = [self.W, self.b]

    def forward(self, X):
        return ag.matmul(self.W, X) + self.b

    def backward(self, context):
        learning_rate = context["learning_rate"]
        self.W.data = self.W.data - learning_rate * self.W.grad
        self.b.data = self.b.data - learning_rate * self.b.grad

class Sigmoid(Module):
    def forward(self, x):
        return ag.Tensor(1) / (1.0 + ag.exp(-x))

class Softmax(Module):
    def forward(self, X):
        return softmax(X)

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
    
def main():
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
    x_train = ag.Tensor(x_train.reshape(-1, 1))
    y_train = ag.Tensor(y_train.reshape(10, 1))

    model = Sequence([
        Dense(784, 10),
        Sigmoid(),
        # Dense(784, 784),
        # Sigmoid(),
        # Dense(784,10),
        Softmax()
    ])
    x = model.forward(x_train)
    loss = cross_entropy_loss(x, y_train)
    loss.backward()
    params, predGrads = model.get_params_grads()
    def forward(params):
        model.set_params(params)
        pred = model.forward(x_train)
        loss = cross_entropy_loss(pred, y_train)
        return loss.value()
    grad, diffs = ag.utils.numeric_gradient_check(forward, params, predGrads)
    print(grad)
    print(predGrads)
    print(diffs)
    print(ag.Node._NODE_AUTO_ID)


def main2():
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
    x_train, y_train = preprocess_data(x_train, y_train, 5)
    x_test, y_test = preprocess_data(x_test, y_test, 5)

    # x_train = np.array([
    #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # ], dtype=np.float64)

    x_train = np.reshape(x_train, (-1, 784, 1))
    # np.random.seed(1)
    # x_train = np.arange(3, dtype=np.float64).reshape(1,3, 1) + 1
    # y_train = np.zeros(3, dtype=np.float64).reshape(1,3, 1)

    model = Model()
    # output = model.forward(x_train)
    # print(output.shape)

    grads, diff = model.gradientCheck(x_train[:1], y_train[:1])
    pprint(grads)
    pprint(diff)
    ## 2.051177863180922e-08
    return

if __name__ == "__main__":
    main()
    

