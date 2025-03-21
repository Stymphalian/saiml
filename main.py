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
def main():
    def tanh(x):
        return x
    
    x = ag.Tensor(np.array([1,2,3]))



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
    

