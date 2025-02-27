import numpy as np
import struct
from array import array
from os.path  import join
import matplotlib.pyplot as plt
from pprint import pprint


import torch.nn as nn
import torch

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            img = img.reshape(28*28)
            images[i][:] = img
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)    

def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_derivative(X):
    return sigmoid(X) * (1 - sigmoid(X))

def softmax(X):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = X - np.max(X)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)
    # return np.exp(X) / np.sum(np.exp(X))

def softmax_gradient(z):
    """Computes the gradient of the softmax function.

    z: (T, 1) array of input values where the gradient is computed. T is the
       number of output classes.

    Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
    is DjSi - the partial derivative of Si w.r.t. input j.
    """
    Sz = softmax(z)
    # -SjSi can be computed using an outer product between Sz and itself. Then
    # we add back Si for the i=j cases by adding a diagonal matrix with the
    # values of Si on its diagonal.
    D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
    return D

def softmax_gradient_simple(z):
    """Unvectorized computation of the gradient of softmax.

    z: (T, 1) column array of input values.

    Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
    is DjSi - the partial derivative of Si w.r.t. input j.
    """
    Sz = softmax(z)
    N = z.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = Sz[i, 0] * (np.float32(i == j) - Sz[j, 0])
    return D

def softmax_derivative(X):
    S = softmax(X)
    grads = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            if i == j:
                grads[i][j] = S[i][0] * (1 - S[i][0])
            else:
                grads[i][j] = -S[i][0] * S[j][0]
    return grads

def cross_entropy_loss(y_true, y_pred):
    # cross entropy loss
    loss = 0
    for i in range(len(y_true)):
        if y_pred[i] <= 0.0:
            continue
        loss -= y_true[i] * np.log(y_pred[i])
    # prediction = np.log(y_pred)
    # loss = -np.sum(y_true * prediction)
    return loss

def cross_entropy_loss_derivative(y_true, y_pred):
    output = np.zeros(y_pred.shape, dtype=np.float64)
    for i in range(len(y_true)):
        if y_pred[i][0] <= 0.0:
            continue
        output[i][0] = -(y_true[i][0] / y_pred[i][0])
    return output

def mean_square_error(y_true, y_pred):
    # return np.mean(7np.power(y_true - y_pred, 2))
    err = np.mean(np.power(y_true - y_pred, 2))
    return err

def mean_square_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def onehot(y, size):
    onehot = np.zeros(size)
    onehot[y] = 1
    return onehot.reshape(10, 1)

class Layer:
    def __init__(self):
        pass
    def getParameters(self):
        return []
    def setParameters(self, parameters):
        return
    def getGradients(self):
        return []
    def forward(self, context, X):
        pass
    def backward(self, context, dE):
        pass

class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.normal(size=(output_size, input_size))
        self.b = np.random.normal(size=(output_size,1))
        self.dW = np.zeros((output_size, input_size))
        self.db = np.zeros((output_size, 1))

    def getParameters(self):
        return [self.W, self.b]
    def getGradients(self):
        return [self.dW, self.db]
    def setParameters(self, parameters):
        self.W = parameters[0]
        self.b = parameters[1]

    def forward(self, context, X):
        context["input"] = X
        a = np.dot(self.W, X)
        return a + self.b
        
    def backward(self, context, dE):        
        X = context["input"]
        self.dW = np.dot(dE, X.T)
        self.db = dE
        dEdx = np.dot(self.W.T, dE)

        learning_rate = context["learning_rate"]
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db
        return dEdx

class ActivationLayer(Layer):
    def __init__(self, output_size, fn, fn_derivative):
        super().__init__()
        self.fn = fn
        self.fn_derivative = fn_derivative

        self.b = np.zeros((output_size, 1), dtype=np.float64)
        self.db = np.zeros((output_size, 1), dtype=np.float64)

    # def getParameters(self):
    #     return [self.b]
    # def getGradients(self):
    #     return [self.db]
    # def setParameters(self, parameters):
    #     self.b = parameters[0]

    def forward(self, context, X):
        X = X + self.b
        context["input"] = X
        y = self.fn(X)
        return y
        
    def backward(self, context, dE):
        self.db = dE * self.fn_derivative(context["input"])
        return self.db

class SigmoidLayer(ActivationLayer):
    def __init__(self, size):
        super().__init__(size, sigmoid, sigmoid_derivative)
        self.size = size
    
class SoftmaxLayer(Layer):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.b = np.zeros((size, 1), dtype=np.float64)
        self.db = np.zeros((size, 1), dtype=np.float64)

    # def getParameters(self):
    #     return [self.b]
    # def setParameters(self, parameters):
    #     self.b = parameters[0]
    # def getGradients(self):
    #     return [self.db]

    def forward(self, context, input):
        context["input"] = input
        y = softmax(input)
        return y

    def backward(self, context, dE):
        # This version is faster than the one presented in the video
        input = context["input"]
        dydx = softmax_derivative(input)
        return np.dot(dydx.T, dE)
        # input = context["input"]
        # n = np.size(input)
        # self.db = np.dot((np.identity(n) - input.T) * input, dE)
        # return self.db
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)

class Model:
    def __init__(self):
        self.layers = [
            # DenseLayer(784, 784),
            # SigmoidLayer(784),
            # DenseLayer(784, 10),
            # SoftmaxLayer(10)

            DenseLayer(20, 10),
            SigmoidLayer(10),
            SoftmaxLayer(10)
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
        loss = cross_entropy_loss(y_true, y_pred)
        grads = cross_entropy_loss_derivative(y_true, y_pred)
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

        return numericalGradientCheck(compute, parameters, predictedGradients)

        
def numericalGradientCheck(fn, parameters: np.array, predictedGradient, h=1e-7):
    if predictedGradient.shape != parameters.shape:
        raise Exception("Gradients and parameters must be of the same shape")

    numericGradient = np.zeros(parameters.shape, dtype=np.float64)
    for i in range(len(parameters)):
        if i % 1000 == 0:
            print("Checking gradient for parameters {0}/{1}".format(i, len(parameters)))
        oldX = parameters[i][0]
        parameters[i][0] = oldX + h
        x1 = fn(parameters)
        parameters[i][0] = oldX - h
        x2 = fn(parameters)
        parameters[i][0] = oldX
        numericGradient[i] = (x1 - x2) / (2*h)

    numerator = np.linalg.norm(numericGradient - predictedGradient)
    denominator = np.linalg.norm(numericGradient) + np.linalg.norm(predictedGradient)
    diff = numerator / denominator
    return numericGradient, diff

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = np.array(x)
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    new_y = []
    for index in y:
        new_y.append(onehot(index, 10))
    y = new_y
    return x[:limit], y[:limit]

def train(model, x_train, y_train, number_epochs=1, learning_rate=0.1):
    for epoch in range(number_epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            pred = model.forward(x)
            loss, grad = model.loss(pred, y)
            error += loss
            model.backward(grad)

        error /= len(x_train)
        print(f"Epoch {epoch+1}/{number_epochs} - Error: {error}")

def main():
    # input_path = 'data'
    # training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    # training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    # test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    # test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    # mnist_dataloader = MnistDataloader(
    #     training_images_filepath,
    #     training_labels_filepath,
    #     test_images_filepath,
    #     test_labels_filepath)
    # (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    # x_train, y_train = preprocess_data(x_train, y_train, 1000)
    # x_test, y_test = preprocess_data(x_test, y_test, 1000)


    np.random.seed(0)
    X = np.random.rand(20,1)
    # Y = np.array([0.0, 1.0], dtype=np.float64).reshape(10,1)
    # Y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(10,1)
    # X = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(10,1)
    Y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(10,1)
    
    model = Model()
    _, diff = model.gradientCheck(X, Y)
    print(diff)
    # train(model, x_train, y_train, number_epochs=1, learning_rate=0.1)


    # y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(10,1)
    # x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(10,1)

    # x1 = softmax(x)
    # _ = cross_entropy_loss(y, x1)
    # dx1 = softmax_derivative(x)
    # dE = cross_entropy_loss_derivative(y, x1)
    # dEdx1 = np.dot(dx1.T, dE)

    # def loss(input):
    #     a1 = softmax(input)
    #     a2 = cross_entropy_loss(y, a1)
    #     return a2
    # _, diff = numericalGradientCheck(loss, x, dEdx1)
    # print(diff)

    # np.random.seed(0)
    # y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(10,1)
    # x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(10,1)

    # layer3 = DenseLayer(10, 10)
    # layer3Context = {"learning_rate": 0.1}
    # layer2 = SigmoidLayer(10)
    # layer2Context = {"learning_rate": 0.1}
    # layer1 = SoftmaxLayer(10)
    # layer1Context = {}

    # x3 = layer3.forward(layer3Context, x)
    # x2 = layer2.forward(layer2Context, x3)
    # x1 = layer1.forward(layer1Context, x2)
    # error = mean_square_error(y, x1)
    # dE = mean_square_error_derivative(y, x1)
    # dEdx1 = layer1.backward(layer1Context, dE)
    # dEdx2 = layer2.backward(layer2Context, dEdx1)
    # dEdx3 = layer3.backward(layer3Context, dEdx2)

    # print(error)
    # # pprint(dEdx2)

    # def loss(input):
    #     a4 = layer3.forward({}, input)
    #     a3 = layer2.forward({}, a4)
    #     a2 = layer1.forward({}, a3)
    #     a1 = mean_square_error(y, a2)
    #     return a1
    # _, diff = numericalGradientCheck(loss, x, dEdx3)
    # print(diff)

if __name__ == "__main__":
    main()
    

