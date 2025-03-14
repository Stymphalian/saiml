import numpy as np

from loss import *

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
        z = np.matmul(self.W, X)
        z += self.b
        return z
        
    def backward(self, context, dE):        
        X = context["input"]
        self.dW = np.array([np.dot(a, b.T) for a,b in zip(dE, X)])
        self.dW = np.mean(self.dW, axis=0)
        self.db = np.mean(dE, axis=0)
        dEdx = np.array([np.dot(self.W.T, e) for e in dE])

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
        grad = np.multiply(dE, self.fn_derivative(context["input"]))
        self.db = np.mean(grad, axis=0)
        return grad

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

    def getParameters(self):
        return [self.b]
    def getGradients(self):
        return [self.db]
    def setParameters(self, parameters):
        self.b = parameters[0]

    def forward(self, context, X):
        X = X + self.b
        context["input"] = X
        y = softmax(X)
        return y

    def backward(self, context, dE):
        # This version is faster than the one presented in the video
        input = context["input"]
        dydx = softmax_derivative(input)
        grad = np.array([np.dot(a.T,b) for a,b in zip(dydx, dE)])
        self.db = np.mean(grad, axis=0)
        return grad
    
        # TODO: Why does this not work?
        # input = context["input"]
        # n = np.size(input)
        # self.db = np.dot((np.identity(n) - input.T) * input, dE)
        # return self.db

        # Original formula:
        # input = context["input"]
        # output = softmax(input)
        # n = np.size(input)
        # tmp = np.tile(output, n)
        # self.db = np.dot(tmp * (np.identity(n) - np.transpose(tmp)), dE)
        # return self.db