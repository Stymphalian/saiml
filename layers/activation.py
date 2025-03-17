import numpy as np
from .layer import *
from loss import *

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

class ReLULayer(ActivationLayer):
    def __init__(self, size):
        super().__init__(size, relu, relu_derivative)
        self.size = size
    
class SoftmaxLayer(Layer):
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.b = np.zeros((size, 1), dtype=np.float64)
        self.db = np.zeros((size, 1), dtype=np.float64)

    def getParameters(self):
        if self.frozen:
            return []
        return [self.b]
    def getGradients(self):
        if self.frozen:
            return []
        return [self.db]
    def setParameters(self, parameters):
        if self.frozen:
            return
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

