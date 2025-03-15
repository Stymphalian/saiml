import numpy as np
from .layer import Layer

class DenseLayer(Layer):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.normal(size=(output_size, input_size))
        self.b = np.random.normal(size=(output_size,1))
        self.dW = np.zeros((output_size, input_size))
        self.db = np.zeros((output_size, 1))

    def getParameters(self):
        if self.frozen:
            return []
        return [self.W, self.b]
    def getGradients(self):
        if self.frozen:
            return []
        return [self.dW, self.db]
    def setParameters(self, parameters):
        if self.frozen:
            return
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

        if not self.frozen:
            learning_rate = context["learning_rate"]
            self.W = self.W - learning_rate * self.dW
            self.b = self.b - learning_rate * self.db
        return dEdx