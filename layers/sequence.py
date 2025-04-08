from .module import Module
import numpy as np

class Sequence(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.params = layers

    def forward(self, X):
        for layer in self.layers:
            if isinstance(X, (list, tuple)):
                X = layer.forward(*X)
            else:
                X = layer.forward(X)
        return X
    
    def backward(self, context):
        for layer in reversed(self.layers):
            layer.backward(context)