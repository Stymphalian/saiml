from .module import Module
import numpy as np

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