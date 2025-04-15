import autograd2 as ag
# import numpy as np
from devices import xp
from . import Module

class LayerNorm(Module):
    EPSILON = ag.Tensor(1e-8)
    
    def forward(self, x):
        mean = ag.mean(x)
        stddev = ag.sqrt(ag.variance(x))  
        y = (x - mean) / (stddev + self.EPSILON)
        return y
    
class LayerNorm2(Module):
    def __init__(self, features_shape, eps=1e-8):
        self.features_shape = features_shape
        self.w = ag.Tensor(xp.ones(features_shape), requires_grad=True)
        self.b = ag.Tensor(xp.zeros(features_shape), requires_grad=True)
        self.params = [self.w, self.b]
        self.epsilon = ag.Tensor(eps)
    
    def forward(self, x):
        index = x.ndim - len(self.features_shape)
        reduced_shape = x.shape[:index] + (1,)*len(self.features_shape)
        axes = tuple(range(index, x.ndim))

        mean = ag.mean(x, axis=axes).reshape(reduced_shape)
        u = x - mean
        var = ag.mean(u*u, axis=axes)
        stddev = ag.sqrt(var).reshape(reduced_shape)

        z = u * self.w
        z = z / (stddev + self.epsilon)
        z = z + self.b
        return z