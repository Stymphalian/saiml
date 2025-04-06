import autograd2 as ag
import numpy as np
from . import Module

class LayerNorm(Module):
    EPSILON = ag.Tensor(1e-8)
    
    def forward(self, x):
        mean = ag.mean(x)
        stddev = ag.sqrt(ag.variance(x))  
        y = (x - mean) / (stddev + self.EPSILON)
        return y