from .module import Module
import autograd2 as ag

class Reshape(Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return ag.reshape(x, self.shape)
    
class Flatten(Module):
    def forward(self, x):
        return ag.reshape(x, (x.size, 1))