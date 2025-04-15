import autograd2 as ag
from devices import xp
from . import Module

class Dropout(Module):
    def __init__(self, is_training, p=0.5):
        super().__init__()
        self.p = p
        self.is_training = is_training

    def forward(self, x):
        if self.is_training:
            rand = xp.random.binomial(1, self.p, size=x.shape) 
            y = x * ag.Tensor(rand)
            return y
        else:
            return x * self.p