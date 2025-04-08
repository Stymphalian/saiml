import numpy as np
import autograd2 as ag
from loss import *
from .module import Module

    
class Sigmoid(Module):
    def forward(self, x):
        return ag.sigmoid(x)

class Softmax(Module):
    def forward(self, x):
        return ag.softmax(x)
    
class LogSoftmax(Module):
    def forward(self, x):
        # TODO: Need to batch this.
        # axis = tuple(range(1, x.ndim))
        return ag.softmax(x, axis=(-1,))

class ReLU(Module):
    def forward(self, x):
        return ag.relu(x)
