import autograd2 as ag
import numpy as np
from . import Module

class Dense(Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.W = ag.Tensor(np.random.normal(scale=(2.0/(output_size*input_size)), size=(output_size, input_size)), requires_grad=True)
        self.b = ag.Tensor(np.random.normal(scale=(2.0/output_size), size=(output_size, 1)), requires_grad=True)
        self.params = [self.W, self.b]

    def forward(self, X):
        # TODO: Flatten the input to the correct size so that other modules 
        # don't have to do it themselves
        # y = ag.batch_matmul(self.W, X) + self.b
        # return y
        return ag.matmul(self.W, X) + self.b