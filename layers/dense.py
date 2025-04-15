import autograd2 as ag
# import numpy as np
from devices import xp
from . import Module

class Dense(Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.W = ag.Tensor(xp.random.normal(scale=(2.0/(output_size*input_size)), size=(output_size, input_size)), requires_grad=True)
        self.b = ag.Tensor(xp.random.normal(scale=(2.0/output_size), size=(output_size, 1)), requires_grad=True)
        self.params = [self.W, self.b]

    def forward(self, X):
        # TODO: Flatten the input to the correct size so that other modules 
        # don't have to do it themselves
        # y = ag.batch_matmul(self.W, X) + self.b
        # return y
        return ag.matmul(self.W, X) + self.b
    
class Linear(Module):
    def __init__(self, input_embed, output_embed):
        super().__init__()
        self.input_embed = input_embed
        self.output_embed = output_embed

        total_size = input_embed * output_embed
        w = xp.random.normal(scale=(2.0/total_size), size=(input_embed, output_embed))
        b = xp.random.normal(scale=(2.0/total_size), size=(1, 1))
        self.w = ag.Tensor(w, requires_grad=True)
        self.b = ag.Tensor(b, requires_grad=True)
        self.params = [self.w, self.b]

    def forward(self, x):
        return ag.batch_matmul(x, self.w) + self.b