
import math
import numpy as np
import autograd2 as ag
from .module import Module

       
class Conv2d(Module):

    def __init__(self, input_shape, num_kernels, kernel_size=3, stride=1, padding=0):
        self.input_shape = input_shape
        self.input_channels = input_shape[-3]
        self.input_height = input_shape[-2]
        self.input_width = input_shape[-1]
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = math.sqrt(2.0/(self.input_channels*self.input_height*self.input_width))
        kernels = [np.random.normal(scale=scale, size=(self.input_channels, kernel_size, kernel_size)) for x in range(self.num_kernels)]
        bias = [np.random.normal(scale=scale, size=(1,1)) for x in range(self.num_kernels)]
        self.W = [ag.Tensor(d, requires_grad=True) for d in kernels]
        self.b = [ag.Tensor(d, requires_grad=True) for d in bias]
        self.params = self.W + self.b

        # scale = math.sqrt(2.0/(self.input_channels*self.input_height*self.input_width))
        # kernels = np.random.normal(scale=scale, size=(self.num_kernels, self.input_channels, kernel_size, kernel_size))
        # bias = np.random.normal(scale=scale, size=(self.num_kernels, 1,1))
        # self.W = ag.Tensor(kernels, requires_grad=True)
        # self.b = ag.Tensor(bias, requires_grad=True)
        # self.params = [self.W, self.b]

    def forward(self, x):
        assert x.shape[-3:] == self.input_shape

        output = []
        for k in range(self.num_kernels):
            kernel = self.W[k]
            bias = self.b[k]
            z1 = ag.convolve2d(x, kernel, stride=self.stride, padding=self.padding)
            z2 = ag.add(z1, bias)
            output.append(z2)
        return ag.concatenate(output, axis=1)
    
class Conv2dTranspose(Module):
    def __init__(self, input_shape, num_kernels, kernel_size=3, stride=1, padding=0, outer_padding=0):
        self.input_shape = input_shape
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.outer_padding = outer_padding

        kernels = [np.random.normal(scale=(2.0/self.input_channels*kernel_size*kernel_size), size=(self.input_channels, kernel_size, kernel_size)) for x in range(self.num_kernels)]
        bias = [np.random.normal(scale=2, size=(1,1)) for x in range(self.num_kernels)]
        self.W = [ag.Tensor(d, requires_grad=True) for d in kernels]
        self.b = [ag.Tensor(d, requires_grad=True) for d in bias]
        self.params = self.W + self.b

    def forward(self, x):
        assert x.shape[1:] == self.input_shape

        output = []
        for k in range(self.num_kernels):
            kernel = self.W[k]
            bias = self.b[k]
            z1 = ag.convolve2d_transpose(
                x, kernel, stride=self.stride, padding=self.padding,
                outer_padding=self.outer_padding)
            z2 = ag.add(z1, bias)
            output.append(z2)
        return ag.vstack(output)