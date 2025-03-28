
import numpy as np
from .module import Module
import autograd2 as ag

class Pool(Module):
    MAX = "max" 
    AVERAGE = "average"

    def __init__(self, input_shape, mode=MAX, kernel_size=3, stride=1):
        super().__init__()
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode

        self.output_channels = self.input_channels
        self.output_height = (self.input_height - self.kernel_size) // self.stride + 1
        self.output_width = (self.input_width - self.kernel_size) // self.stride + 1

        ks = self.kernel_size * self.kernel_size
        self.average_kernel = ag.Tensor(
            np.ones((1, self.kernel_size, self.kernel_size)) / ks,
            requires_grad=False
        )

    def forward(self, x):
        kh,kw = (self.kernel_size, self.kernel_size)
        if self.mode == Pool.AVERAGE:
            return ag.convolve2d(
                x, 
                self.average_kernel, 
                stride=self.stride
            )
        elif self.mode == Pool.MAX:
            return ag.convolve2d(
                x,
                self.average_kernel,
                stride=self.stride
            )

# class PoolLayer(Layer):
#     MAX = "max"
#     AVERAGE = "average"

#     def __init__(self, input_shape, mode=MAX, kernel_size=3, stride=1):
#         super().__init__()
#         # input_shape = (channels, height, width)
#         self.input_channels = input_shape[0]
#         self.input_height = input_shape[1]
#         self.input_width = input_shape[2]
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.mode = mode

#         self.output_channels = self.input_channels
#         self.output_height = (self.input_height - self.kernel_size) // self.stride + 1
#         self.output_width = (self.input_width - self.kernel_size) // self.stride + 1

#     def getParameters(self):
#         return []
#     def getGradients(self):
#         return []
#     def setParameters(self, parameters):
#         return

#     def forward(self, context, X):
#         context["input"] = X
#         batches, _, _, _ = X.shape
        
#         output_shape = (batches, self.output_channels, self.output_height, self.output_width)
#         output = np.zeros(output_shape, dtype=np.float64)
#         kh, kw = (self.kernel_size, self.kernel_size)
        
#         for batch in range(batches):    
#             for c in range(self.input_channels):
#                 for row in range(self.output_height):
#                     for col in range(self.output_width):
#                         h = row * self.stride
#                         w = col * self.stride
                        
#                         x_slice = X[batch, c, h:h+kh, w:w+kw]
#                         if self.mode == PoolLayer.AVERAGE:
#                             kernel = np.mean(x_slice)
#                         elif self.mode == PoolLayer.MAX:
#                             kernel = np.max(x_slice)

#                         output[batch, c, row, col] = kernel
#         return output
    
#     def backward(self, context, dEdY):
#         X = context["input"]
#         batches = X.shape[0]
        
#         dEdx = np.zeros(X.shape, dtype=np.float64)
#         kh, kw = (self.kernel_size, self.kernel_size)

#         for batch in range(batches):
#             for c in range(self.input_channels):
#                 for row in range(self.output_height):
#                     for col in range(self.output_width):
#                         h = row * self.stride
#                         w = col * self.stride
                        
#                         if self.mode == PoolLayer.AVERAGE:
#                             kernel = np.ones((kh,kw), dtype=np.float64) / (kh*kw)
#                         elif self.mode == PoolLayer.MAX:
#                             x_slice = X[batch, c, h:h+kh, w:w+kw]
#                             kernel = (x_slice == np.max(x_slice))
#                         kernel = np.multiply(kernel, dEdY[batch, c, row, col])

#                         dEdx[batch, c, h:h+kh, w:w+kw] += kernel
        
#         return dEdx