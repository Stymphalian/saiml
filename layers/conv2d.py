
import numpy as np
from .layer import *

class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, context, X):
        context["input_shape"] = X.shape
        return np.reshape(X, (X.shape[0], -1, 1))
    
    def backward(self, context, dE):
        X_shape = context["input_shape"]
        return np.reshape(dE, X_shape)

class Conv2DLayer(Layer):
    def __init__(self, input_shape, num_kernels=1, kernel_size=3, stride=1, padding=0):
        super().__init__()
        # input_shape = (channels, height, width)
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        output_shape = self.get_output_shape()
        self.output_channels = output_shape[0]
        self.output_height = output_shape[1]
        self.output_width = output_shape[2]
        
        # kernels, and biases
        self.W = np.random.normal(size=(num_kernels, self.input_channels, kernel_size, kernel_size))
        self.b = np.random.normal(size=(self.num_kernels,1,1))
        self.dW = np.zeros(self.W.shape, dtype=np.float64)
        self.db = np.zeros(self.b.shape, dtype=np.float64)

    def get_output_shape(self):
        depth = self.num_kernels
        height = (self.input_height - self.kernel_size + 2*self.padding) // self.stride + 1
        width = (self.input_width - self.kernel_size + 2*self.padding) // self.stride + 1
        return (depth, height, width)

    def getParameters(self):
        return [self.W, self.b]
    def getGradients(self):
        return [self.dW, self.db]
    def setParameters(self, parameters):
        self.W = parameters[0]
        self.b = parameters[1]

    def forward(self, context, X):
        context["input"] = X
        # X = (batches, channels, height, width)
        batches, _, _, _ = X.shape
        
        output_shape = (batches, self.output_channels, self.output_height, self.output_width)
        output = np.zeros(output_shape, dtype=np.float64)
        padX = utils.zero_pad(X, self.padding)
        
        for batch in range(batches):    
            for k in range(self.num_kernels):
                for c in range(self.input_channels):
                    output[batch,k] += utils.convolve2D(padX[batch,c], self.W[k,c], stride=self.stride)
                output[batch,k] += self.b[k]
        return output

    def backward(self, context, dEdY):
        X = context["input"]
        batches = X.shape[0]
        
        dW = np.zeros(self.W.shape, dtype=np.float64)
        db = np.zeros(self.b.shape, dtype=np.float64)
        dEdx = np.zeros(X.shape, dtype=np.float64)
        padX = utils.zero_pad(X, self.padding)
        pad = self.padding

        for batch in range(batches):
            for k in range(self.num_kernels):
                for c in range(self.input_channels):
                    dEdY_dilate = utils.zero_dilate(dEdY[batch,k], self.stride-1)
                    tempdW = utils.convolve2D(padX[batch,c], dEdY_dilate)
                    tempdEdX = utils.full_convolve2D(dEdY[batch, k], self.W[k,c], stride=self.stride)
                    if self.padding > 0:
                        tempdEdX = tempdEdX[pad:-pad, pad:-pad]
                    
                    dW[k,c] += tempdW
                    dEdx[batch, c] += tempdEdX
                db[k] += np.sum(dEdY[batch, k])

        self.dW = dW
        self.db = db
        if not self.frozen:
            learning_rate = context["learning_rate"]
            self.W = self.W - learning_rate * self.dW
            self.b = self.b - learning_rate * self.db
        
        return dEdx

class Conv2DLayerHardCode(Layer):
    def __init__(self, input_shape):
        super().__init__()
        assert(input_shape == (3,3,1))
        self.W = np.random.normal(size=(2, 2, 1, 1))
        self.b = np.random.normal(size=(1,1,1,1))
        self.dW = np.zeros(self.W.shape, dtype=np.float64)
        self.db = np.zeros(self.b.shape, dtype=np.float64)
        # height, width, depth, batches
        self.output_shape = (2,2,1)

    def getParameters(self):
        return [self.W.copy(), self.b.copy()]
    def getGradients(self):
        return [self.dW.copy(), self.db.copy()]
    def setParameters(self, parameters):
        self.W = parameters[0]
        self.b = parameters[1]

    def forward(self, context, X):
        context["input"] = X
        x1 = X[0,0,0]
        x2 = X[0,0,1]
        x3 = X[0,0,2]
        x4 = X[0,1,0]
        x5 = X[0,1,1]
        x6 = X[0,1,2]
        x7 = X[0,2,0]
        x8 = X[0,2,1]
        x9 = X[0,2,2]
        w1 = self.W[0,0,0,0]
        w2 = self.W[0,1,0,0]
        w3 = self.W[1,0,0,0]
        w4 = self.W[1,1,0,0]
        b = self.b[0,0,0,0]

        z1 = x1*w1 + x2*w2 + x4*w3 + x5*w4 + b
        z2 = x2*w1 + x3*w2 + x5*w3 + x6*w4 + b
        z3 = x4*w1 + x5*w2 + x7*w3 + x8*w4 + b
        z4 = x5*w1 + x6*w2 + x8*w3 + x9*w4 + b
        return np.array([[
            [z1,z2],
            [z3,z4]
        ]])

    def backward(self, context, dEdY):
        X = context["input"]
        x1 = X[0,0,0]
        x2 = X[0,0,1]
        x3 = X[0,0,2]
        x4 = X[0,1,0]
        x5 = X[0,1,1]
        x6 = X[0,1,2]
        x7 = X[0,2,0]
        x8 = X[0,2,1]
        x9 = X[0,2,2]
        w1 = self.W[0,0,0,0]
        w2 = self.W[0,1,0,0]
        w3 = self.W[1,0,0,0]
        w4 = self.W[1,1,0,0]
        b = self.b[0,0,0,0]

        dz1 = dEdY[0,0,0,0]
        dz2 = dEdY[0,0,1,0]
        dz3 = dEdY[0,1,0,0]
        dz4 = dEdY[0,1,1,0]

        dEdw1 = dz1*x1 + dz2*x2 + dz3*x4 + dz4*x5
        dEdw2 = dz1*x2 + dz2*x3 + dz3*x5 + dz4*x6
        dEdw3 = dz1*x4 + dz2*x5 + dz3*x7 + dz4*x8
        dEdw4 = dz1*x5 + dz2*x6 + dz3*x8 + dz4*x9
        dEdb = dz1 + dz2 + dz3 + dz4

        dEdx1 = dz1*w1
        dEdx2 = dz1*w2 + dz2*w1
        dEdx3 = dz2*w2
        dEdx4 = dz1*w3 + dz3*w1
        dEdx5 = dz1*w4 + dz2*w3 + dz3*w2 + dz4*w1
        dEdx6 = dz2*w4 + dz4*w2
        dEdx7 = dz3*w3
        dEdx8 = dz3*w4 + dz4*w3
        dEdx9 = dz4*w4

        self.dW = np.array([
            [[dEdw1], [dEdw2]],
            [[dEdw3], [dEdw4]]
        ])
        self.db = np.array([
            [[[dEdb]]]
        ])
        self.dEdx = np.array([[
            [[dEdx1],[dEdx2],[dEdx3]],
            [[dEdx4],[dEdx5],[dEdx6]],
            [[dEdx7],[dEdx8],[dEdx9]],
        ]])
        return self.dEdx
    
class Conv2DLayerReference(Layer):
    def __init__(self, input_shape, num_kernels=1, kernel_size=3, stride=1, padding=0):
        super().__init__()
        # input_shape = (channels, height, width,)
        
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = np.random.normal(size=(num_kernels, self.input_channels, kernel_size, kernel_size))
        self.b = np.random.normal(size=(num_kernels,1,1,1))
        self.dW = np.zeros(self.W.shape, dtype=np.float64)
        self.db = np.zeros(self.b.shape, dtype=np.float64)

    def getParameters(self):
        return [self.W, self.b]
    def getGradients(self):
        return [self.dW, self.db]
    def setParameters(self, parameters):
        self.W = parameters[0]
        self.b = parameters[1]

    def forward(self, context, X):
        hparameters = {"pad" : self.padding, "stride": self.stride}
        z, cache = utils.conv_forward(X, self.W, self.b, hparameters)
        context["cache"] = cache
        return z

    def backward(self, context, dEdY):
        cache = context["cache"]
        dA, dW, db = utils.conv_backward(dEdY, cache)
        self.dW = dW
        self.db = db
        return dA

