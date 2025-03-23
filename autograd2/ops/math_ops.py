import numpy as np
from typing import *
# from utils.conv import convolve2d, convolve2d_gradient
from utils import conv
from ..base import Operator, Tensor

class TensorAdd(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.add(inputs[0].value(), inputs[1].value())
    def gradients(self, node, outGrad):
        return (outGrad, outGrad)

class TensorAddScalar(Operator):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, *inputs: Tuple[Tensor]):
        return inputs[0].value() + self.scalar
    def gradients(self, node, outGrad):
        return (outGrad,)
    
class TensorSub(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.subtract(inputs[0].value(), inputs[1].value())
    def gradients(self, node, outGrad):
        return (outGrad, -outGrad)
    
class TensorSubScalar(Operator):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, *inputs: Tuple[Tensor]):
        return inputs[0].value() - self.scalar
    def gradients(self, node, outGrad):
        return (outGrad,)
        
class TensorMult(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.multiply(inputs[0].value(), inputs[1].value())
    def gradients(self, node, outGrad):
        a = node.inputs[0].value()
        b = node.inputs[1].value()
        da = outGrad * b
        db = outGrad * a

        if a.shape < da.shape:
            da = np.sum(da)
        if b.shape < db.shape:
            db = np.sum(db)
        assert da.shape == a.shape
        assert db.shape == b.shape

        return (da, db)
    
class TensorMultScalar(Operator):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, *inputs: Tuple[Tensor]):
        return inputs[0].value() * self.scalar
    def gradients(self, node, outGrad):
        return (outGrad * self.scalar,)
    
class TensorDiv(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.divide(inputs[0].value(), inputs[1].value())

    def gradients(self, node, outGrad):
        a = node.inputs[0].value()
        b = node.inputs[1].value()
        da = outGrad / b
        db = -outGrad * a / (b * b)

        if b.shape < db.shape:
            db = np.sum(db)
        if a.shape < da.shape:    
            da = np.sum(da)
        assert da.shape == a.shape
        assert db.shape == b.shape

        return (da,db)
    
class TensorDivScalar(Operator):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, *inputs: Tuple[Tensor]):
        return inputs[0].value() / self.scalar
    def gradients(self, node, outGrad):
        return (outGrad / self.scalar,)

class TensorSin(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.sin(inputs[0].value())
    def gradients(self, node, outGrad):
        return (
            np.multiply(outGrad, np.cos(node.inputs[0].value())),
        )
    
class TensorLog(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.log(inputs[0].value())
    def gradients(self, node, outGrad):
        return (
            np.multiply(outGrad, 1.0 / node.inputs[0].value()),
        )
    
class TensorMatMul(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        X = inputs[0].value()
        W = inputs[1].value()
        return np.matmul(X, W)
    def gradients(self, node, outGrad):
        X = node.inputs[0].value()
        W = node.inputs[1].value()
        dx = np.matmul(outGrad, W.T) 
        dw = np.matmul(X.T, outGrad) 
        return (dx, dw)
    
class TensorSum(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.sum(inputs[0].value())
    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        dz = np.ones(x.shape)
        return (np.multiply(outGrad, dz),)
    
class TensorExp(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.exp(inputs[0].value())
    def gradients(self, node, outGrad):
        x = np.exp(node.inputs[0].value())
        return (np.multiply(outGrad, x),)
    
class TensorMean(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.mean(inputs[0].value())
    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        grad = np.ones(x.shape) / x.size
        return (np.multiply(outGrad, grad),)
    
class TensorPower(Operator):
    def __init__(self, power, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.power = power
    def compute(self, *inputs: Tuple[Tensor]):
        return np.power(inputs[0].value(), self.power)
    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        dx = np.power(x, self.power - 1) * self.power
        return (np.multiply(outGrad, dx),)
    
class TensorMax(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.max(inputs[0].value())
    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        xi = np.argmax(x)
        dx = np.zeros(node.inputs[0].value().shape)
        dx[xi] = 1
        return (outGrad * dx,)
    
class TensorNegate(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return -inputs[0].value()
    def gradients(self, node, outGrad):
        return (-outGrad,)
    
class TensorReshape(Operator):
    def __init__(self, shape):
        self.shape = shape
    def compute(self, *inputs: Tuple[Tensor]):
        return inputs[0].value().reshape(self.shape)
    def gradients(self, node, outGrad):
        return (outGrad.reshape(node.inputs[0].value().shape),)
    
class TensorConvolve2D(Operator):
    def __init__(self, stride, padding, dilate):
        assert(stride >= 1)
        assert(padding >= 0)
        assert(dilate >= 0)
        self.stride = stride
        self.padding = padding
        self.dilate = dilate

    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        kernel = inputs[1].value()
        return conv.convolve2d(
            x, kernel, self.stride, self.padding, self.dilate)

    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        kernel = node.inputs[1].value()
        return conv.convolve2d_gradient(
            x, kernel, outGrad, self.stride, self.padding, self.dilate)
    
class TensorConvolve2DTranspose(Operator):
    def __init__(self, stride, padding):
        assert(stride >= 1)
        assert(padding >= 0)
        self.stride = stride
        self.padding = padding

    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        kernel = inputs[1].value()

        assert(kernel.shape[0] == kernel.shape[1])
        return conv.convolve2d_transpose(
            x, kernel, self.stride, self.padding)

    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        kernel = node.inputs[1].value()
        return conv.convolve2d_transpose_gradient(
            x, kernel, outGrad, self.stride, self.padding)

def constant(a):
    return Tensor(a, requires_grad=False)
def add(a, b):
    return TensorAdd().tensor(a,b)
def add_scalar(a, b):
    return TensorAddScalar(b).tensor(a)
def sub(a, b):
    return TensorSub().tensor(a, b)
def sub_scalar(a, b):
    return TensorSubScalar(b).tensor(a)
def mult(a, b):
    return TensorMult().tensor(a, b)
def mult_scalar(a, b):
    return TensorMultScalar(b).tensor(a)
def div(a, b):
    return TensorDiv().tensor(a, b)
def div_scalar(a, b):
    return TensorDivScalar(b).tensor(a)
def matmul(a, b):
    return TensorMatMul().tensor(a, b)
def sin(a):
    return TensorSin().tensor(a)
def log(a):
    return TensorLog().tensor(a)
def sum(a):
    return TensorSum().tensor(a)
def exp(a):
    return TensorExp().tensor(a)
def mean(a):
    return TensorMean().tensor(a)
def power(a, power):
    return TensorPower(power).tensor(a)
def max(a):
    return TensorMax().tensor(a)
def neg(a):
    return TensorNegate().tensor(a)
def reshape(a, shape):
    return TensorReshape(shape).tensor(a)
def convolve2d(x, kernel, stride=1, padding=0, dilate=0):
    return TensorConvolve2D(stride, padding, dilate).tensor(x, kernel)
def convolve2d_transpose(x, kernel, stride=1, padding=0):
    return TensorConvolve2DTranspose(stride, padding).tensor(x, kernel)