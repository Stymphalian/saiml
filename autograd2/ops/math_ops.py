import numpy as np
from typing import *
from ..base import Operator, Tensor

class TensorAdd(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.add(inputs[0].value(), inputs[1].value())
    def gradients(self, node, outGrad):
        a = node.inputs[0].value()
        b = node.inputs[1].value()
        da = outGrad
        db = outGrad
        if a.shape < da.shape:
            da = np.sum(da).reshape(a.shape)
        if b.shape < db.shape:
            db = np.sum(db).reshape(b.shape)
        assert da.shape == a.shape
        assert db.shape == b.shape

        return (da, db)

class TensorAddScalar(Operator):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, *inputs: Tuple[Tensor]):
        return inputs[0].value() + self.scalar
    def gradients(self, node, outGrad):
        assert outGrad.shape == node.inputs[0].shape
        return outGrad
    
class TensorSub(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.subtract(inputs[0].value(), inputs[1].value())
    def gradients(self, node, outGrad):
        a = node.inputs[0].value()
        b = node.inputs[1].value()
        da = outGrad
        db = -outGrad
        if a.shape < da.shape:
            da = np.sum(da).reshape(a.shape)
        if b.shape < db.shape:
            db = np.sum(db).reshape(b.shape)
        assert da.shape == a.shape
        assert db.shape == b.shape

        return (da, db)
    
class TensorSubScalar(Operator):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, *inputs: Tuple[Tensor]):
        return inputs[0].value() - self.scalar
    def gradients(self, node, outGrad):
        assert outGrad.shape == node.inputs[0].shape
        return outGrad
        
class TensorMult(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.multiply(inputs[0].value(), inputs[1].value())
    def gradients(self, node, outGrad):
        a = node.inputs[0].value()
        b = node.inputs[1].value()
        da = outGrad * b
        db = outGrad * a

        if a.shape < da.shape:
            da = np.sum(da).reshape(a.shape)
        if b.shape < db.shape:
            db = np.sum(db).reshape(b.shape)
        assert da.shape == a.shape
        assert db.shape == b.shape

        return (da, db)
    
class TensorMultScalar(Operator):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, *inputs: Tuple[Tensor]):
        return inputs[0].value() * self.scalar
    def gradients(self, node, outGrad):
        dz = outGrad * self.scalar
        assert dz.shape == node.inputs[0].shape
        return dz
    
class TensorDiv(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.divide(inputs[0].value(), inputs[1].value())

    def gradients(self, node, outGrad):
        a = node.inputs[0].value()
        b = node.inputs[1].value()
        da = outGrad / b
        db = -outGrad * a / (b * b)

        if b.shape < db.shape:
            db = np.sum(db).reshape(b.shape)
        if a.shape < da.shape:    
            da = np.sum(da).reshape(a.shape)
        assert da.shape == a.shape
        assert db.shape == b.shape

        return (da,db)
    
class TensorDivScalar(Operator):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, *inputs: Tuple[Tensor]):
        return inputs[0].value() / self.scalar
    def gradients(self, node, outGrad):
        dz = outGrad / self.scalar
        assert dz.shape == node.inputs[0].shape
        return dz

class TensorSin(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.sin(inputs[0].value())
    def gradients(self, node, outGrad):
        dz = np.multiply(outGrad, np.cos(node.inputs[0].value()))
        assert dz.shape == node.inputs[0].shape
        return dz
    
class TensorCos(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.cos(inputs[0].value())
    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        dz = np.multiply(outGrad, -np.sin(x))
        assert dz.shape == node.inputs[0].shape
        return dz
    
class TensorTan(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.tan(inputs[0].value())
    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        dz = np.multiply(outGrad, 1.0 / np.cos(x)**2)
        assert dz.shape == node.inputs[0].shape
        return dz
    
class TensorLog(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.log(inputs[0].value())
    def gradients(self, node, outGrad):
        dz = np.multiply(outGrad, 1.0 / node.inputs[0].value())
        assert dz.shape == node.inputs[0].shape
        return dz
    
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

        assert dx.shape == X.shape
        assert dw.shape == W.shape
        return (dx, dw)
    
class TensorSum(Operator):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        y = np.sum(x, axis=self.axis, keepdims=self.keepdims)
        if self.keepdims:
            if self.axis is None:
                x_shape= (1,)*x.ndim    
            else:
                x_shape = list(x.shape)
                x_shape[self.axis] = 1
                x_shape = tuple(x_shape)
            assert y.shape == x_shape
        return y

    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        assert outGrad.ndim <= x.ndim
        if outGrad.ndim < x.ndim:
            if self.axis is None:
                outGradShape = (1,)*x.ndim
            else:
                outGradShape = list(x.shape)
                outGradShape[self.axis] = 1
            outGrad = outGrad.reshape(tuple(outGradShape))
        dz = np.broadcast_to(outGrad, x.shape)
        # dz = np.ones(x.shape)
        # dz = np.multiply(outGrad, dz)
        assert dz.shape == x.shape
        return dz
    
class TensorExp(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.exp(inputs[0].value())
    def gradients(self, node, outGrad):
        x = np.exp(node.inputs[0].value())
        dx = np.multiply(outGrad, x)
        assert dx.shape == x.shape
        return dx
    
class TensorMean(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.mean(inputs[0].value())
    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        grad = np.ones(x.shape) / x.size
        dx = np.multiply(outGrad, grad)
        assert dx.shape == x.shape
        return dx
    
class TensorPower(Operator):
    def __init__(self, power, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.power = power
    def compute(self, *inputs: Tuple[Tensor]):
        return np.power(inputs[0].value(), self.power)
    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        dx = np.power(x, self.power - 1) * self.power
        dx = np.multiply(outGrad, dx)
        assert (dx.shape == x.shape)
        return dx
    
class TensorMax(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.max(inputs[0].value())
    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        xi = np.argmax(x)
        dx = np.zeros(node.inputs[0].value().shape)
        dx[xi] = 1
        dx = outGrad * dx
        assert dx.shape == x.shape
        return dx
    
class TensorNegate(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return -inputs[0].value()
    def gradients(self, node, outGrad):
        assert outGrad.shape == node.inputs[0].shape
        return -outGrad
        
class TensorNorm(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.linalg.norm(inputs[0].value())   
    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        dx = outGrad * x / np.linalg.norm(x)
        assert dx.shape == x.shape
        return dx

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
def cos(a):
    return TensorCos().tensor(a)
def tan(a):
    return TensorTan().tensor(a)
def log(a):
    return TensorLog().tensor(a)
def sum(a, axis=None, keepdims=False):
    return TensorSum(axis, keepdims).tensor(a)
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
def norm(a):
    return TensorNorm().tensor(a)