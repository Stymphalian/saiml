import numpy as np
from typing import *
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
        W = inputs[0].value()
        X = inputs[1].value()
        return np.matmul(W, X)
    def gradients(self, node, outGrad):
        W = node.inputs[0].value()
        X = node.inputs[1].value()
        return (
            np.dot(outGrad, X.T),
            np.dot(W.T, outGrad)
        )
    
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