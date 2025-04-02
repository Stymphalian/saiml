import numpy as np
from typing import *
from ..base import Operator, Tensor, TensorTuple

class TensorAdd(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.add(inputs[0].value(), inputs[1].value())
    def gradients(self, node, outGrad):
        a = node.inputs[0]
        b = node.inputs[1]
        da = outGrad
        db = outGrad
        if a.shape < da.shape:
            da = summation(da).reshape(a.shape)
        if b.shape < db.shape:
            db = summation(db).reshape(b.shape)
        assert da.shape == a.shape
        assert db.shape == b.shape
        return (da, db)

        # a = node.inputs[0].value()
        # b = node.inputs[1].value()
        # da = outGrad
        # db = outGrad
        # if a.shape < da.shape:
        #     da = np.sum(da).reshape(a.shape)
        # if b.shape < db.shape:
        #     db = np.sum(db).reshape(b.shape)
        # assert da.shape == a.shape
        # assert db.shape == b.shape

        # return (da, db)

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
        a = node.inputs[0]
        b = node.inputs[1]
        da = outGrad
        db = -outGrad
        if a.shape < da.shape:
            da = summation(da).reshape(a.shape)
        if b.shape < db.shape:
            db = summation(db).reshape(b.shape)
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
        a = node.inputs[0]
        b = node.inputs[1]
        da = outGrad * b
        db = outGrad * a

        if a.shape < da.shape:
            da = summation(da).reshape(a.shape)
        if b.shape < db.shape:
            db = summation(db).reshape(b.shape)
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
        a = node.inputs[0]
        b = node.inputs[1]
        da = outGrad / b
        db = -outGrad * a / (b * b)

        if b.shape < db.shape:
            db = summation(db).reshape(b.shape)
        if a.shape < da.shape:    
            da = summation(da).reshape(a.shape)
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
        dz = outGrad * cos(node.inputs[0])
        assert dz.shape == node.inputs[0].shape
        return dz
    
class TensorCos(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.cos(inputs[0].value())
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        dz = outGrad * -sin(x)
        assert dz.shape == node.inputs[0].shape
        return dz
    
class TensorTan(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.tan(inputs[0].value())
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        dz = mult(outGrad, Tensor(1.0) / power(cos(x), 2))
        assert dz.shape == node.inputs[0].shape
        return dz
    
class TensorLog(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.log(inputs[0].value())
    def gradients(self, node, outGrad):
        dz = mult(outGrad, Tensor(1.0) / node.inputs[0])
        assert dz.shape == node.inputs[0].shape
        return dz
    
class TensorMatMul(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        X = inputs[0].value()
        W = inputs[1].value()
        return np.matmul(X, W)
    def gradients(self, node, outGrad):
        X = node.inputs[0]
        W = node.inputs[1]
        dx = matmul(outGrad, W.T) 
        dw = matmul(X.T, outGrad) 
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

    def gradients(self, node: Tensor, outGrad: Tensor):
        x = node.inputs[0]
        assert outGrad.ndim <= x.ndim
        if outGrad.ndim < x.ndim:
            if self.axis is None:
                outGradShape = (1,)*x.ndim
            else:
                outGradShape = list(x.shape)
                outGradShape[self.axis] = 1
            outGrad = outGrad.reshape(tuple(outGradShape))
        dz = broadcast(outGrad, x.shape)
        # dz = np.ones(x.shape)
        # dz = np.multiply(outGrad, dz)
        assert dz.shape == x.shape
        return dz
    
class TensorExp(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.exp(inputs[0].value())
    def gradients(self, node, outGrad):
        x = exp(node.inputs[0])
        dx = mult(outGrad, x)
        assert dx.shape == x.shape
        return dx
    
class TensorMean(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.mean(inputs[0].value())
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        grad = Tensor(np.ones(x.shape)) / x.size
        dx = mult(outGrad, grad)
        assert dx.shape == x.shape
        return dx
    
class TensorPower(Operator):
    def __init__(self, power, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.power = power
    def compute(self, *inputs: Tuple[Tensor]):
        return np.power(inputs[0].value(), self.power)
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        dx = power(x, self.power - 1) * self.power
        dx = mult(outGrad, dx)
        assert (dx.shape == x.shape)
        return dx
    
class TensorMax(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.max(inputs[0].value())
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        xi = np.argmax(x.value())
        dx = np.zeros(x.shape)
        dx[xi] = 1

        dx = outGrad * Tensor(dx)
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
        x = node.inputs[0]
        dx = outGrad * x / norm(x)
        assert dx.shape == x.shape
        return dx
    
#################################################
# Tensor Tuples
#################################################

class TensorTupleMake(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return tuple([x.value() for x in inputs])
    
    def gradients(self, node, outGrad):
        assert isinstance(outGrad, TensorTuple)
        dx = [x for x in outGrad]
        assert len(dx) == len(node.inputs)
        return (*dx,)

class TensorTupleGetItem(Operator):
    def __init__(self, index):
        assert isinstance(index, int)
        self.index = index

    def compute(self, *inputs: Tuple[TensorTuple]):
        # TODO: How to handle input of tensorTuples
        assert len(inputs) == 1
        assert isinstance(inputs[0], TensorTuple)
        assert isinstance(inputs[0].value(), (list, tuple))
        assert isinstance(inputs[0].value()[self.index], Tensor)

        x = inputs[0].value()
        return x[self.index].value()
        # return inputs[0].value()[self.index].value()

    def gradients(self, node, outGrad):
        x = node.inputs[0]
        assert isinstance(x, TensorTuple)
        assert isinstance(outGrad, Tensor)
        
        dx = []
        for i in range(len(x)):
            if i == self.index:
                dx.append(outGrad)
            else:
                zero = Tensor(np.zeros(x[i].shape))
                dx.append(zero)
        assert len(dx) == len(x)
        return make_tuple(*dx)
    


#################################################
# SHAPING OPERATIONS
#################################################

class TensorReshape(Operator):
    def __init__(self, shape):
        self.shape = shape
    def compute(self, *inputs: Tuple[Tensor]):
        return inputs[0].value().reshape(self.shape)
    def gradients(self, node, outGrad):
        dx = outGrad.reshape(node.inputs[0].shape)
        assert dx.shape == node.inputs[0].shape
        return dx
        
# Broadcasting allows for these cases:
# 1. Promoting '1' dimension axes: (1, 2, 1, 3) => (4, 2, 5, 3)
# 2. Left-expanding axis:          (4,) => (x, y, z, 4)
class TensorBroadcast(Operator):
    def __init__(self, shape):
        if isinstance(shape, int):
            self.shape = (shape,)
        self.shape = shape
        assert isinstance(self.shape, tuple)

    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        # assert x.ndim == len(self.shape)
        return np.broadcast_to(x, self.shape)

    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        assert(x.ndim == outGrad.ndim)

        sum_axes = []
        for axis in range(x.ndim):
            s1 = x.shape[axis]
            s2 = outGrad.shape[axis]
            if s1 != s2:
                assert s1 == 1
                # The axis of the original shape is 1 so we need to do a sum 
                # along the previous axis to compute the gradient
                sum_axes.append(axis)

        dx = summation(outGrad, axis=tuple(sum_axes))
        dx = reshape(dx, x.shape)
        assert dx.shape == x.shape, f"dx shape: {dx.shape}, x shape: {x.shape}"
        return dx
    
class TensorTranspose(Operator):
    def __init__(self, axis=None):
        self.axis = axis
    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        return np.transpose(x, self.axis)
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        dx = transpose(outGrad, self.axis)
        assert dx.shape == x.shape
        return dx
    
class TensorRepeat(Operator):
    def __init__(self, repeats, axis=None):
        assert isinstance(repeats, int)
        self.repeats = repeats
        self.axis = axis

    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        return np.repeat(x, self.repeats, axis=self.axis)
    
    def gradients(self, node, outGrad):
        x = node.inputs[0]

        axis = self.axis
        if self.axis is None:
            axis = x.ndim - 1

        newShape = list(x.shape)
        newShape.insert(axis + 1, self.repeats)
        newShape = tuple(newShape)
        outGrad = outGrad.reshape(newShape)
        dx = summation(outGrad, axis=axis+1, keepdims=True)
        dx = dx.reshape(x.shape)
        return dx
    
class TensorTile(Operator):
    def __init__(self, repeats):
        if isinstance(repeats, int):
            repeats = (repeats,)
        self.repeats = repeats

    def _make_shape(self, x_shape, repeats):
        if len(repeats) < len(x_shape):
            repeats = (1,) * (len(x_shape) - len(repeats)) + repeats
        elif len(repeats) > len(x_shape):
            x_shape = (1,) * (len(repeats) - len(x_shape)) + x_shape
        return tuple(repeats), tuple(x_shape)

    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        return np.tile(x, self.repeats)

    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        repeats_shape, x_shape = self._make_shape(x.shape, self.repeats)
        assert len(repeats_shape) == len(x_shape)
        
        # Create the new_shape by expanding the repeated sections.
        # for example given the original shape (3,2,1) with a repeat of (4,5,6)
        # the new shape would be of shape (12, 10, 6)
        # but in order to do the sum we need to reshape the outGrad to 
        # (4,3  5,2  6,1). This allows us to do sums on the repeated axes
        new_shape = []
        for axis, r in enumerate(repeats_shape):
            new_shape.append(r)
            new_shape.append(x_shape[axis])
        new_shape = tuple(new_shape)
        sum_axes = tuple([axis * 2 for axis in range(len(repeats_shape))])

        dx = reshape(outGrad, new_shape)
        dx = summation(dx, axis=sum_axes)
        dx = reshape(dx, x.shape)
        return dx
    
class TensorStack(Operator):
    def __init__(self, axis=0):
        self.axis = axis

    def compute(self, *inputs: Tuple[TensorTuple]):
        assert len(inputs) == 1
        assert isinstance(inputs[0], TensorTuple)
        ref_shape = inputs[0][0].shape
        for x in inputs[0]:
            assert x.shape == ref_shape

        flat = [x.value() for x in inputs[0]]
        y = np.stack(flat, axis=self.axis)
        return y

    def gradients(self, node, outGrad):
        assert isinstance(outGrad, Tensor)
        assert isinstance(node.inputs[0], TensorTuple)
        x = node.inputs[0]
        x_shape = x[0].shape
        dx = unstack(outGrad, axis=self.axis)
        # dx = split(outGrad, len(x), axis=self.axis)
        # dx = [reshape(a, x_shape) for a in dx]

        assert len(dx) == len(x)
        return make_tuple(*dx)
        # return tuple(dx)
    
class TensorUnstack(Operator):
    def __init__(self, axis=0):
        self.axis = axis

    def compute(self, *inputs: Tuple[Tensor]):
        assert isinstance(inputs[0], Tensor)
        y = np.unstack(inputs[0].value(), axis=self.axis)
        return y

    def gradients(self, node, outGrad):
        dx = stack(outGrad, axis=self.axis)
        return dx
        
class TensorConcatenate(Operator):
    def __init__(self, axis=None):
        self.axis = axis

    def compute(self, *inputs: Tuple[TensorTuple]):
        a = [x.value() for x in inputs[0]]
        y = np.concatenate(a, axis=self.axis)
        return y

    def gradients(self, node, outGrad):
        assert isinstance(outGrad, Tensor)
        x = node.inputs[0]
        if self.axis is None:
            x_shape = x[0].shape
            dx = split(outGrad, len(x), axis=0)
            dx = [reshape(a, x_shape) for a in dx]
        else:
            dx = split(outGrad, len(x), axis=self.axis)

        assert len(dx) == len(x)
        return make_tuple(*dx)
    
class TensorSplit(Operator):
    def __init__(self, num_splits, axis=0):
        self.num_splits = num_splits
        self.axis = axis

    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        out = np.split(x, self.num_splits, axis=self.axis)
        assert len(out) == self.num_splits
        return out

    def gradients(self, node, outGrad):
        assert isinstance(outGrad, TensorTuple)
        assert len(outGrad) == self.num_splits
        x = node.inputs[0].value()

        # dy = [y for y in outGrad]
        # dy_shape = dy[0].shape
        # out = split(dy, self.num_splits, axis=0)

        # dy_shape = outGrad[0].shape
        # out = [reshape(a, dy_shape[1:]) for a in outGrad]
        out = [a for a in outGrad]
        dx = stack(out, axis=self.axis)
        dx = reshape(dx, x.shape)

        assert dx.shape == x.shape, f"dx shape: {dx.shape}, x shape: {x.shape}"
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
def summation(a, axis=None, keepdims=False):
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



def make_tuple(*input):
    return TensorTupleMake().tensor_tuple(*input)

def tuple_get_item(x, index):
    return TensorTupleGetItem(index).tensor(x)

def reshape(a, shape):
    return TensorReshape(shape).tensor(a)
def broadcast(a, shape):
    return TensorBroadcast(shape).tensor(a)
def transpose(x, axis=None):
    return TensorTranspose(axis).tensor(x)
def repeat(x, repeats, axis=None):
    return TensorRepeat(repeats, axis=axis).tensor(x)
def tile(x, repeats):
    return TensorTile(repeats).tensor(x)
def stack(inputs: List[Tensor], axis=0):
    return TensorStack(axis=axis).tensor(make_tuple(*inputs))
def unstack(a, axis=0):
    return TensorUnstack(axis=axis).tensor_tuple(a)
def concatenate(inputs, axis=None):
    return TensorConcatenate(axis).tensor(make_tuple(*inputs))
def split(a, num_splits, axis=0):
    return TensorSplit(num_splits, axis).tensor_tuple(a)
def vstack(arr):
    return concatenate(arr, axis=0)
def vsplit(a, num_splits):
    return split(a, num_splits, axis=0)