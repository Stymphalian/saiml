import numpy as np
from typing import *
from ..base import Operator, Tensor, TensorTuple


def get_broadcast_shape(outGrad, x, axis):
    """
    Given the axis in which an operation was applied to x. 
    Get the new output_shape with (1,) put into the axis
    """
    if axis is None:
        outGradShape = (1,)*x.ndim
        outGradSize = x.size
    else:
        outGradShape = np.array(x.shape)                
        axes = list(axis) if isinstance(axis, (list, tuple)) else [axis]
        outGradSize = np.prod(outGradShape[axes])
        outGradShape[axes] = 1
    return tuple(outGradShape), outGradSize

def get_broadcasting_axes(a_shape, b_shape):
    axes = []
    a_ndim = len(a_shape)
    b_ndim = len(b_shape)
    if a_ndim < b_ndim:
        a_shape = (1,)*(b_ndim - a_ndim) + a_shape
        axes.extend(range(b_ndim - a_ndim, b_ndim))
    else:
        b_shape = (1,)*(a_ndim - b_ndim) + b_shape
        axes.extend(range(a_ndim - b_ndim, a_ndim))
    
    assert len(a_shape) == len(b_shape)
    num_axes = len(a_shape)

    axes = []
    for axis in range(num_axes):
        if a_shape[axis] != b_shape[axis]:
            axes.append(axis)
    return tuple(axes)


class TensorAdd(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        assert len(inputs) == 2
        a = inputs[0].value()
        b = inputs[1].value()
        y = np.add(a, b)
        assert y.shape == a.shape or y.shape == b.shape
        return y

    def gradients(self, node, outGrad):
        a = node.inputs[0]
        b = node.inputs[1]
        da = outGrad
        db = outGrad

        if a.size < da.size:
            axes = get_broadcasting_axes(a.shape, b.shape)
            da = summation(da, axis=axes).reshape(a.shape)
        if b.size < db.size:
            axes = get_broadcasting_axes(a.shape, b.shape)
            db = summation(db, axis=axes).reshape(b.shape)
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
        a = node.inputs[0]
        b = node.inputs[1]
        da = outGrad
        db = -outGrad
        if a.size < da.size:
            axes = get_broadcasting_axes(a.shape, b.shape)
            da = summation(da, axis=axes).reshape(a.shape)
        if b.size < db.size:
            axes = get_broadcasting_axes(a.shape, b.shape)
            db = summation(db, axis=axes).reshape(b.shape)
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

        if a.size < da.size:
            axes = get_broadcasting_axes(a.shape, b.shape)
            da = summation(da, axis=axes).reshape(a.shape)
        if b.size < db.size:
            axes = get_broadcasting_axes(a.shape, b.shape)
            db = summation(db, axis=axes).reshape(b.shape)
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

        if a.size < da.size:
            axes = get_broadcasting_axes(a.shape, b.shape)
            da = summation(da, axis=axes).reshape(a.shape)
        if b.size < db.size:
            axes = get_broadcasting_axes(a.shape, b.shape)
            db = summation(db, axis=axes).reshape(b.shape)
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
        dz = mult(outGrad, 1.0 / node.inputs[0])
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

def parse_einsum_equation(equation:str, *operands):
    assert "->" in equation    
    in_eq, out_eq = equation.split("->")
    
    in_eqs = in_eq.strip().split(",")
    in_eqs = [eq.replace(" ", "") for eq in in_eqs]
    out_eq = out_eq.strip().replace(" ", "")

    assert len(in_eqs) == len(operands)
    for i in range(len(in_eqs)):
        assert len(in_eqs[i]) == operands[i].ndim

    return in_eqs, out_eq, operands

class TensorEinsteinSum(Operator):
    def __init__(self, equation):
        self.equation = equation

    def compute(self, *inputs: Tuple[Tensor]):
        # assert(len(inputs) == 2)
        x = [a.value() for a in inputs]
        y = np.einsum(self.equation, *x)
        return y

    def gradients(self, node, outGrad):
        in_eqs, out_eq, ops = parse_einsum_equation(self.equation, *node.inputs)
        assert(outGrad.ndim == len(out_eq))

        derivatives = []
        for arg in range(len(in_eqs)):
            current_eq = in_eqs[arg]
            rest_eq = in_eqs[:arg] + in_eqs[arg+1:]
            current_op = ops[arg]
            rest_ops = ops[:arg] + ops[arg+1:]

            new_rest_eq = ",".join(rest_eq)
            new_rest_ops = rest_ops        
            missing = [
                (index, symbol) for index, symbol in enumerate(current_eq)
                if symbol not in out_eq
            ]
            if len(missing) > 0:
                # when the einsum output equation is missing a symbol, it means
                # we are doing a sum over that column. When computing the gradient
                # backward we need to expand out that missing dimension
                missing_eq = "".join([symbol for _, symbol in missing])
                missing_shapes = [current_op.shape[index] for index, _ in missing]
                missing_ops = Tensor(np.ones(missing_shapes))

                new_rest_eq = ",".join([missing_eq] + rest_eq)
                new_rest_ops = (missing_ops,) + rest_ops
            
            new_out_eq = f"{out_eq},{new_rest_eq}->{current_eq}"
            dx = einsum(new_out_eq, outGrad, *new_rest_ops)
            derivatives.append(dx)

        for xi, x in enumerate(node.inputs):
            assert derivatives[xi].shape == x.shape
        return tuple(derivatives)
    
class TensorSum(Operator):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        y = np.sum(x, axis=self.axis, keepdims=self.keepdims)
        # if self.keepdims:
        #     if self.axis is None:
        #         x_shape= (1,)*x.ndim    
        #     else:
        #         x_shape = list(x.shape)
        #         x_shape[self.axis] = 1
        #         x_shape = tuple(x_shape)
        #     assert y.shape == x_shape
        return y

    def gradients(self, node: Tensor, outGrad: Tensor):
        x = node.inputs[0]
        assert outGrad.ndim <= x.ndim
        # if outGrad.ndim < x.ndim:
        if outGrad.shape != x.shape:
            outGradShape, _ = get_broadcast_shape(outGrad, x, self.axis)
            outGrad = outGrad.reshape(outGradShape)
        dz = broadcast(outGrad, x.shape)
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
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
    def compute(self, *inputs: Tuple[Tensor]):
        return np.mean(
            inputs[0].value(), 
            axis=self.axis,
            keepdims=self.keepdims
        )
    def gradients(self, node, outGrad):
        # axis == None: (x,y,z) -> (1,1,1)
        # axis == 1:    (x,y,z) -> (x,1,z)
        # axes == (1,2) (x,y,z) -> (x,)
        x = node.inputs[0]
        outGradSize = x.size
        if outGrad.shape != x.shape:
            outGradShape, outGradSize = get_broadcast_shape(outGrad, x, self.axis)
            outGrad = outGrad.reshape(outGradShape)
        dx = broadcast(outGrad, x.shape) / outGradSize

        assert dx.shape == x.shape
        return dx
    
class TensorPower(Operator):
    def __init__(self, power, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.power = power
    def compute(self, *inputs: Tuple[Tensor]):
        if self.power < 0:
            return 1.0 / np.power(inputs[0].value(), abs(self.power))
        else:
            return np.power(inputs[0].value(), self.power)
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        dx = power(x, self.power - 1) * self.power
        dx = mult(outGrad, dx)
        assert (dx.shape == x.shape)
        return dx
    
class TensorMax(Operator):
    def __init__(self, axis=None):
        self.axis = axis
    def compute(self, *inputs: Tuple[Tensor]):
        return np.max(inputs[0].value(), axis=self.axis)
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        if self.axis is None:
            xi = np.argmax(x.value())
            dx = np.zeros(x.shape)
            np.put(dx, xi, 1)
        else:
            xi = np.argmax(x.value(), axis=self.axis, keepdims=True)
            dx = np.zeros(x.shape)
            np.put_along_axis(dx, xi, 1, axis=self.axis)
            outGradShape, _ = get_broadcast_shape(outGrad, x, self.axis)
            outGrad = outGrad.reshape(outGradShape)

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
    def __init__(self, axis=None):
        self.axis = axis
    def compute(self, *inputs: Tuple[Tensor]):
        return np.linalg.norm(inputs[0].value(), axis=self.axis)
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        outGradShape, _ = get_broadcast_shape(outGrad, x, self.axis)
        outGrad = outGrad.reshape(outGradShape)
        dx = outGrad * x / norm(x, axis=self.axis)
        assert dx.shape == x.shape
        return dx
    
class TensorSqrt(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        return np.sqrt(inputs[0].value())
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        dx = outGrad / (2 * sqrt(x))
        assert dx.shape == x.shape
        return dx
    
#################################################
# Tensor Tuples
#################################################

class TensorTupleMake(Operator):
    def compute(self, *inputs: Tuple[Tensor]) -> Union[np.ndarray, Tuple[np.ndarray]]:
        assert len(inputs) > 0
        assert isinstance(inputs[0], Tensor)
        return tuple(inputs)
    
    def gradients(self, node, outGrad):
        assert isinstance(outGrad, TensorTuple)
        dx = [x for x in outGrad]
        assert len(dx) == len(node.inputs)
        return tuple(dx)

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

        x_tuple = inputs[0].value()
        return x_tuple[self.index].value()

    def gradients(self, node, outGrad: Tensor):
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
    
class TensorTupleGetSlice(Operator):
    def __init__(self, index_slice):
        assert isinstance(index_slice, slice)
        self.slice = index_slice

    def compute(self, *inputs: Tuple[TensorTuple]):
        # TODO: How to handle input of tensorTuples
        assert len(inputs) == 1
        assert isinstance(inputs[0], TensorTuple)
        assert isinstance(inputs[0].value(), (list, tuple))
        assert isinstance(inputs[0].value()[0], Tensor)

        x = inputs[0].value() # list/tuple
        return tuple(x[self.slice])

    def gradients(self, node, outGrad):
        x = node.inputs[0]
        assert isinstance(x, TensorTuple)
        assert isinstance(outGrad, TensorTuple)
        
        dx = []
        for i in range(len(x)):
            zero = Tensor(np.zeros(x[i].shape))
            dx.append(zero)
        for i, d in enumerate(outGrad[self.slice]):
            dx[i] = d

        assert len(dx) == len(x)
        return make_tuple(*dx)

# class TensorTupleAdd(Operator):
#     def compute(self, *inputs: Tuple[TensorTuple]):
#         assert len(inputs) == 2
#         assert isinstance(inputs[0], TensorTuple)
#         assert isinstance(inputs[1], TensorTuple)
#         a = inputs[0]
#         b = inputs[1]

#         y = [a1 + b1 for a1,b1 in zip(a.value(), b.value())]
#         return tuple(y)

#     def gradients(self, node, outGrad):
#         da = outGrad
#         db = outGrad
#         return (da, db)

class TensorTupleSum(Operator):
    def compute(self, *inputs: Tuple[TensorTuple]):
        assert len(inputs) >= 1
        assert isinstance(inputs[0], TensorTuple)

        y = None
        for t in inputs:
            if y is None:
                y = list(t.value())
            else:
                for i, tensor in enumerate(t.value()):
                    y[i] += tensor
        return tuple(y)

    def gradients(self, node, outGrad: TensorTuple):
        assert isinstance(outGrad, TensorTuple)
        dz = []
        for _ in range(len(node.inputs)):
            dz.append(outGrad)
        return tuple(dz)
            
def make_tuple(*input):
    return TensorTupleMake().tensor_tuple(*input)
def tuple_get_item(x, index):
    return TensorTupleGetItem(index).tensor(x)
def tuple_get_slice(x, slice):
    return TensorTupleGetSlice(slice).tensor_tuple(x)
# def tuple_add(a, b):
#     return TensorTupleAdd().tensor_tuple(a, b)
def tuple_sum(*inputs):
    return TensorTupleSum().tensor_tuple(*inputs)

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
        x_shape = x.shape
        if x.ndim < outGrad.ndim:
            x_shape = (1,)*(outGrad.ndim - x.ndim) + x_shape
        assert len(x_shape) == outGrad.ndim
        # assert(x.ndim <= outGrad.ndim)

        sum_axes = []
        for axis in range(outGrad.ndim):
            s1 = x_shape[axis]
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

    def compute(self, *inputs: Tuple[TensorTuple]) -> Tensor:
        assert len(inputs) == 1
        assert isinstance(inputs[0], TensorTuple)
        ref_shape = inputs[0][0].shape
        for x in inputs[0]:
            assert x.shape == ref_shape

        flat = [x.value() for x in inputs[0]]
        y = np.stack(flat, axis=self.axis)
        return y

    def gradients(self, node, outGrad: Tensor):
        assert isinstance(outGrad, Tensor)
        assert isinstance(node.inputs[0], TensorTuple)
        x = node.inputs[0]
        dx = unstack(outGrad, axis=self.axis)

        assert len(dx) == len(x)
        return dx
    
class TensorUnstack(Operator):
    def __init__(self, axis=0):
        self.axis = axis

    def compute(self, *inputs: Tuple[Tensor]):
        assert isinstance(inputs[0], Tensor)
        y = np.unstack(inputs[0].value(), axis=self.axis)
        return tuple(Tensor(a) for a in y)

    def gradients(self, node, outGrad):
        dx = stack(outGrad, axis=self.axis)
        return dx
        
class TensorConcatenate(Operator):
    def __init__(self, axis=None):
        self.axis = axis

    def compute(self, *inputs: Tuple[TensorTuple]) -> np.ndarray:
        a = [x.value() for x in inputs[0]]
        y = np.concatenate(a, axis=self.axis)
        return y

    def gradients(self, node, outGrad:Tensor) -> TensorTuple:
        assert isinstance(outGrad, Tensor)
        x = node.inputs[0]
        if self.axis is None:
            x_shape = x[0].shape
            dx = split(outGrad, len(x), axis=0)
            dx = [reshape(a, x_shape) for a in dx]
            assert len(dx) == len(x)
            return make_tuple(*dx)            
        else:
            dx = split(outGrad, len(x), axis=self.axis)
            assert len(dx) == len(x)
            return dx
    
class TensorSplit(Operator):
    def __init__(self, num_splits, axis=0):
        self.num_splits = num_splits
        self.axis = axis

    def compute(self, *inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        x = inputs[0].value()
        y = np.split(x, self.num_splits, axis=self.axis)
        assert len(y) == self.num_splits
        return tuple([Tensor(a) for a in y])

    def gradients(self, node, outGrad):
        assert isinstance(outGrad, TensorTuple)
        assert len(outGrad) == self.num_splits
        x = node.inputs[0].value()

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
def einsum(equation, *inputs):
    return TensorEinsteinSum(equation).tensor(*inputs)
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
def mean(a, axis=None, keepdims=False):
    return TensorMean(axis=axis, keepdims=keepdims).tensor(a)
def power(a, power):
    return TensorPower(power).tensor(a)
def max(a, axis=None):
    return TensorMax(axis=axis).tensor(a)
def neg(a):
    return TensorNegate().tensor(a)
def norm(a, axis=None):
    return TensorNorm(axis=axis).tensor(a)
def sqrt(a):
    return TensorSqrt().tensor(a)

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
def concatenate(inputs: Sequence[Tensor], axis=None):
    return TensorConcatenate(axis).tensor(make_tuple(*inputs))
def split(a, num_splits, axis=0):
    return TensorSplit(num_splits, axis).tensor_tuple(a)
def vstack(arr: Sequence[Tensor]):
    return concatenate(arr, axis=0)
def vsplit(a, num_splits):
    return split(a, num_splits, axis=0)