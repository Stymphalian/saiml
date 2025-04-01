
import numpy as np
from typing import *
from ..base import Operator, Tensor

class TensorReshape(Operator):
    def __init__(self, shape):
        self.shape = shape
    def compute(self, *inputs: Tuple[Tensor]):
        return inputs[0].value().reshape(self.shape)
    def gradients(self, node, outGrad):
        dx = outGrad.reshape(node.inputs[0].value().shape)
        assert dx.shape == node.inputs[0].shape
        return dx
    
# class TensorVerticalStack(Operator):
#     def compute(self, *inputs: Tuple[Tensor]):
#         # Only allow stacking of the same shape.
#         for x in inputs:
#             assert x.shape == inputs[0].shape
#         flat = np.array([x.value() for x in inputs])
#         return np.vstack(flat)

#     def gradients(self, node, outGrad):
#         x = node.inputs
#         dx = np.vsplit(outGrad, len(x))
#         return tuple(dx)
    
# class TensorVerticalSplit(Operator):
#     def __init__(self, num_splits, axis=0):
#         self.num_splits = num_splits
#         self.axis = axis
#     def compute(self, *inputs: Tuple[Tensor]):
#         x = inputs[0].value()
#         out = np.split(x, self.num_splits, axis=self.axis)
#         return out
#     def gradients(self, node, outGrad):
#         assert len(outGrad) == self.num_splits
#         x = node.inputs[0].value()
#         for grad in outGrad:
#             assert grad[self.axis].shape == x[self.axis].shape
#         dx = np.vstack(outGrad)
#         return dx
    
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

        dx = np.sum(outGrad, axis=tuple(sum_axes))
        dx = np.reshape(dx, x.shape)
        assert dx.shape == x.shape, f"dx shape: {dx.shape}, x shape: {x.shape}"
        return dx
    
class TensorTranspose(Operator):
    def __init__(self, axis=None):
        self.axis = axis
    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        return np.transpose(x, self.axis)
    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        dx = np.transpose(outGrad, self.axis)
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
        x = node.inputs[0].value()

        axis = self.axis
        if self.axis is None:
            axis = x.ndim - 1

        newShape = list(x.shape)
        newShape.insert(axis + 1, self.repeats)
        newShape = tuple(newShape)
        outGrad = outGrad.reshape(newShape)
        dx = np.sum(outGrad, axis=axis+1, keepdims=True)
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

        dx = np.reshape(outGrad, new_shape)
        dx = np.sum(dx, axis=sum_axes)
        dx = np.reshape(dx, x.shape)
        return dx
    
class TensorStack(Operator):
    def __init__(self, axis=0):
        self.axis = axis

    def compute(self, *inputs: Tuple[Tensor]):
        # Only allow stacking of the same shape.
        for x in inputs:
            assert x.shape == inputs[0].shape
        # flat = np.array()
        flat = [x.value() for x in inputs]
        y = np.stack(flat, axis=self.axis)
        return y

    def gradients(self, node, outGrad):
        x = node.inputs
        x_shape = x[0].shape
        dx = np.split(outGrad, len(x), axis=self.axis)
        dx = [np.reshape(a, x_shape) for a in dx]

        assert len(dx) == len(x)
        return tuple(dx)
    
class TensorUnstack(Operator):
    def __init__(self, axis=0):
        self.axis = axis

    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        return np.unstack(x, axis=self.axis)

    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        out = np.split(outGrad, x.shape[self.axis], axis=self.axis)
        out = [np.reshape(a, outGrad.shape[1:]) for a in out]
        dx = np.stack(out, axis=self.axis)
        dx = np.reshape(dx, x.shape)
        return dx
        
class TensorConcatenate(Operator):
    def __init__(self, axis=None):
        self.axis = axis

    def compute(self, *inputs: Tuple[Tensor]):
        a = [x.value() for x in inputs]
        y = np.concatenate(a, axis=self.axis)
        return y

    def gradients(self, node, outGrad):
        x = node.inputs
        if self.axis is None:
            x_shape = x[0].shape
            dx = np.split(outGrad, len(x), axis=0)
            dx = [np.reshape(a, x_shape) for a in dx]
        else:
            dx = np.split(outGrad, len(x), axis=self.axis)

        assert len(dx) == len(x)
        return tuple(dx)
    
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
        assert len(outGrad) == self.num_splits
        x = node.inputs[0].value()
        assert self.num_splits == len(outGrad)

        out = np.split(outGrad, self.num_splits, axis=0)
        out = [np.reshape(a, outGrad.shape[1:]) for a in out]
        dx = np.stack(out, axis=self.axis)
        dx = np.reshape(dx, x.shape)

        assert dx.shape == x.shape, f"dx shape: {dx.shape}, x shape: {x.shape}"
        return dx
    
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
def stack(inputs, axis=0):
    return TensorStack(axis=axis).tensor(*inputs)
def unstack(a, axis=0):
    return TensorUnstack(axis=axis).tensor(a)
def concatenate(arr, axis=None):
    return TensorConcatenate(axis).tensor(*arr)
def split(a, num_splits, axis=0):
    return TensorSplit(num_splits, axis).tensor(a)
def vstack(arr):
    return concatenate(arr, axis=0)
def vsplit(a, num_splits):
    return split(a, num_splits, axis=0)