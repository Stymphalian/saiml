
import numpy as np
import math
from typing import *
from utils import conv
from .base import Operator, Tensor
import autograd2 as ag
from devices import xp


class TensorConvolve2D(Operator):
    def __init__(self, stride, padding):
        assert(stride >= 1)
        assert(padding >= 0)
        self.stride = stride
        self.padding = padding

    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        kernel = inputs[1].value()
        return conv.convolve2d(
            x, kernel, self.stride, self.padding)

    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        kernel = node.inputs[1].value()
        dx, dk = conv.convolve2d_gradient(
            x, kernel, outGrad.value(), self.stride, self.padding)
        return Tensor(dx), Tensor(dk)
    
class TensorConvolve2DTranspose(Operator):
    def __init__(self, stride, padding):
        assert(stride >= 1)
        assert(padding >= 0)
        self.stride = stride
        self.padding = padding

    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        kernel = inputs[1].value()

        assert(kernel.ndim ==  3)
        assert(kernel.shape[1] == kernel.shape[2])
        return conv.convolve2d_transpose(
            x, kernel, self.stride, self.padding)

    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        kernel = node.inputs[1].value()
        dy = outGrad.value()
        dx, dk = conv.convolve2d_transpose_gradient(
            x, kernel, dy, self.stride, self.padding)
        return Tensor(dx), Tensor(dk)

from loss import cross_entropy_loss as loss_cross_entropy_loss
from loss import cross_entropy_loss_derivative as loss_cross_entropy_loss_derivative
class TensorCrossEntropyLoss(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        pred = inputs[0].value()
        true = inputs[1].value()
        pred = pred.reshape((1,) + pred.shape)
        true = true.reshape((1,) + true.shape)
        return loss_cross_entropy_loss(true, pred).reshape(1)

    def gradients(self, node, outGrad):
        pred = node.inputs[0].value()
        true = node.inputs[1].value()
        pred = pred.reshape((1,pred.size,1))
        true = true.reshape((1,true.size,1))
        d_pred = loss_cross_entropy_loss_derivative(true, pred)
        d_pred = d_pred.reshape(node.inputs[0].value().size)
        return (outGrad * d_pred, outGrad * d_pred)
    


def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))
def _positive_sigmoid_derivative(x: Tensor):
    # one = ag.Tensor(1.0)
    # s =  one / (one + ag.exp(-x))
    # return (s * (ag.Tensor(1) - s))

    s = 1.0 / (1.0 + np.exp(-x))
    return (s * (1.0 - s))
def _negative_sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))
def _negative_sigmoid_derivative(x: Tensor):
    # s = ag.exp(x) / (1.0 + ag.exp(x))
    # return (s * (1.0 - s))
    s = np.exp(x) / (1.0 + np.exp(x))
    return (s * (1.0 - s))
class TensorSigmoid(Operator):
    def compute(self, *inputs: Tuple[Tensor]):
        # x = inputs[0].value()
        # y = _positive_sigmoid(x)
        # return y
        x = inputs[0].value()
        y = np.where(
            x >= 0,
            _positive_sigmoid(x),
            _negative_sigmoid(x)
        )
        return y
    def gradients(self, node, outGrad):
        # x = node.inputs[0]
        # return _positive_sigmoid_derivative(x) * outGrad
        x = node.inputs[0]
        dy = outGrad.value()
        dx = np.where(
            x.value() >= 0,
            _positive_sigmoid_derivative(x.value()) * dy,
            _negative_sigmoid_derivative(x.value()) * dy
        )
        return Tensor(dx)
    
class TensorMaxPool(Operator):
    def __init__(self, kernel_size, stride, padding):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        return conv.max_pool2d(x, self.kernel_size, self.stride, self.padding)
    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        dy = outGrad.value()
        dx = conv.max_pool2d_gradient(x, self.kernel_size, dy, self.stride, self.padding)
        return Tensor(dx)
    
class TensorInverseDropout(Operator):
    def __init__(self, is_training, p=0.5, rng_seed=None):
        self.is_training = is_training
        self.p = p
        self.rng_seed = rng_seed
        self.rng = xp.random.default_rng(rng_seed)

        # TODO: This is probably gonna be a source of a bug, because it makes
        # this operator stateful.
        self.last_rand = None
    
    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        if self.is_training:
            self.last_rand = self.rng.binomial(1, self.p, size=x.shape)
            y = x * self.last_rand
            return y
        else:
            return x
        
    def gradients(self, node, outGrad):
        x = node.inputs[0]
        if self.is_training:
            # scale the gradient by the inverse of the probability
            # This is so that we can avoid doing this during test time
            dx = outGrad * ag.Tensor(self.last_rand)
            dx = dx / self.p

            assert dx.shape == x.shape
            return dx
        else:
            assert outGrad.shape == x.shape
            return outGrad

# from loss import (
#     softmax_internal,
#     softmax_derivative_vectorized
# )
# class TensorSoftmax(Operator):
#     def compute(self, *inputs: Tuple[Tensor]):
#         x = inputs[0].value()
#         return softmax_internal(x)
#     def gradients(self, node, outGrad):
#         x = node.inputs[0].value()
#         return outGrad * softmax_derivative_vectorized(x)

# def cross_entropy_loss(pred, true):
#     return TensorCrossEntropyLoss().tensor(pred, true)
def mean_square_error(y_pred, y_true, axis=None):
    return ag.mean(ag.power(y_true - y_pred, 2), axis=axis)
def cross_entropy_loss(y_pred, y_true, axis=None):
    return -ag.summation(y_true * ag.log(y_pred), axis=axis)
def convolve2d(x, kernel, stride=1, padding=0):
    return TensorConvolve2D(stride, padding).tensor(x, kernel)
def convolve2d_transpose(x, kernel, stride=1, padding=0):
    return TensorConvolve2DTranspose(stride, padding).tensor(x, kernel)
def sigmoid(x):
    return TensorSigmoid().tensor(x)
    # return ag.Tensor(1) / (ag.exp(-x) + 1.0)
def softplus(x):
    return ag.log(ag.exp(x) + 1.0)
def softmax(x, axis=None):
    shiftx = x - ag.max(x, axis=axis, keepdims=True)
    exps = ag.exp(shiftx)
    return exps / ag.summation(exps, axis=axis, keepdims=True)
def log_softmax(x, axis=None):
    exps = ag.exp(x)
    z = ag.summation(exps, axis=axis, keepdims=True)
    z = ag.log(z)
    y = x - z
    return y
def relu(x, axis=None, keepdims=True):
    return (x + ag.norm(x, axis=axis, keepdims=keepdims)) / 2.0
def max_pool2d(x, kernel_size, stride=1, padding=0):
    return TensorMaxPool(kernel_size, stride, padding).tensor(x)
def variance(x, axis=None, mean=None):
    if mean is None:
        mean = ag.mean(x, axis=axis)
    z1 = x - mean
    return ag.mean(z1*z1, axis=axis)
def std(x, axis=None):
    return ag.sqrt(variance(x, axis=axis))
def batch_matmul(a, b):
    if a.ndim == b.ndim:
        return ag.matmul(a, b)
    assert a.ndim >= 3 or b.ndim >= 3
    # elif a.ndim <= 2 and b.ndim <= 2:
    #     return ag.matmul(a, b)
    
    if a.ndim < b.ndim:
        a_shape = (1,)*(b.ndim - a.ndim) + a.shape
        a = ag.reshape(a, a_shape)
        return ag.matmul(a, b)
    elif a.ndim > b.ndim:
        if b.ndim == 1:
            b_shape = (1,)*(a.ndim - b.ndim - 1) + (b.shape[0], 1)
            # b = ag.reshape(b, (b.shape[0], 1))
        else:
            b_shape = (1,)*(a.ndim - b.ndim) + b.shape
        b = ag.reshape(b, b_shape)
        return ag.matmul(a, b)

    # if a.ndim == 3 or b.ndim == 3:
    #     # (
    # elif a.ndim == 2 or b.ndim == 2:
    # else:


    # assert a.ndim <= 3
    # assert b.ndim <= 3
    # if a.ndim == 3 or b.ndim == 3:
    #     batch_size = a.shape[0] if a.ndim == 3 else b.shape[0]
        
    #     # TODO: See if we can switch to just purely use einsum equations
    #     # instead of broadcasting/shapping the input
    #     if a.ndim == 2:
    #         a = ag.broadcast(a, (batch_size, a.shape[0], a.shape[1]))
    #     elif a.ndim == 1:
    #         a = ag.broadcast(a, (batch_size, 1, a.shape[0]))
    #     if b.ndim == 2:
    #         b = ag.broadcast(b, (batch_size, b.shape[0], b.shape[1]))
    #     elif b.ndim == 1:
    #         b = ag.reshape(b, (b.shape[0], 1))
    #         b = ag.broadcast(b, (batch_size, b.shape[0], 1))
    #     return ag.einsum("Bij,Bjk->Bik", a, b)
    
    # else:
    #     assert a.ndim == 2 or b.ndim == 2
    #     if a.ndim > b.ndim:
    #         return ag.einsum("Bi,i->B", a, b)
    #     elif a.ndim < b.ndim:
    #         return ag.einsum("i,Bi->B", a, b)
    #     else:
    #         return ag.einsum("Bi,Bi->B", a, b)

# def convolve2d_input(x, kernel_size, stride=1, padding=0):

def mask_fill(x, mask, value):
    # TODO: Make this more efficient, unneeded np.where to convert to value_mask
    assert isinstance(mask, ag.Tensor)

    if isinstance(mask, ag.Tensor):
        mask = mask.value()
    if mask.shape != x.shape:
        mask_shape = (1,) * (x.ndim - mask.ndim) + mask.shape
        mask = np.reshape(mask, mask_shape)
        mask = np.broadcast_to(mask, x.shape)
    value_mask = np.where(mask, value, 0)
    assert value_mask.shape == x.shape

    z = ag.where(mask, ag.Tensor(value_mask), x)
    return z

def inverse_dropout(x, is_training, p=0.5, rng_seed=None):
    return TensorInverseDropout(is_training, p, rng_seed).tensor(x)