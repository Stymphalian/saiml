
import numpy as np
from typing import *
from utils import conv
from .base import Operator, Tensor
import autograd2 as ag


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
        dx, dk = conv.convolve2d_gradient(
            x, kernel, outGrad.value(), self.stride, self.padding, self.dilate)
        return Tensor(dx), Tensor(dk)
    
class TensorConvolve2DTranspose(Operator):
    def __init__(self, stride, padding, outer_padding):
        assert(stride >= 1)
        assert(padding >= 0)
        self.stride = stride
        self.padding = padding
        self.outer_padding = outer_padding

    def compute(self, *inputs: Tuple[Tensor]):
        x = inputs[0].value()
        kernel = inputs[1].value()

        assert(kernel.ndim ==  3)
        assert(kernel.shape[1] == kernel.shape[2])
        return conv.convolve2d_transpose(
            x, kernel, self.stride, self.padding, self.outer_padding)

    def gradients(self, node, outGrad):
        x = node.inputs[0].value()
        kernel = node.inputs[1].value()
        dy = outGrad.value()
        dx, dk = conv.convolve2d_transpose_gradient(
            x, kernel, dy, self.stride, self.padding, self.outer_padding)
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
def mean_square_error(y_pred, y_true):
    return ag.mean(ag.power(y_true - y_pred, 2))
def cross_entropy_loss(y_pred, y_true):
    return -ag.summation(y_true * ag.log(y_pred))
def convolve2d(x, kernel, stride=1, padding=0, dilate=0):
    return TensorConvolve2D(stride, padding, dilate).tensor(x, kernel)
def convolve2d_transpose(x, kernel, stride=1, padding=0, outer_padding=0):
    return TensorConvolve2DTranspose(stride, padding, outer_padding=outer_padding).tensor(x, kernel)
def sigmoid(x):
    return TensorSigmoid().tensor(x)
    # return ag.Tensor(1) / (ag.exp(-x) + 1.0)
def softplus(x):
    return ag.log(ag.exp(x) + 1.0)
def softmax(x):
    shiftx = x - ag.max(x)
    exps = ag.exp(shiftx)
    return exps / ag.summation(exps)
def relu(x):
    return (x + ag.norm(x)) / 2.0
def max_pool2d(x, kernel_size, stride=1, padding=0):
    return TensorMaxPool(kernel_size, stride, padding).tensor(x)
def variance(x):
    z1 = x - ag.mean(x)
    return ag.mean(z1*z1)
def std(x):
    return ag.sqrt(variance(x))

# def convolve2d_input(x, kernel_size, stride=1, padding=0):

