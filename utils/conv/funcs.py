from .iterative import (
    _convolve2d_transpose_iterative,
    _convolve2d_transpose_gradient_iterative,
    _max_pool2d_iterative,
    _max_pool2d_gradient_iterative,
)
from .vectorized import (
    _convolve2d_vectorized,
    _convolve2d_gradient_vectorized,
    _max_pool2d_vectorized,
    _max_pool2d_gradient_vectorized
)

def convolve2d(x, kernel, stride=1, padding=0, dilate=0):
    return _convolve2d_vectorized(x, kernel, stride=stride, padding=padding)

def convolve2d_gradient(x, kernel, outGrad, stride=1, padding=0, dilate=0):
    return _convolve2d_gradient_vectorized(
        x, kernel, outGrad, stride=stride, padding=padding)

def convolve2d_transpose(y, kernel, stride=1, padding=0, outer_padding=0):
    return _convolve2d_transpose_iterative(
        y,
        kernel,
        stride=stride,
        padding=padding,
        outer_padding=outer_padding)
    
def convolve2d_transpose_gradient(y, kernel, outGrad, stride=1, padding=0, outer_padding=0):
    return _convolve2d_transpose_gradient_iterative(
        y,
        kernel,
        outGrad,
        stride=stride,
        padding=padding,
        outer_padding=outer_padding)

def max_pool2d(x, kernel_size, stride=1, padding=0):
    return _max_pool2d_vectorized(
        x,
        kernel_size,
        stride=stride,
        padding=padding
    )
    
def max_pool2d_gradient(x, kernel_size, outGrad, stride=1, padding=0):
    return _max_pool2d_gradient_vectorized(
        x,
        kernel_size,
        outGrad,
        stride=stride,
        padding=padding
    )