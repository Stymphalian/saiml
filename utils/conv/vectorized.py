import math
import numpy 
from devices import xp
from .conv_utils import *

def _conv_forward(x, kernel, stride=1, pad=0):
    """
    Convolve the kernel over the input x.
    x.shape == (...b, k, ch, xh, xw)
    kernel.shape == (...k, ch, kh, kw)
    Returns convolved output (...b, k, nh, nw)
    """
    assert x.ndim >= 4, "X must be atleast (..., k, ch, xh, xw)"
    assert kernel.ndim >= 3, "Kernel must be atleast (..., ch, kh, kw)"
    # assert x.shape[-3] == kernel.shape[-3], "Channels must match between x and kernel"
    
    xh, xw = x.shape[-2:]
    kh, kw = kernel.shape[-2:]
    yh, yw  = get_conv2d_height_width((xh, xw), (kh, kw), stride, pad)
    ch = kernel.shape[-3]
    k = int(numpy.prod(kernel.shape[:-3]))

    assert x.shape[-4] == 1 or x.shape[-4] == k
    assert x.shape[-3] == 1 or x.shape[-3] == ch

    rows, cols = get_convolution_positions((xh, xw), (kh,kw), stride, pad)      # (yh*yw, kh*kw)
    x2 = utils.zero_pad(x, pad, axes=(-1,-2))                                   # (...b, k, ch, xh, xw)
    x2 = x2[..., rows, cols]                                                    # (...b, k, ch, yh*yw, kh*kw)

    k1 = kernel                                                                 # (...k, ch, kh, kw)
    k1 = xp.reshape(k1, (k, ch, kh*kw, 1))                                      # (k, ch, kh*kw, 1)

    y = xp.matmul(x2, k1)                                                       # (...b, k, ch, yh*yw, 1)
    y = xp.reshape(y, y.shape[:-4] + (k,ch) + (yh,yw))                          # (...b, k, ch, yh, yw)
    return y

def _conv_for_kernel(x, k_shape, dy, stride=1, pad=0):
    """
    x.shape == (...b, k, ch, xh, xw)
    dy.shape == (...b, k, ch, nh, nw)
    Returns convolved output (...b, k, ch, kh, kw)
    """
    assert x.ndim >= 4, "X must be atleast (...b, k, ch, xh, xw)"
    assert len(k_shape) >= 2, "K must be atleast (..., kh, kw)"
    assert dy.ndim >= 4, "dy must be atleast (...b, k, ch, yh, yw)"
    xh, xw = x.shape[-2:]
    yh, yw = dy.shape[-2:]
    kh, kw = k_shape[-2:]
    ch = x.shape[-3] if x.shape[-3] != 1 else dy.shape[-3]
    k = x.shape[-4] if x.shape[-4] != 1 else dy.shape[-4]
    # assert ch > 1 or k > 1
    
    xpad = utils.zero_pad(x, pad, axes=(-1,-2))                                 # (...b, k, ch, xh, xw)
    rows, cols = get_convolution_positions((xh, xw), (kh, kw), stride, pad)     # (yh*yw, kh*kw)
    x2 = xpad[..., rows, cols]                                                  # (...b, k, ch, yh*yw, kh*kw)
    x2 = xp.swapaxes(x2, -1, -2)                                                # (...b, k, ch, kh*kw, yh*yw)
    dy2 = dy                                                                    # (...b, k, ch, yh, yw)
    dy2 = xp.reshape(dy2, dy2.shape[:-2] + (-1, 1))                             # (...b, k, ch, yh*yw, 1)

    dk = xp.matmul(x2, dy2)                                                     # (...b, k, ch, kh*kw, 1)
    dk = xp.reshape(dk, dk.shape[:-2] + (kh, kw))                               # (...b, k, ch, kh, kw)
    return dk

def _conv_transpose(y, kernel, stride=1, pad=0, x_shape=None):
    """
    Do the conv tranpose from y using the kernel to get X.
    y.shape == (...b, k, ch, yh, yw)
    kernel.shape (...k, ch, kh, kw)
    returns conv_transposed shape (...b, k, xh, xw)
    """
    assert y.ndim >= 4, "y must be atleast (...b, k, ch, yh, yw)"
    assert kernel.ndim >= 3, "Kernel must be atleast (...k, ch, kh, kw)"
    
    kh, kw = kernel.shape[-2:]
    yh, yw = y.shape[-2:]
    if x_shape is not None:
        xh, xw = x_shape[-2:]
    else:
        xh, xw = get_conv2d_transpose_height_width((yh, yw), (kh, kw), stride, pad)
    ch = kernel.shape[-3]
    k = int(numpy.prod(kernel.shape[:-3]))
    xh1, xw1 = xh + 2*pad, xw + 2*pad

    assert y.shape[-4] == 1 or y.shape[-4] == k
    assert y.shape[-3] == 1 or y.shape[-3] == ch

    C = vectorize_kernel((xh, xw), kernel, stride, pad)                         # (...k, ch, yh*yw, xh*xw)
    C = xp.reshape(C, (k,ch) + C.shape[-2:])                                    # (k, ch, yh*yw, xh*xw)
    C = xp.swapaxes(C, -1, -2)                                                  # (k, ch, xh*xw, yh*yw)
    y2 = y                                                                      # (...b, k?, ch, yh, yw)
    y2 = xp.reshape(y2, y2.shape[:-2] + (-1, 1))                                # (...b, k?, ch, yh*yw, 1)

    x = xp.matmul(C, y2)                                                        # (...b, k, ch, xh*xw, 1)
    x = xp.reshape(x, x.shape[:-4] + (k, ch) + (xh1, xw1))                      # (...b, k, ch, xh, xw)

    # handle removal of padding
    if pad > 0 :
        x = x[..., pad:-pad, pad:-pad]
    return x

def _convolve2d_vectorized(x, kernel, stride=1, pad=0):
    """
    Convolve the input X with the kernel. Must have matching channels
    X.shape == (...b, ch, xh, xw)
    kernel.shape == (...k, ch, kh, kw)
    Returns the convolved output of shape (...b, k, nh, nw)
    """
    assert kernel.ndim >= 3, "Kernel must be atleast (..., ch, kh, kw)"
    assert x.ndim >= 3, "X must be atleast (..., ch, xh, xw)"

    x1 = xp.expand_dims(x, -4)                              # (...b, 1, ch, xh, xw)
    z = _conv_forward(x1, kernel, stride, pad)              # (...b, k, ch, yh, yw)
    z = xp.sum(z, axis=-3)                                  # (...b, k, yh, yw)
    return z

def _convolve2d_gradient_vectorized(x, kernel, outGrad, stride=1, pad=0):
    """
    Return the gradient of the convolved output outGrad with respect to X and kernel
    X.shape == (...b, ch, xh, ww)
    kernel.shape == (...k, ch, kh, kw)
    outGrad.shape == (...b, k, nh, nw)
    Returns dx and dk
    dx.shape == (..., ch, xh, xw)
    dk.shape == (..., ch, kh, kw)
    """
    assert outGrad.ndim >= 3, "outGrad must be atleast (..., num_kernels, nh, nw)"
    assert kernel.ndim >= 3, "Kernel must be atleast (..., ch, kh, kw)"
    assert x.ndim >= 3, "X must be atleast (..., ch, xh, xw)"

    # dx
    dy1 = xp.expand_dims(outGrad, -3)                                           # (...b, k, 1, yh, yw)
    dx = _conv_transpose(dy1, kernel, stride, pad, x_shape=x.shape)             # (...b, k, ch, xh, xw)
    dx = xp.sum(dx, axis=-4)                                                    # (...b, ch, xh, xw)

    # dk
    x1 = xp.expand_dims(x, -4)                                                  # (...b, 1, ch, xh, xw)
    dy1 = xp.expand_dims(outGrad, -3)                                           # (...b, k, 1, yh, yw)
    dk = _conv_for_kernel(x1, kernel.shape, dy1, stride, pad)                   # (...b, k, ch, kh, kw)
    dk = xp.sum(dk, axis=0)                                                     # (k, ch, kh, kw)
    dk = xp.reshape(dk, kernel.shape)                                           # (...k, ch, kh, kw)

    assert dx.shape == x.shape
    assert dk.shape == kernel.shape
    return (dx, dk)

def _convolve2d_transpose_vectorized(y, kernel, stride=1, pad=0):
    assert y.ndim >= 3, "Y must be atleast (...b, ch, yh, yw)"
    assert kernel.ndim >= 3, "Kernel must be atleast (...k, ch, kh, kw)"
    assert y.shape[-3] == kernel.shape[-3], "Channels must match between y and kernel"

    y1 = xp.expand_dims(y, -4)                      # (...b, 1, ch, yh, yw)
    x = _conv_transpose(y1, kernel, stride, pad)    # (...b, k, ch, xh, xw)
    x = xp.sum(x, axis=-3)                          # (...b, k, xh, xw)
    return x

def _convolve2d_transpose_gradient_vectorized(y, kernel, outGrad, stride=1, pad=0):
    assert y.ndim >= 3, "Y must be atleast (...b, ch, yh, yw)"
    assert kernel.ndim >= 3, "Kernel must be atleast (...k, ch, kh, kw)"
    assert outGrad.ndim >= 3, "OutGrad must be atleast (...b, k, xh, xw)"
    assert y.shape[-3] == kernel.shape[-3], "Channels must match between y and kernel"

    # dy
    dx = xp.expand_dims(outGrad, -3)                                            # (...b, k, 1, xh, xw)
    dy = _conv_forward(dx, kernel, stride, pad)                                 # (...b, k, ch, yh, yw)
    dy = xp.sum(dy, axis=-4)                                                    # (...b, ch, yh, yw)

    # dk
    dx1 = xp.expand_dims(outGrad, -3)                                           # (...b, k, 1, xh, xw)
    dy1 = xp.expand_dims(y, -4)                                                 # (...b, 1, ch, yh, yw)   
    dk = _conv_for_kernel(dx1, kernel.shape, dy1, stride, pad)                  # (...b, k, ch, kh, kw)   
    dk = xp.sum(dk, axis=0)                                                     # (k, ch, kh, kw)
    dk = xp.reshape(dk, kernel.shape)                                           # (...k, ch, kh, kw)

    assert dy.shape == y.shape
    assert dk.shape == kernel.shape
    return (dy, dk)


def _pool2d_vectorized(x, kernel_size, pool_func, stride=1, padding=0):
    """
    pool_func takes in input.shape (channels, new_height*new_width, kernel_size*kernel_size)
    where input is the x values of the convolution

    """
    assert x.ndim == 3
    xc, xh, xw = x.shape
    kc, kh, kw = (1, kernel_size, kernel_size)
    new_height = (xh - kh + 2*padding) // stride + 1
    new_width = (xw - kw + 2*padding) // stride + 1

    x = utils.zero_pad(x, padding, axes=(1,2))
    rows, cols = get_convolution_positions(x.shape, (kc, kh, kw), stride)
    x1 = x[:, rows, cols]
    x2 = pool_func(x1)
    # x2 = xp.max(x1, axis=2)
    x3 = xp.reshape(x2, (xc, new_height, new_width))
    return x3

def _pool2d_gradient_vectorized(x, kernel_size, outGrad, pool_fn, stride=1, padding=0):
    assert x.ndim == 3
    xc, xh, xw = x.shape
    kc, kh, kw = (xc, kernel_size, kernel_size)
    new_height = (xh - kh + 2*padding) // stride + 1
    new_width = (xw - kw + 2*padding) // stride + 1
    assert outGrad.shape == (xc, new_height, new_width)
    
    x = utils.zero_pad(x, padding, axes=(1,2))
    vkernel = vectorize_kernel_with_fn(x, (kc, kh, kw), pool_fn, stride)
    assert vkernel.shape == (xc*new_height*new_width, xc*xh*xw)

    dx = xp.matmul(vkernel.T, outGrad.reshape(-1))
    dx = xp.reshape(dx, x.shape)
    return dx

def _max_pool2d_vectorized(x, kernel_size, stride=1, padding=0):
    max_fn = lambda x : xp.max(x, axis=2)
    return _pool2d_vectorized(x, kernel_size, max_fn, stride, padding)
    # assert x.ndim == 3
    # xc, xh, xw = x.shape
    # kc, kh, kw = (1, kernel_size, kernel_size)
    # new_height = (xh - kh + 2*padding) // stride + 1
    # new_width = (xw - kw + 2*padding) // stride + 1

    # x = utils.zero_pad(x, padding, axes=(1,2))
    # rows, cols = get_convolution_positions(x.shape, (kc, kh, kw), stride)
    # xpad = utils.zero_pad(x, padding, axes=(1,2))
    # x1 = xpad[:, rows, cols]
    # x2 = xp.max(x1, axis=2)
    # x3 = xp.reshape(x2, (1, new_height, new_width))
    # return x3

def _max_pool2d_gradient_vectorized(x, kernel_size, outGrad, stride=1, padding=0):
    # max_fn = lambda x: xp.max(x, axis=2)

    def max_fn(flat_vx, rows, cols):
        xc, xh, xw = x.shape
        max_x = xp.argmax(flat_vx, axis=1)
    
        linear = xp.mod(xp.arange(max_x.size), len(rows))
        offset = xp.repeat(xp.arange(xc)*xh, len(rows))
        r = rows[linear, max_x] + offset
        c = cols[linear, max_x]

        flat_x = xp.concatenate(x, axis=0)
        indices = xp.ravel_multi_index((r,c), dims=flat_x.shape)
        x_identity = xp.identity(flat_x.size)
        vk = x_identity[indices]
        return vk

    return _pool2d_gradient_vectorized(
        x,
        kernel_size,
        outGrad,
        max_fn,
        stride=stride,
        padding=padding
    )
    # assert x.ndim == 3
    # xc, xh, xw = x.shape
    # kc, kh, kw = (1, kernel_size, kernel_size)
    # new_height = (xh - kh + 2*padding) // stride + 1
    # new_width = (xw - kw + 2*padding) // stride + 1
    # assert outGrad.shape == (1, new_height, new_width)
    
    # x = utils.zero_pad(x, padding, axes=(1,2))
    # vkernel = vectorize_kernel_maxpool(x, (kc, kh, kw), stride)
    # assert vkernel.shape == (new_height*new_width, xc*xh*xw)

    # dx = xp.matmul(vkernel.T, outGrad.reshape(-1))
    # dx = xp.reshape(dx, x.shape)
    # return dx

def _avg_pool2d_vectorized(x, kernel_size, stride=1, padding=0):
    fn = lambda x : xp.mean(x, axis=2)
    return _pool2d_vectorized(x, kernel_size, fn, stride, padding)

def _avg_pool2d_gradient_vectorized(x, kernel_size, outGrad, stride=1, padding=0):
    # max_fn = lambda x: xp.max(x, axis=2)

    def max_fn(flat_vx, rows, cols):
        xc, xh, xw = x.shape
        max_x = xp.argmax(flat_vx, axis=1)
    
        linear = xp.mod(xp.arange(max_x.size), len(rows))
        offset = xp.repeat(xp.arange(xc)*xh, len(rows))
        r = rows[linear, max_x] + offset
        c = cols[linear, max_x]

        flat_x = xp.concatenate(x, axis=0)
        indices = xp.ravel_multi_index((r,c), dims=flat_x.shape)
        x_identity = xp.identity(flat_x.size)
        vk = x_identity[indices]
        return vk

    return _pool2d_gradient_vectorized(
        x,
        kernel_size,
        outGrad,
        max_fn,
        stride=stride,
        padding=padding
    )

