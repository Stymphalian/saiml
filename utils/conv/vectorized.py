import math
import numpy 
from devices import xp
from .conv_utils import *

def _convolve2d_vectorized2(x, kernel, stride=1, pad=0):
    """
    Convolve the input X with the kernel. Must have matching channels
    X.shape == (...b, ch, xh, xw)
    kernel.shape == (...ks, ch, kh, kw)
    Returns the convolved output of shape (...b, ks, nh, nw)
    """
    assert kernel.ndim >= 3, "Kernel must be atleast (..., ch, kh, kw)"
    assert x.ndim >= 3, "X must be atleast (..., ch, xh, xw)"
    
    x_shape = x.shape[-3:]
    k_shape = kernel.shape[-3:]
    assert x_shape[0] == k_shape[0]
    ks = int(numpy.prod(kernel.shape[:-3]))       # number of kernels, if any

    new_height, new_width = get_conv2d_height_width(x_shape, k_shape, stride, pad)
    # Get row,cols array indices of the convolutions positions
    # (rows/cols).shape == (new_height*new_width, kh*kw)
    rows, cols = get_convolution_positions(x_shape, k_shape, stride, pad)
    # pad the spatial dimensions
    x_pad = utils.zero_pad(x, pad, axes=(-1,-2))                                # (...b, ch, xh, xw)
    # extract out conv subarrays from x
    x_prime = x_pad[..., rows, cols]                                            # (...b, ch, nh*nw, kh*kw)        

    # keep prev dimensions, add one for num_kernels
    new_x_shape = x_prime.shape[:-3] + (1,) + x_prime.shape[-3:]                # (...b, 1, ch, nh*nw, kh*kw) 
    # flatten the kernel spatial dimensions, but keep previous dimensions
    new_kernel_shape = (ks,) + (kernel.shape[-3],-1, 1)                         # (ks, ch, kh*kw, 1)
    x_prime = xp.reshape(x_prime, new_x_shape)                                  # (...b, 1, ch, nh*nw, kh*kw) 
    flat_kernel = xp.reshape(kernel, new_kernel_shape)                          # (     ks, ch, kh*kw,     1)

    z = xp.matmul(x_prime, flat_kernel)                                         # (...b, ks, ch, nh*nw, 1)
    z = xp.sum(z, axis=-3)  # sum over the channels                             # (...b, ks, nh*nw, 1)
    # reshape to keep prev dimensions but reshape into new spatial dims
    z = xp.reshape(z, z.shape[:-2] + (new_height, new_width))                   # (...b, ks, nh, nw)
    # TODO: Should I reshape Z back to (...b, ...ks, nh, nw) or leave it?
    return z

def _convolve2d_vectorized(x, kernel, stride=1, pad=0):
    """
    Convolve the input X with the kernel. Must have matching channels
    X.shape == (...b, ch, xh, xw)
    kernel.shape == (...k, ch, kh, kw)
    Returns the convolved output of shape (...b, k, nh, nw)
    """
    assert kernel.ndim >= 3, "Kernel must be atleast (..., ch, kh, kw)"
    assert x.ndim >= 3, "X must be atleast (..., ch, xh, xw)"
    
    x_shape = x.shape[-3:]
    k_shape = kernel.shape[-3:]
    assert x_shape[0] == k_shape[0]
    k = int(numpy.prod(kernel.shape[:-3]))       # number of kernels, if any
    ch = kernel.shape[-3]
    ks = k*ch
    new_height, new_width = get_conv2d_height_width(x_shape, k_shape, stride, pad)

    # pad the spatial dimensions
    C = vectorize_kernel(x.shape, kernel, stride, pad)               # (...k, ch, nh*nw, xh1*xw1)
    C = xp.reshape(C, (k, ch) + C.shape[-2:])                        # (k, ch, nh*nw, xh1*xw1)
    x_pad = utils.zero_pad(x, pad, axes=(-1,-2))                     # (...b, ch, xh1, xw1)

    # reshape x_pad into proper form
    flat_x = x_pad.flatten().reshape(x.shape[:-3] + (1, ch, -1, 1))  # (...b, 1, ch, xh1*xw1, 1)
    z = xp.matmul(C, flat_x)                                         # (...b, k, ch, nh*nw, 1)

    # reshape z into proper form
    z = xp.sum(z, axis=-3)                                           # (...b, k, nh*nw, 1)
    z = xp.reshape(z, z.shape[:-2] + (new_height, new_width))        # (...b, k, nh, nw)
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

    x_shape = x.shape[-3:]
    ch, xh, xw = x_shape
    k_shape = kernel.shape[-3:]
    ch, kh, kw = k_shape
    assert x_shape[0] == k_shape[0]
    k = int(numpy.prod(kernel.shape[:-3]))       # number of kernels, if any
    ch = kernel.shape[-3]
    ks = k*ch
    nh, nw = get_conv2d_height_width(x_shape, k_shape, stride, pad)

    # pad the spatial dimensions
    xpad = utils.zero_pad(x, pad, axes=(-1,-2))                                 # (...b, ch, xh, xw)
    C = vectorize_kernel(x.shape, kernel, stride, pad)                          # (...k, ch, nh*nw, xh*xw)
    C = xp.reshape(C, (k, ch) + C.shape[-2:])                                   # (k, ch, nh*nw, xh*xw)
    CT = xp.swapaxes(C, -1, -2)                                                 # (k, ch, xh*xw, nh*nw)

    # dx
    dy = outGrad                                                                # (...b, k, nh, nw)
    dy = xp.reshape(dy, dy.shape[:-3] + (k, 1, -1, 1))                          # (...b, k, 1, nh*nw, 1)

    dx = xp.matmul(CT, dy)                                                      # (...b, k, ch, xh*xw, 1)
    dx = xp.sum(dx, axis=-4)                                                    # (...b, ch, xh*xw, 1)
    dx = xp.reshape(dx, xpad.shape)                                             # (...b, ch, xh, xw)
    if pad > 0:
        dx = dx[..., pad:-pad, pad:-pad]

    # dk
    rows, cols = get_convolution_positions(x_shape, k_shape, stride, pad)       # (nh*nw, kh*kw)
    x2 = xpad[..., rows, cols]                                                  # (...b, ch, nh*nw, kh*kw)
    x2 = xp.swapaxes(x2, -1, -2)                                                # (...b, ch, kh*kw, nh*nw)
    x2 = xp.reshape(x2, x2.shape[:-3] + (1,ch) + x2.shape[-2:])                 # (...b, 1, ch, kh*kw, nh*nw))
    dy = outGrad                                                                # (...b, k, nh, nw)
    dy = xp.reshape(dy, dy.shape[:-3] + (k, 1, -1, 1))                          # (...b, k, 1, nh*nw, 1)

    dk = xp.matmul(x2, dy)                                                      # (...b, k, ch, kh*kw, 1)
    dk = xp.sum(dk, axis=0)                                                     # (k, ch, kh*kw, 1)
    dk = xp.reshape(dk, kernel.shape)                                           # (...k, ch, kh, kw)
    
    assert dx.shape == x.shape
    assert dk.shape == kernel.shape
    return (dx, dk)

def _convolve2d_transpose_vectorized(y, kernel, stride=1, pad=0):
    assert y.ndim >= 3, "Y must be atleast (...b, ch, yh, yw)"
    assert kernel.ndim >= 3, "Kernel must be atleast (...k, ch, kh, kw)"
    assert y.shape[-3] == kernel.shape[-3], "Channels must match between y and kernel"

    yh, yw = y.shape[-2:]
    kh, kw = kernel.shape[-2:]
    # We want to calculate the stride for the conv_transpose such that we 
    # can get the same shape as the forward input X.
    # From the formula for calculating the normal forward conv2d output shape
    #   y = (x - k + 2p / s) + 1  (1)
    # isolate for x in that equation.
    #   x = (y - 1)*s + k - 2p    (2)
    xh = (yh-1)*stride + kh - 2*pad
    xw = (yw-1)*stride + kw - 2*pad
    xh1 = xh + 2*pad
    xw1 = xw + 2*pad

    x_shape = (xh, xw)
    k_shape = (kh, kw)
    k = int(numpy.prod(kernel.shape[:-3]))       # number of kernels, if any
    ch = kernel.shape[-3]
    ks = k*ch
    # new_height, new_width = get_conv2d_height_width(x_shape, k_shape, stride, pad)

    # pad the spatial dimensions
    C = vectorize_kernel(x_shape, kernel, stride, pad)    # (...k, ch, nh*nw, xh*xw)
    C = xp.reshape(C, (k,ch) + C.shape[-2:])              # (k, ch, nh*nw, xh*xw)
    CT = xp.swapaxes(C, -1, -2)                           # (k, ch, xh*xw, nh*nw)

    # Reshape y into the proper form                       # (...b, ch, nh, nw) == y
    flat_y = xp.reshape(y, y.shape[:-3] + (1, ch, -1, 1))  # (...b, 1, ch, nh*nw, 1)
    x = xp.matmul(CT, flat_y)                              # (...b, k, ch, xh*xw, 1)
    
    # reshape x into proper form
    x = xp.sum(x, axis=-3)                                # (...b, k, xh*xw, 1)
    x = xp.reshape(x, x.shape[:-2] + (xh1, xw1))          # (...b, k, xh, xw)

    # handle removal of padding
    if pad > 0 :
        x = x[..., pad:-pad, pad:-pad]
    return x

def _convolve2d_transpose_gradient_vectorized(y, kernel, outGrad, stride=1, pad=0):
    assert y.ndim >= 3, "Y must be atleast (...b, ch, yh, yw)"
    assert kernel.ndim >= 3, "Kernel must be atleast (...k, ch, kh, kw)"
    assert outGrad.ndim >= 3, "OutGrad must be atleast (...b, k, xh, xw)"
    assert y.shape[-3] == kernel.shape[-3], "Channels must match between y and kernel"

    yh, yw = y.shape[-2:]
    kh, kw = kernel.shape[-2:]
    # We want to calculate the stride for the conv_transpose such that we 
    # can get the same shape as the forward input X.
    # From the formula for calculating the normal forward conv2d output shape
    #   y = (x - k + 2p / s) + 1  (1)
    # isolate for x in that equation.
    #   x = (y - 1)*s + k - 2p    (2)
    xh = (yh-1)*stride + kh - 2*pad
    xw = (yw-1)*stride + kw - 2*pad
    assert ((xh, xw) == outGrad.shape[-2:]), "Output shape does not match expected output shape"

    k = int(numpy.prod(kernel.shape[:-3]))       # number of kernels, if any
    ch = kernel.shape[-3]
    outGradPad = utils.zero_pad(outGrad, pad, axes=(-1,-2))


    # pad the spatial dimensions
    C = vectorize_kernel((xh, xw), kernel, stride, pad)                         # (...k, ch, yh*yw, xh*xw)
    C = xp.reshape(C, (k,ch) + C.shape[-2:])                                    # (k, ch, yh*yw, xh*xw)
    CT = xp.swapaxes(C, -1, -2)                                                 # (k, ch, xh*xw, yh*yw)

    # dy (convolve the kernel over the gradient X to get dY)
    dx = outGradPad                                                             # (...b, k, xh, xw)
    dx = xp.reshape(dx, dx.shape[:-3] + (k, 1, -1, 1))                          # (...b, k, 1, xh*xw, 1)
    C                                                                           # (k, ch, yh*yw, xh*xw)
    dy = xp.matmul(C, dx)                                                       # (...b, k, ch, yh*yw, 1)
    dy = xp.sum(dy, axis=-4)                                                    # (...b, ch, yh*yw, 1)
    dy = xp.reshape(dy, dy.shape[:-2] + (yh, yw))                               # (...b, ch, yh, yw)
    dy = xp.reshape(dy, y.shape)                                                # (...b, ch, yh, yw)

    # dk
    rows, cols = get_convolution_positions((xh,xw), (kh, kw), stride, pad)      # (yh*yw, kh*kw)
    dx = outGradPad[..., rows, cols]                                            # (...b, k, yh*yw, kh*kw)
    dx = xp.swapaxes(dx, -1, -2)                                                # (...b, k, kh*kw, yh*yw)
    dx = xp.reshape(dx, dx.shape[:-3] + (k, 1) + dx.shape[-2:])                 # (...b, k, 1, kh*kw, yh*yw))
    y2 = y                                                                      # (...b, ch, yh, yw)
    y2 = xp.reshape(y2, y2.shape[:-3] + (1, ch, -1, 1))                         # (...b, 1, ch, yh*yw, 1)

    dk = xp.matmul(dx, y2)                                                      # (...b, k, ch, kh*kw, 1)
    dk = xp.sum(dk, axis=0)                                                     # (k, ch, kh*kw, 1)
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

