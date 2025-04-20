import math
import numpy 
from devices import xp
from .conv_utils import *

# def _convolve2d_vectorized2(x, kernel, stride=1, padding=0, dilate=0):
#     assert kernel.ndim == 3
#     assert x.ndim == 3
#     assert x.shape[0] == kernel.shape[0]

#     new_height, new_width = get_conv2d_height_width(
#         x.shape, kernel.shape, stride, padding, dilate)

#     rows, cols = get_convolution_positions(x.shape, kernel.shape, stride, padding)
#     x_pad = utils.zero_pad(x, padding, axes=(1,2))
#     x_prime = x_pad[:, rows, cols]
#     flat_kernel = kernel.reshape(kernel.shape[0], -1, 1)
#     z = xp.matmul(x_prime, flat_kernel)
#     z = xp.sum(z, axis=0)
#     z = xp.reshape(z, (1, new_height, new_width))
#     return z

# TODO: Allow for multiple kernels?
def _convolve2d_vectorized(x, kernel, stride=1, pad=0):
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

def _convolve2d_gradient_vectorized_old(x, kernel, outGrad, stride=1, padding=0, dilate=0):
    assert outGrad.ndim == 3
    assert kernel.ndim == 3
    assert x.ndim == 3
    assert x.shape[0] == kernel.shape[0]
    assert outGrad.shape[0] == 1

    xpad = utils.zero_pad(x, padding, axes=(1,2))

    vkernel = vectorize_kernel(x.shape, kernel, stride, padding)
    vkernel = xp.transpose(vkernel, (0,2,1))
    outGrad = outGrad.flatten().reshape(1,-1,1)
    dx = xp.matmul(vkernel, outGrad).reshape(xpad.shape)
    if padding > 0:
        dx = dx[:, padding:-padding, padding:-padding]

    rows, cols = get_convolution_positions(x.shape, kernel.shape, stride, padding)
    x2 = xpad[:, rows, cols]
    x2 = xp.transpose(x2, (0,2,1))
    dk = xp.matmul(x2, outGrad).reshape(kernel.shape)

    return (dx, dk)

# outGrad.shape == (...b, k*ch, nh, nw)
# TODO: Vectorized version of this DOES NOT WORK. You get OOM errors when doing
# the matmul against the vectorized kernel and the gradient if they are big enough
# The size blows up because we have to create k*ch size kernels.
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
    kernel_shape = kernel.shape[-3:]
    assert x_shape[0] == kernel_shape[0]
    assert outGrad.shape[-3] == int(numpy.prod(kernel.shape[:-3]))

    k = int(numpy.prod(kernel.shape[:-3]))                                      # k
    ch = kernel.shape[-3]                                                       # ch
    ks = k*ch                                                                   # k*ch
    # flatten the kernel to make it easier to work with
    kernel2 = xp.reshape(kernel, (ks,) + kernel.shape[-2:])                     # (k*ch, kh, kw)

    # Make the gradient into the correct shape
    grad2 = outGrad.flatten().reshape(outGrad.shape[:-2] + (-1, 1))             # (...b, k, nh*nw, 1)
    grad2 = xp.expand_dims(grad2, axis=-3)                                      # (...b, k, 1, nh*nw, 1) 
    grad2 = xp.broadcast_to(grad2, grad2.shape[:-4] + (k,ch) + grad2.shape[-2:])# (...b, k, ch, nh*nw, 1)
    grad2 = xp.reshape(grad2, grad2.shape[:-4] + (k*ch,) + grad2.shape[-2:])    # (...b, k*ch, nh*nw, 1)

    # Matmul against the vectorized kernel to get dx
    xpad = utils.zero_pad(x, pad, axes=(-1,-2))                                 # (...b, ch, xh, xw)
    vkernel = vectorize_kernel(x_shape, kernel2, stride, pad)                   # (k*ch, nh*nw, xh*xw)
    vkernel = xp.swapaxes(vkernel, -1, -2)                                      # (k*ch, xh*xw, nh*nw)
                                                                                # (...b, k*ch, nh*nw, 1) == grad2.shape
    dx = xp.matmul(vkernel, grad2)                                              # (...b, k*ch, xh*xw, 1)

    # Sum over the kernels (not including the channels) and expand 
    # back into X spatial dimensions.
    dx = xp.reshape(dx, dx.shape[:-3] + (k, ch) + dx.shape[-2:])                # (...b, k, ch, xh*xw, 1)
    dx = xp.sum(dx, axis=-4)                                                    # (...b, ch, xh*xw, 1)
    dx = xp.reshape(dx, xpad.shape)                                             # (...b, ch, xh, xw)
    if pad > 0:
        dx = dx[..., pad:-pad, pad:-pad]

    rows, cols = get_convolution_positions(x_shape, kernel_shape, stride, pad)  # (nh*nw, kh*kw)
    x2 = xpad[..., rows, cols]                                                  # (...b, ch, nh*nw, kh*kw)
    x2 = xp.swapaxes(x2, -1, -2)                                                # (...b, ch, kh*kw, nh*nw)
    x2 = xp.reshape(x2, x2.shape[:-3] + (1,ch) + x2.shape[-2:])                 # (...b, 1, ch, kh*kw, nh*nw))
    x2 = xp.broadcast_to(x2, x2.shape[:-4] + (k, ch) + x2.shape[-2:])           # (...b, k, ch, kh*kw, nh*nw))
    x2 = xp.reshape(x2, x2.shape[:-4] + (k*ch,) + x2.shape[-2:])                # (...b, k*ch, kh*kw, nh*nw))
                                                                                # (...b, k*ch, nh*nw, 1) == grad2
    dk = xp.matmul(x2, grad2)                                                   # (...b, k*ch, kh*kw, 1)
    dk = xp.sum(dk, axis=0)                                                     # (k*ch, kh*kw, 1)
    dk = xp.reshape(dk, kernel.shape)                                           # (...k, ch, kh, kw)

    assert dx.shape == x.shape
    assert dk.shape == kernel.shape
    return (dx, dk)


def _convolve2d_transpose_vectorized(y, kernel, stride=1, padding=0, outer_padding=0):
    raise NotImplementedError()

def _convolve2d_transpose_gradient_vectorized(y, kernel, outGrad, stride=1, padding=0, outer_padding=0):
    raise NotImplementedError()

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

