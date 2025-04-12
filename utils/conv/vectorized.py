from devices import xp
from .conv_utils import *

def _convolve2d_vectorized(x, kernel, stride=1, padding=0, dilate=0):
    assert kernel.ndim == 3
    assert x.ndim == 3
    assert x.shape[0] == kernel.shape[0]

    new_height, new_width = get_new_height_width(
        x.shape, kernel.shape, stride, padding, dilate)

    rows, cols = get_convolution_positions(x.shape, kernel.shape, stride, padding)
    x_pad = utils.zero_pad(x, padding, axes=(1,2))
    x_prime = x_pad[:, rows, cols]
    flat_kernel = kernel.reshape(kernel.shape[0], -1, 1)
    z = xp.matmul(x_prime, flat_kernel)
    z = xp.sum(z, axis=0)
    z = xp.reshape(z, (1, new_height, new_width))
    return z

def _convolve2d_gradient_vectorized(x, kernel, outGrad, stride=1, padding=0, dilate=0):
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

