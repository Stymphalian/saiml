
from .conv_utils import *
import utils

def _convolve2d_iterative(x, kernel, stride=1, padding=0, dilate=0):
    assert x.ndim == kernel.ndim
    assert x.ndim == 3
    spatial_axes = (1,2)
    assert kernel.shape[spatial_axes[0]] == kernel.shape[spatial_axes[1]]
    
    new_height, new_width = get_conv2d_height_width(
        x.shape, kernel.shape, stride, padding, dilate)
    x = utils.zero_dilate(x, dilate, axes=spatial_axes)
    x = utils.zero_pad(x, padding, axes=spatial_axes)
    kc, kh, kw = kernel.shape
    output = xp.zeros((1, new_height, new_width), dtype=x.dtype)
    xc, xh, xw = x.shape

    for row in range(new_height):
        for col in range(new_width):
            h = row * stride
            w = col * stride
            if h + kh > xh or w + kw > xw:
                raise Exception("Kernel is out of bounds")
                continue
            x_slice = x[:, h:h+kh, w:w+kw]
            assert x_slice.shape == kernel.shape
            v = xp.sum(x_slice * kernel)
            output[0,row,col] += v
    return output

def _convolve2d_gradient_iterative(x, kernel, outGrad, stride=1, padding=0, dilate=0):
    assert x.ndim == kernel.ndim
    assert x.ndim == 3
    spatial_axes = (1,2)
    assert kernel.shape[spatial_axes[0]] == kernel.shape[spatial_axes[1]]
    new_height, new_width = get_conv2d_height_width(
        x.shape, kernel.shape, stride, padding, dilate)
    
    x = utils.zero_dilate(x, dilate, axes=spatial_axes)
    x = utils.zero_pad(x, padding, axes=spatial_axes)
    kc, kh, kw = kernel.shape
    xc, xh, xw = x.shape
    
    dx = xp.zeros(x.shape)
    dk = xp.zeros(kernel.shape)

    for row in range(new_height):
        for col in range(new_width):
            h = row * stride
            w = col * stride
            x_slice = x[:, h:h+kh, w:w+kw]
            dY = outGrad[0, row, col]

            if h + kh > xh or w + kw > xw:
                raise Exception("Kernel is out of bounds")
                continue
            dx[:, h:h+kh, w:w+kw] += xp.multiply(kernel, dY)
            dk += xp.multiply(x_slice, dY)
        
    if padding > 0:
        dx = dx[:, padding:-padding, padding:-padding]
    if dilate > 0:
        dx = utils.zero_undilate(dx, dilate, axes=spatial_axes)
    return (dx, dk)

def _convolve2d_transpose_iterative(y, kernel, stride=1, padding=0, outer_padding=0):
    assert y.ndim == kernel.ndim
    assert kernel.ndim == 3
    spatial_axes = (1,2)
    assert kernel.shape[spatial_axes[0]] == kernel.shape[spatial_axes[1]]

    yc, yh, yw = y.shape
    kc, kh, kw = kernel.shape

    # We want to calculate the stride for the conv_transpose such that we 
    # can get the same shape as the forward input X.
    # From the formula for calculating the normal forward conv2d output shape
    #   y = (x - k + 2p / s) + 1  (1)
    # isolate for x in that equation.
    #   x = (y - 1)*s + k - 2p    (2)
    # The equation for the conv2d_transpose outupt shape is this:
    #   x = y + (k-1)*s'          (3)
    # Where s' is the stride we want to determine.
    # Isolate for s'
    #   s' = (x - y) / (k-1)      (4)
    # Substitude (2) into (4) and we can get the desired stride s'
    xh = (yh-1)*stride + kh - 2*padding
    xw = (yw-1)*stride + kw - 2*padding
    stride_row = math.ceil((xh - yh) / (kh -1))
    stride_col = math.ceil((xw - yw) / (kw -1))
    if (stride_row < 0) or (stride_col < 0):
        raise Exception("Stride is negative")
    new_height = yh + (kh - 1)*stride_row
    new_width = yw + (kw - 1)*stride_col

    x = xp.zeros((kc, new_height, new_width), dtype=y.dtype)

    for k in range(kc):
        for row in range(kh):
            for col in range(kw):
                h = row * stride_row
                w = col * stride_col
                x[k, h:h+yh, w:w+yw] += y[0] * kernel[k,row,col]

    if outer_padding > 0:
        pad = outer_padding
        x = x[:, pad:-pad, pad:-pad]

    x = xp.sum(x, axis=0)
    x = x[xp.newaxis, :,:]
    return x

def _convolve2d_transpose_gradient_iterative(y, kernel, outGrad, stride=1, pad=0, outer_padding=0):
    assert y.ndim == kernel.ndim
    assert kernel.ndim == 3
    spatial_axes = (1,2)
    assert kernel.shape[spatial_axes[0]] == kernel.shape[spatial_axes[1]]

    yc, yh, yw = y.shape
    kc, kh, kw = kernel.shape

    xh = (yh-1)*stride + kh - 2*pad
    xw = (yw-1)*stride + kw - 2*pad
    stride_row = math.ceil((xh - yh) / (kh -1))
    stride_col = math.ceil((xw - yw) / (kw -1))
    if (stride_row < 0) or (stride_col < 0):
        return False, None
    new_height = yh + (kh - 1)*stride_row
    new_width = yw + (kw - 1)*stride_col
    if outer_padding > 0:
        outGrad = utils.zero_pad(outGrad, outer_padding, axes=spatial_axes)
    assert outGrad.shape == (1, new_height, new_width), "{} != {}".format(outGrad.shape, (kc, new_height, new_width))
    
    dy = xp.zeros(y.shape)
    dk = xp.zeros(kernel.shape)

    for k in range(kc):
        for row in range(kh):
            for col in range(kw):
                h = row * stride_row
                w = col * stride_col
                out_slice = outGrad[0, h:h+yh, w:w+yw]
                assert (out_slice.shape == y.shape[1:])
                dy += kernel[k,row,col] * out_slice
                dk[k,row,col] += xp.sum(y * out_slice)

                # h = row * stride_row
                # w = col * stride_col
                # out_slice = outGrad[:, h:h+yh, w:w+yw]
                # assert (out_slice.shape[1:] == y.shape)
                # dy += xp.sum(kernel[:,row,col] * out_slice, axis=0)
                # dk[:,row, col] += xp.sum(y * out_slice)

    return (dy, dk)

def _max_pool2d_iterative(x, kernel_size, stride=1, padding=0):
    assert x.ndim == 3
    xc, xh, xw = x.shape
    kc, kh, kw = (1, kernel_size, kernel_size)
    new_height = (xh - kh + 2*padding) // stride + 1
    new_width = (xw - kw + 2*padding) // stride + 1

    xpad = utils.zero_pad(x, padding, axes=(1,2))
    y = xp.zeros((1, new_height, new_width), dtype=x.dtype)

    for c in range(xc):
        for row in range(new_height):
            for col in range(new_width):
                h = row * stride
                w = col * stride
                x_slice = xpad[c, h:h+kh, w:w+kw]
                y[0, row, col] = xp.max(x_slice)
    return y

def _max_pool2d_gradient_iterative(x, kernel_size, outGrad, stride=1, padding=0):
    assert x.ndim == 3
    assert outGrad.ndim == 3
    assert outGrad.shape[0] == 1

    xc, xh, xw = x.shape
    kc, kh, kw = (1, kernel_size, kernel_size)
    new_height = (xh - kh + 2*padding) // stride + 1
    new_width = (xw - kw + 2*padding) // stride + 1
    assert outGrad.shape == (1, new_height, new_width)

    xpad = utils.zero_pad(x, padding, axes=(1,2))
    dx = xp.zeros(xpad.shape)

    for c in range(xc):
        for row in range(new_height):
            for col in range(new_width):
                h = row * stride
                w = col * stride
                
                dY = outGrad[0, row, col]
                x_slice = xpad[c, h:h+kh, w:w+kw]
                max_coords = xp.unravel_index(xp.argmax(x_slice), x_slice.shape)
                dx[c, h+max_coords[0], w+max_coords[1]] += dY
        
    if padding > 0:
        dx = dx[:, padding:-padding, padding:-padding]
    return dx

    # rows, cols = get_convolution_positions(x.shape, (kc, kh, kw), stride, padding)
    # xpad = utils.zero_pad(x, padding, axes=(1,2))
    # dx = xp.zeros(xpad.shape)
    # x1 = x[:, rows, cols]
    # xmax = xp.argmax(x1, axis=2, keepdims=False).flatten()
    # len_y = xp.arange(new_height*new_width)
    # row_indices = rows[len_y, xmax]
    # col_indices = cols[len_y, xmax]
    # dx[:, row_indices, col_indices] = outGrad
    
    # if padding > 0:
    #     dx = dx[:, padding:-padding, padding:-padding]
    # return dx
