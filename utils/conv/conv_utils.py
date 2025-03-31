import math
import functools
import numpy as np
from .. import utils

def check_if_shapes_work(x_shape, ks, stride, padding):
    xh, xw = x_shape
    kh, kw = ks, ks

    yh = ((xh - kh + 2*padding) // stride) + 1
    yw = ((xw - kw + 2*padding) // stride) + 1

    xh_ = (yh-1)*stride + kh - 2*padding
    xw_ = (yw-1)*stride + kw - 2*padding
    stride_row = math.ceil((xh_ - yh) / (kh -1))
    stride_col = math.ceil((xw_ - yw) / (kw -1))
    if (stride_row < 0) or (stride_col < 0):
        return False, None
    new_height = yh + (kh - 1)*stride_row
    new_width = yw + (kw - 1)*stride_col

    valid = (new_height == xh and  new_width == xw)
    return (valid, (yh, yw))

def get_new_height_width(x_shape, kernel_shape, stride, padding, dilate):
    if len(x_shape) == 3:
        kc, kh, kw = kernel_shape
        xc, xh, xw = x_shape
    else:
        kh, kw = kernel_shape
        xh, xw = x_shape
    xh = xh + (xh - 1) * dilate
    xw = xw + (xw - 1) * dilate

    new_height = (xh - kh + 2*padding) // stride + 1
    new_width = (xw - kw + 2*padding) // stride + 1
    return (new_height, new_width)

def vectorize_input_for_convolution(x, kernel_shape, stride=1, padding=0):
    assert x.ndim == len(kernel_shape)
    assert x.ndim == 3
    assert kernel_shape[1] == kernel_shape[2]
    rows, cols = get_convolution_positions(x.shape, kernel_shape, stride, padding)
    x_pad = utils.zero_pad(x, padding, axes=(1,2))
    return x_pad[:, rows, cols]


"""

np.random.seed(1)
nc = 2
x = np.arange(nc*4*3)
np.random.shuffle(x)
x = x.reshape(nc,3,4) + 1
k = np.random.rand(nc,2,2)
dy = np.round(np.random.rand(1,2,2))

xc, xh, xw = x.shape
kc, kh, kw = k.shape

rows, cols = get_convolution_positions(x.shape, k.shape)
vx = x[:,rows, cols]
vx = np.concatenate(vx, axis=0)
flat_x = np.concatenate(x, axis=0)
max_x = np.argmax(vx,axis=1)

print("x", x)
print("rows.shape", rows.shape)
print("vx.shape", vx.shape)
print("max_x.shape", max_x.shape)
print("vx", vx)
print("max_x", max_x)
print("flat_x", flat_x)


linear = np.mod(np.arange(max_x.size), len(rows))
offset = np.repeat(np.arange(nc)*xh, len(rows))
r = rows[linear, max_x] + offset
c = cols[linear, max_x]
print("r", r)
print("c", c)
print("flat_x[r,c]", flat_x[r, c])

linear_nc = np.arange(nc)
indices = np.ravel_multi_index((r,c), dims=flat_x.shape)
print("indices", indices)

x_identity = np.identity(flat_x.size)
vk = x_identity[indices]
print("vk.shape", vk.shape)
print("vk", vk)
"""


@functools.lru_cache()
def get_convolution_positions(x_shape, kernel_shape, stride=1, padding=0):
    """
    Returns the rows, and cols positions of the convolutions. Working from 
    the top-left corner to the bottom right corner.
    rows.shape = (#convolutions, kernel_height*kernel_width)
    cols.shape = (#convolutions, kernel_height*kernel_width)
    """
    xc, xh, xw = x_shape
    kc, kh, kw = kernel_shape
    if padding > 0:
        xh += 2 * padding
        xw += 2 * padding

    horz = np.array(range(0, xw-kw+1, stride))
    vert = np.array(range(0, xh-kh+1, stride))
    horz_len = len(horz)
    vert_len = len(vert)

    # First get the x coordinate positions along a single row
    kernel_x = np.tile(np.arange(kw), kh)
    kernel_row = np.tile(kernel_x, horz_len)
    row_incr = np.repeat(horz, kh*kw)
    row = kernel_row + row_incr

    # Second get the y coordinates along a single column
    kernel_y = np.repeat(np.arange(kh), kw)
    kernel_col = np.tile(kernel_y, vert_len)
    col_incr = np.repeat(vert, kh*kw)
    col = kernel_col + col_incr
    col = col.reshape(vert_len, kh*kw)

    # Get the columns across all the rows
    # Get the rows across all the columns
    rows_across_cols = np.tile(row, vert_len).reshape(-1, kh*kw)
    cols_across_rows = np.tile(col, horz_len).reshape(-1, kh*kw)

    return cols_across_rows, rows_across_cols


def vectorize_kernel_maxpool(x, kernel_shape, stride=1):
    assert x.ndim == 3
    assert kernel_shape[1] == kernel_shape[2]
    assert x.shape[0] == kernel_shape[0]
    xc, xh, xw = x.shape
    kc, kh, kw = kernel_shape

    rows, cols = get_convolution_positions(x.shape, kernel_shape, stride)
    assert len(rows) == len(cols)
    
    vx = x[:,rows, cols]
    flat_vx = np.concatenate(vx, axis=0)
    max_x = np.argmax(flat_vx, axis=1)
    
    linear = np.mod(np.arange(max_x.size), len(rows))
    offset = np.repeat(np.arange(xc)*xh, len(rows))
    r = rows[linear, max_x] + offset
    c = cols[linear, max_x]

    flat_x = np.concatenate(x, axis=0)
    indices = np.ravel_multi_index((r,c), dims=flat_x.shape)
    x_identity = np.identity(flat_x.size)
    vk = x_identity[indices]
    return vk
    
def vectorize_kernel(x_shape, kernel, stride=1, padding=0, dilate=0):
    """
    Output.shape == (new_height*new_width, x_shape.size)
    """
    # TODO: How to do this operation purely using numpy?
    assert len(x_shape) == 3
    assert len(kernel.shape) == 3
    assert kernel.shape[1] == kernel.shape[2]
    assert x_shape[0] == kernel.shape[0]

    xc, xh, xw = x_shape
    if dilate > 0:
        xh += (xh - 1) * dilate
        xw += (xw - 1) * dilate
    if padding > 0:
        xh += 2 * padding
        xw += 2 * padding
    kc, kh, kw = kernel.shape

    new_height, new_width = get_new_height_width(
        x_shape, kernel.shape, stride, padding, dilate)
    output_size = new_height * new_width
    input_size = xh * xw
    M = np.zeros((kc, output_size, input_size), dtype=np.float64)
    scratch = np.zeros((xc, xh, xw), dtype=np.float64)

    kernel_index = 0
    for row in range(new_height):
        for col in range(new_width):
            h = row * stride
            w = col * stride

            scratch[:, h:h+kh, w:w+kw] = kernel[:]
            M[:, kernel_index] = scratch.reshape(kc, -1)
            scratch[:, h:h+kh, w:w+kw] = 0
            kernel_index += 1
    return M

def vectorize_kernel_with_fn(x, kernel_shape, fn, stride=1):
    assert x.ndim == 3
    assert kernel_shape[1] == kernel_shape[2]
    assert x.shape[0] == kernel_shape[0]
    xc, xh, xw = x.shape
    kc, kh, kw = kernel_shape

    rows, cols = get_convolution_positions(x.shape, kernel_shape, stride)
    assert len(rows) == len(cols)
    
    vx = x[:,rows, cols]
    flat_vx = np.concatenate(vx, axis=0)
    vk = fn(flat_vx, rows, cols)
    return vk


# def full_convolve2D(x, kernel, stride=1):
#     """
#     X.shape == (height, widhth)
#     kernel.shape == (height, width)
#     """
#     kh, kw = kernel.shape
#     kernel = np.rot90(kernel, 2)
#     x = utils.zero_dilate_2d(x, stride-1)
#     x = np.pad(x, (kw-1, kh-1) , 'constant', constant_values=0)
#     x = x[np.newaxis,:,:]
#     kernel = kernel[np.newaxis,:,:]
#     return convolve2d(x, kernel)

#     # kh, kw = kernel.shape
#     # kernel = np.rot90(kernel, 2)
#     # X = np.pad(X, (kw-1, kh-1) , 'constant', constant_values=0)
#     # return convolve2D(X, kernel)

# def convolve3D(X, kernel, stride=1, padding=0):
#     height, width, channels = X.shape
#     kh, kw, kc = kernel.shape
#     assert(kc == channels)

#     new_height = (height - kh + 2*padding) // stride + 1
#     new_width = (width - kw + 2*padding) // stride + 1
#     output = np.zeros((new_height, new_width), dtype=np.float64)
#     for h in range(height-kh+1):
#         for w in range(width-kw+1):
#             x_slice = X[h:h+kh, w:w+kw, :]
#             v = np.sum(x_slice * kernel)
#             output[h,w] = v
#     return output

# def full_convolve3D(X, kernel):
#     kh, kw, kc = kernel.shape
#     kernel = np.rot90(kernel, 2)
#     X = np.pad(X, [(kw-1, kh-1), (kw-1, kh-1), (0,0)] , 'constant', constant_values=0)
#     return convolve3D(X, kernel)
