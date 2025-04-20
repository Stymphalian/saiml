import math
import functools
from devices import xp
from .. import utils

def get_slicing(x_shape, axes, replacements):
    assert len(axes) == len(replacements)
    axes = [axis if axis >= 0 else len(x_shape) + axis for axis in axes]
    mapping = {axis: replacement for axis, replacement in zip(axes, replacements)}
    slices = []
    for axis in range(len(x_shape)):
        if axis in axes:
            slices.append(mapping[axis])
        else:
            slices.append(None)
    return slices

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

def get_conv2d_height_width(x_shape, kernel_shape, stride, padding, dilate=0):
    assert len(x_shape) >= 2
    assert len(kernel_shape) >= 2

    kh, kw = kernel_shape[-2:]
    xh, xw = x_shape[-2:]
    xh = xh + (xh - 1) * dilate
    xw = xw + (xw - 1) * dilate

    new_height = (xh - kh + 2*padding) // stride + 1
    new_width = (xw - kw + 2*padding) // stride + 1
    return (new_height, new_width)

def get_conv2d_transpose_height_width(y_shape, kernel_shape, stride, padding):
    yc, yh, yw = y_shape
    kc, kh, kw = kernel_shape

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
    return (new_height, new_width)

def validate_conv2d_sequence(seq):
    next_ts = None
    for ts in seq:
        (nc, h,w), (kc, kh, kw), stride, padding = ts
        if next_ts is not None and next_ts != ts[0]:
            raise Exception("Shapes do not match up")
        next_ts = get_conv2d_height_width((nc, h, w), (kc, kh, kw), stride, padding, 0)
        next_ts = (kc, next_ts[0], next_ts[1])
        print(f"{ts} -> {next_ts}")

def validate_conv2d_transpose_sequence(seq):
    next_ts = None
    for ts in seq:
        (nc, h,w), (kc, kh, kw), stride, padding = ts
        if next_ts is not None and next_ts != ts[0]:
            raise Exception("Shapes do not match up")
        next_ts = get_conv2d_transpose_height_width((nc, h, w), (kc, kh, kw), stride, padding)
        next_ts = (kc, next_ts[0], next_ts[1])
        print(f"{ts} -> {next_ts}")

def vectorize_input_for_convolution(x, kernel_shape, stride=1, padding=0):
    assert x.ndim == len(kernel_shape)
    assert x.ndim == 3
    assert kernel_shape[1] == kernel_shape[2]
    rows, cols = get_convolution_positions(x.shape, kernel_shape, stride, padding)
    x_pad = utils.zero_pad(x, padding, axes=(1,2))
    return x_pad[:, rows, cols]


"""

xp.random.seed(1)
nc = 2
x = xp.arange(nc*4*3)
xp.random.shuffle(x)
x = x.reshape(nc,3,4) + 1
k = xp.random.rand(nc,2,2)
dy = xp.round(xp.random.rand(1,2,2))

xc, xh, xw = x.shape
kc, kh, kw = k.shape

rows, cols = get_convolution_positions(x.shape, k.shape)
vx = x[:,rows, cols]
vx = xp.concatenate(vx, axis=0)
flat_x = xp.concatenate(x, axis=0)
max_x = xp.argmax(vx,axis=1)

print("x", x)
print("rows.shape", rows.shape)
print("vx.shape", vx.shape)
print("max_x.shape", max_x.shape)
print("vx", vx)
print("max_x", max_x)
print("flat_x", flat_x)


linear = xp.mod(xp.arange(max_x.size), len(rows))
offset = xp.repeat(xp.arange(nc)*xh, len(rows))
r = rows[linear, max_x] + offset
c = cols[linear, max_x]
print("r", r)
print("c", c)
print("flat_x[r,c]", flat_x[r, c])

linear_nc = xp.arange(nc)
indices = xp.ravel_multi_index((r,c), dims=flat_x.shape)
print("indices", indices)

x_identity = xp.identity(flat_x.size)
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
    assert len(x_shape) >= 2
    assert len(kernel_shape) >= 2
    xh, xw = x_shape[-2:]
    kh, kw = kernel_shape[-2:]
    if padding > 0:
        xh += 2 * padding
        xw += 2 * padding

    horz = xp.array(range(0, xw-kw+1, stride))
    vert = xp.array(range(0, xh-kh+1, stride))
    horz_len = len(horz)
    vert_len = len(vert)

    # First get the x coordinate positions along a single row
    kernel_x = xp.tile(xp.arange(kw), kh)
    kernel_row = xp.tile(kernel_x, horz_len)
    row_incr = xp.repeat(horz, kh*kw)
    row = kernel_row + row_incr

    # Second get the y coordinates along a single column
    kernel_y = xp.repeat(xp.arange(kh), kw)
    kernel_col = xp.tile(kernel_y, vert_len)
    col_incr = xp.repeat(vert, kh*kw)
    col = kernel_col + col_incr
    col = col.reshape(vert_len, kh*kw)

    # Get the columns across all the rows
    # Get the rows across all the columns
    rows_across_cols = xp.tile(row, vert_len).reshape(-1, kh*kw)
    cols_across_rows = xp.tile(col, horz_len).reshape(-1, kh*kw)

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
    flat_vx = xp.concatenate(vx, axis=0)
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
    
def vectorize_kernel(x_shape, kernel, stride=1, pad=0, dilate=0):
    """
    x_shape = (...b, xh, xw)
    kernel = (...ks, kh, kw)
    Output.shape == (..., new_height*new_width, x_shape.size)
    """
    # TODO: How to do this operation purely using numpy?
    assert len(x_shape) >= 2, "Input must be atleast (..., h, w)"
    assert len(kernel.shape) >= 2, "Kernel must be atleast (...ks, kh, kw)"
    assert kernel.shape[-1] == kernel.shape[-2]
    orig_kernel_shape = kernel.shape
    x_shape = x_shape[-2:]
    k_shape = kernel.shape[-2:]
    kernel = xp.reshape(kernel, (-1,) + kernel.shape[-2:])

    xh, xw = x_shape
    kh, kw = k_shape
    ks = kernel.shape[0]
    if dilate > 0:
        xh += (xh - 1) * dilate
        xw += (xw - 1) * dilate
    if pad > 0:
        xh += 2 * pad
        xw += 2 * pad
    
    new_height, new_width = get_conv2d_height_width(x_shape, k_shape, stride, pad, dilate)
    output_size = new_height * new_width
    input_size = xh * xw

    
    rows, cols = get_convolution_positions(x_shape, k_shape, stride, pad)   # (nh*nw, xh*xw)
    # flatten the kernel. flatten spatial dimensions and merge other dims.  # (...ks, kh, kw) == kernel.shape
    kernel2 = xp.reshape(kernel, (1, ks, kh*kw))                            # (1, ks, kh*kw)
    # broadcast to the output_size in order to allow for indexing into M
    kernel2 = xp.broadcast_to(kernel2, (output_size, ks, kh*kw))            # (nh*nw, ks, kh*kw)
    # Need to move the 'ks' to the end. Due to how numpy indexing works
    # any ":" selection in the indexing of M is moved to the end of the shape
    kernel2 = xp.transpose(kernel2, (0,2,1))                                # (nh*nw, kh*kw, ks)
    
    os = xp.arange(0, output_size).reshape((output_size, 1))                # (nh*nw, 1)
    M = xp.zeros((output_size, ks, xh, xw), dtype=xp.float64)               # (nh*nw, ks, xh, xw)
    # assign the kernel into the convolution positions
    M[os, :, rows, cols] = kernel2                                          # (nh*nw, ks, xh, xw)
    M = xp.transpose(M, (1,0,2,3))                                          # (ks, nh*nw, xh, xw)
    M = M.reshape(orig_kernel_shape[:-2] + (output_size, input_size))       # (...ks, nh*nw, xh*xw)
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
    flat_vx = xp.concatenate(vx, axis=0)
    vk = fn(flat_vx, rows, cols)
    return vk


# def full_convolve2D(x, kernel, stride=1):
#     """
#     X.shape == (height, widhth)
#     kernel.shape == (height, width)
#     """
#     kh, kw = kernel.shape
#     kernel = xp.rot90(kernel, 2)
#     x = utils.zero_dilate_2d(x, stride-1)
#     x = xp.pad(x, (kw-1, kh-1) , 'constant', constant_values=0)
#     x = x[xp.newaxis,:,:]
#     kernel = kernel[xp.newaxis,:,:]
#     return convolve2d(x, kernel)

#     # kh, kw = kernel.shape
#     # kernel = xp.rot90(kernel, 2)
#     # X = xp.pad(X, (kw-1, kh-1) , 'constant', constant_values=0)
#     # return convolve2D(X, kernel)

# def convolve3D(X, kernel, stride=1, padding=0):
#     height, width, channels = X.shape
#     kh, kw, kc = kernel.shape
#     assert(kc == channels)

#     new_height = (height - kh + 2*padding) // stride + 1
#     new_width = (width - kw + 2*padding) // stride + 1
#     output = xp.zeros((new_height, new_width), dtype=xp.float64)
#     for h in range(height-kh+1):
#         for w in range(width-kw+1):
#             x_slice = X[h:h+kh, w:w+kw, :]
#             v = xp.sum(x_slice * kernel)
#             output[h,w] = v
#     return output

# def full_convolve3D(X, kernel):
#     kh, kw, kc = kernel.shape
#     kernel = xp.rot90(kernel, 2)
#     X = xp.pad(X, [(kw-1, kh-1), (kw-1, kh-1), (0,0)] , 'constant', constant_values=0)
#     return convolve3D(X, kernel)
