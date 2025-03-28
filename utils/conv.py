import math
import numpy as np
import utils

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

def _convolve2d_iterative(x, kernel, stride=1, padding=0, dilate=0):
    assert x.ndim == kernel.ndim
    assert x.ndim == 3
    spatial_axes = (1,2)
    assert kernel.shape[spatial_axes[0]] == kernel.shape[spatial_axes[1]]
    
    new_height, new_width = get_new_height_width(
        x.shape, kernel.shape, stride, padding, dilate)
    x = utils.zero_dilate(x, dilate, axes=spatial_axes)
    x = utils.zero_pad(x, padding, axes=spatial_axes)
    kc, kh, kw = kernel.shape
    output = np.zeros((1, new_height, new_width), dtype=x.dtype)
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
            v = np.sum(x_slice * kernel)
            output[0,row,col] += v
    return output

def _convolve2d_gradient_iterative(x, kernel, outGrad, stride=1, padding=0, dilate=0):
    assert x.ndim == kernel.ndim
    assert x.ndim == 3
    spatial_axes = (1,2)
    assert kernel.shape[spatial_axes[0]] == kernel.shape[spatial_axes[1]]
    new_height, new_width = get_new_height_width(
        x.shape, kernel.shape, stride, padding, dilate)
    
    x = utils.zero_dilate(x, dilate, axes=spatial_axes)
    x = utils.zero_pad(x, padding, axes=spatial_axes)
    kc, kh, kw = kernel.shape
    xc, xh, xw = x.shape
    
    dx = np.zeros(x.shape)
    dk = np.zeros(kernel.shape)

    for row in range(new_height):
        for col in range(new_width):
            h = row * stride
            w = col * stride
            x_slice = x[:, h:h+kh, w:w+kw]
            dY = outGrad[0, row, col]

            if h + kh > xh or w + kw > xw:
                raise Exception("Kernel is out of bounds")
                continue
            dx[:, h:h+kh, w:w+kw] += np.multiply(kernel, dY)
            dk += np.multiply(x_slice, dY)
        
    if padding > 0:
        dx = dx[:, padding:-padding, padding:-padding]
    if dilate > 0:
        dx = utils.zero_undilate(dx, dilate, axes=spatial_axes)
    return (dx, dk)

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
    z = np.matmul(x_prime, flat_kernel)
    z = np.sum(z, axis=0)
    z = np.reshape(z, (1, new_height, new_width))
    return z

def _convolve2d_gradient_vectorized(x, kernel, outGrad, stride=1, padding=0, dilate=0):
    assert outGrad.ndim == 3
    assert kernel.ndim == 3
    assert x.ndim == 3
    assert x.shape[0] == kernel.shape[0]
    assert outGrad.shape[0] == 1

    xpad = utils.zero_pad(x, padding, axes=(1,2))

    vkernel = vectorize_kernel(x.shape, kernel, stride, padding)
    vkernel = np.transpose(vkernel, (0,2,1))
    outGrad = outGrad.flatten().reshape(1,-1,1)
    dx = np.matmul(vkernel, outGrad).reshape(xpad.shape)
    if padding > 0:
        dx = dx[:, padding:-padding, padding:-padding]


    rows, cols = get_convolution_positions(x.shape, kernel.shape, stride, padding)
    x2 = xpad[:, rows, cols]
    x2 = np.transpose(x2, (0,2,1))
    dk = np.matmul(x2, outGrad).reshape(kernel.shape)

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

    x = np.zeros((kc, new_height, new_width), dtype=y.dtype)

    for k in range(kc):
        for row in range(kh):
            for col in range(kw):
                h = row * stride_row
                w = col * stride_col
                x[k, h:h+yh, w:w+yw] += y[0] * kernel[k,row,col]

    if outer_padding > 0:
        pad = outer_padding
        x = x[:, pad:-pad, pad:-pad]

    x = np.sum(x, axis=0)
    x = x[np.newaxis, :,:]
    return x

def _convolve2d_transpose_gradient_iterative(y, kernel, outGrad, stride=1, padding=0, outer_padding=0):
    assert y.ndim == kernel.ndim
    assert kernel.ndim == 3
    spatial_axes = (1,2)
    assert kernel.shape[spatial_axes[0]] == kernel.shape[spatial_axes[1]]

    yc, yh, yw = y.shape
    kc, kh, kw = kernel.shape

    xh = (yh-1)*stride + kh - 2*padding
    xw = (yw-1)*stride + kw - 2*padding
    stride_row = math.ceil((xh - yh) / (kh -1))
    stride_col = math.ceil((xw - yw) / (kw -1))
    if (stride_row < 0) or (stride_col < 0):
        return False, None
    new_height = yh + (kh - 1)*stride_row
    new_width = yw + (kw - 1)*stride_col
    if outer_padding > 0:
        outGrad = utils.zero_pad(outGrad, outer_padding, axes=spatial_axes)
    assert outGrad.shape == (1, new_height, new_width), "{} != {}".format(outGrad.shape, (kc, new_height, new_width))
    
    dy = np.zeros(y.shape)
    dk = np.zeros(kernel.shape)

    for k in range(kc):
        for row in range(kh):
            for col in range(kw):
                h = row * stride_row
                w = col * stride_col
                out_slice = outGrad[0, h:h+yh, w:w+yw]
                assert (out_slice.shape == y.shape[1:])
                dy += kernel[k,row,col] * out_slice
                dk[k,row,col] += np.sum(y * out_slice)

                # h = row * stride_row
                # w = col * stride_col
                # out_slice = outGrad[:, h:h+yh, w:w+yw]
                # assert (out_slice.shape[1:] == y.shape)
                # dy += np.sum(kernel[:,row,col] * out_slice, axis=0)
                # dk[:,row, col] += np.sum(y * out_slice)

    return (dy, dk)

def convolve2d(x, kernel, stride=1, padding=0, dilate=0):
    # return _convolve2d_iterative(x, kernel, stride=stride, padding=padding, dilate=dilate)
    return _convolve2d_vectorized(x, kernel, stride=stride, padding=padding)

def convolve2d_gradient(x, kernel, outGrad, stride=1, padding=0, dilate=0):
    # return _convolve2d_gradient_iterative(
    #     x, kernel, outGrad, stride=stride, padding=padding, dilate=dilate)
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
    
def full_convolve2D(x, kernel, stride=1):
    """
    X.shape == (height, widhth)
    kernel.shape == (height, width)
    """
    kh, kw = kernel.shape
    kernel = np.rot90(kernel, 2)
    x = utils.zero_dilate_2d(x, stride-1)
    x = np.pad(x, (kw-1, kh-1) , 'constant', constant_values=0)
    x = x[np.newaxis,:,:]
    kernel = kernel[np.newaxis,:,:]
    return convolve2d(x, kernel)

    # kh, kw = kernel.shape
    # kernel = np.rot90(kernel, 2)
    # X = np.pad(X, (kw-1, kh-1) , 'constant', constant_values=0)
    # return convolve2D(X, kernel)

def convolve3D(X, kernel, stride=1, padding=0):
    height, width, channels = X.shape
    kh, kw, kc = kernel.shape
    assert(kc == channels)

    new_height = (height - kh + 2*padding) // stride + 1
    new_width = (width - kw + 2*padding) // stride + 1
    output = np.zeros((new_height, new_width), dtype=np.float64)
    for h in range(height-kh+1):
        for w in range(width-kw+1):
            x_slice = X[h:h+kh, w:w+kw, :]
            v = np.sum(x_slice * kernel)
            output[h,w] = v
    return output

def full_convolve3D(X, kernel):
    kh, kw, kc = kernel.shape
    kernel = np.rot90(kernel, 2)
    X = np.pad(X, [(kw-1, kh-1), (kw-1, kh-1), (0,0)] , 'constant', constant_values=0)
    return convolve3D(X, kernel)

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (n_C_prev, f, f)
    W -- Weight parameters contained in a window - matrix of shape (n_C_prev, f, f)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """

    ### START CODE HERE ### (≈ 2 lines of code)
    # Element-wise product between a_slice and W. Do not add the bias yet.
    s = np.multiply(a_slice_prev, W)
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + float(b)
    ### END CODE HERE ###

    return Z

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_C_prev, n_H_prev, n_W_prev)
    W -- Weights, numpy array of shape (n_C, n_C_prev, f, f)
    b -- Biases, numpy array of shape (n_C, 1, 1, 1)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_C, n_H, n_W)
    cache -- cache of values needed for the conv_backward() function
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from A_prev's shape (≈1 line)  
    (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape (≈1 line)
    (n_C, n_C_prev, f, f, ) = W.shape
    
    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    n_H = int((n_H_prev - f + 2 * pad) / stride + 1)
    n_W = int((n_W_prev - f + 2 * pad) / stride + 1)
    
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_C, n_H, n_W))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = utils.zero_pad2(A_prev, pad)
    
    for i in range(m):                               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i, :, :, :]          # Select ith training example's padded activation
        for h in range(n_H):                           # loop over vertical axis of the output volume
            for w in range(n_W):                       # loop over horizontal axis of the output volume
                for c in range(n_C):                   # loop over channels (= #filters) of the output volume
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f
                    
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[:, vert_start: vert_end, horiz_start: horiz_end]
                    
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                    Z[i, c, h, w] = conv_single_step(a_slice_prev, W[c], b[c])
                                        
    ### END CODE HERE ###
    
    # Making sure your output shape is correct
    assert(Z.shape == (m, n_C, n_H, n_W))
    
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache

def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), 
        numpy array of shape (m, n_C, n_H, n_W)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_C_prev, n_H_prev, n_W_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (n_C, n_C_prev, f, f)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (n_C, 1, 1, 1)
    """
    
    ### START CODE HERE ###
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache
    
    # Retrieve dimensions from A_prev's shape
    (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (n_C, n_C_prev, f, f) = W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Retrieve dimensions from dZ's shape
    (m, n_C, n_H, n_W) = dZ.shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_C_prev, n_H_prev, n_W_prev))                           
    dW = np.zeros((n_C, n_C_prev, f, f))
    db = np.zeros((n_C, 1, 1, 1))

    # Pad A_prev and dA_prev
    A_prev_pad = utils.zero_pad2(A_prev, pad)
    dA_prev_pad = utils.zero_pad2(dA_prev, pad)
    
    for i in range(m):                       # loop over the training examples
        
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i, :, :, :]
        da_prev_pad = dA_prev_pad[i, :, :, :]
        
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f
                    
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end] += W[c] * dZ[i, c, h, w]
                    dW[c] += a_slice * dZ[i, c, h, w]
                    db[c] += dZ[i, c, h, w]
                    
        # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        if pad > 0:
            dA_prev[i, :, :, :] = da_prev_pad[:, pad:-pad, pad:-pad]
        else:
            dA_prev[i, :, :, :] = da_prev_pad
    ### END CODE HERE ###
    
    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_C_prev, n_H_prev, n_W_prev))
    
    return dA_prev, dW, db