import numpy as np
import cupy as cp
import numpy
from devices import xp

def numericalGradientCheck(fn, parameters: xp.array, predictedGradient, h=1e-7):
    if predictedGradient.shape != parameters.shape:
        raise Exception("Gradients and parameters must be of the same shape")

    numericGradient = xp.zeros(parameters.shape, dtype=xp.float64)
    for i in range(len(parameters)):
        if i % 1000 == 0:
            print("Checking gradient for parameters {0}/{1}".format(i, len(parameters)))
        oldX = parameters[i][0]
        parameters[i][0] = oldX + h
        x1 = fn(parameters)
        parameters[i][0] = oldX - h
        x2 = fn(parameters)
        parameters[i][0] = oldX
        numericGradient[i][0] = (x1 - x2) / (2*h)

    numerator = xp.linalg.norm(numericGradient - predictedGradient)
    # denominator = xp.linalg.norm(numericGradient + predictedGradient)
    # denominator = xp.linalg.norm(numericGradient) + xp.linalg.norm(predictedGradient)
    # diff = numerator / denominator
    diff = numerator
    return numericGradient, diff

def covariance_matrix(data):
    """
    Return the covariance matrix of the data.
    data should be of shape (num_samples, num_features)
    """
    n = data.shape[0]
    means = xp.mean(data, axis=0)
    x = data - means
    return xp.dot(x.T, x) / n
    # xp.cov(data, rowvar=False)
    # def covariance(data, means, i, j):
    #     mean_i = means[i]
    #     mean_j = means[j]
    #     mean_ij = xp.mean(data[:,i] * data[:, j])
    #     return mean_ij - mean_i * mean_j
    # n = data.shape[1]
    # cm = xp.zeros((n,n))
    # means = xp.mean(data, axis=0)
    # for i in range(n):
    #     for j in range(i, n):
    #         val = covariance(data, means, i, j)            
    #         cm[i][j] = val
    #         cm[j][i] = val
    # return cm


def standardize_dataset(data):    
    """
    Modify the data inplace with mean 0 and standard deviation 1.
    Data: Shape should be (num_samples, num_features)
    """
    # return Standardize_data(data)
    for y in range(data.shape[1]):
        # min_value = xp.min(data[:,y])
        # max_value = xp.max(data[:,y])
        # data[:,y] = (data[:,y] - min_value) / (max_value - min_value)
        xmean = xp.mean(data[:,y])
        xstd = xp.std(data[:,y])
        data[:,y] = (data[:,y] - xmean) / xstd
    return data


def onehot_encode(y, size):
    return xp.identity(size)[y].reshape(size, 1)
    # onehot = xp.zeros(size)
    # onehot[y] = 1
    # return onehot.reshape(size, 1)

def onehot_decode(y):
    return xp.argmax(y)

def create_batches(data, batch_size):
    n = data.shape[0]
    if n % batch_size != 0:
        raise Exception("Number of samples is not divisible by batch size")
    m = n // batch_size
    batches = xp.vsplit(data, m)
    return xp.array(batches)


def zero_dilate(x, space, axes=None):
    """
    For example if X is 2d array of shape (height, width)
    Add zeros between each pixel of the 2d array.
    for example:
    123           10203
    456  becomes  00000
    789           40506
                  00000
                  70809
    """
    # TODO: Change this to pure device.xp code
    if space == 0:
        return x
    
    if isinstance(x, cp.ndarray):
        x = cp.asnumpy(x)

    if axes is None:
        axes = range(x.ndim)
    if x.ndim < len(axes):
        raise Exception("Dimension of x must be at least {0}".format(len(axes)))
    a = x
    for s in range(space):
        for axis in axes:
            shape_len = a.shape[axis]
            a = numpy.insert(a, range(1, shape_len, s+1), 0, axis=axis)
    return xp.array(a)

def zero_undilate(x, space, axes=None):
    """
    If X is 2d array of shape (height, width)
    Remove zeros between each pixel of the 2d array.
    for example:
    10203           123           
    00000  becomes  456  
    40506           789           
    00000                          
    70809               
    """
    if space == 0:
        return x
    if axes is None:
        axes = range(x.ndim)
    if x.ndim < len(axes):
        raise Exception("Dimension of x must be at least {0}".format(len(axes)))
    a = x
    for axis in axes:
        shape_len = a.shape[axis]
        axes_to_remove = range(1, shape_len, space+1)
        axes_to_remove = [range(x,x+space) for x in axes_to_remove]
        a = xp.delete(a, axes_to_remove, axis=axis)
    return a

def zero_dilate_2d(x, space):
    return zero_dilate(x, space, axes=(0,1))
    # if space == 0:
    #     return x
    # a = x
    # for s in range(space):
    #     height, width = a.shape
    #     a = xp.insert(a, range(1, height, s+1), 0, axis=0)
    #     a = xp.insert(a, range(1, width, s+1), 0, axis=1)
    # return a    
    # height, width = X.shape
    # new_height = height + (height-1)*space
    # new_width = width + (width-1)*space
    # Y = xp.zeros((new_height, new_width))
    # for row in range(height):
    #     new_row = row + row*space
    #     for col in range(width):
    #         new_col = col + col*space
    #         Y[new_row, new_col] = X[row, col]
    # return Y

def zero_undilate_2d(x, space):
    return zero_undilate(x, space, axes=(0,1))
    # if space == 0:
    #     return x
    # a = x
    # height, width = a.shape
    # horz = range(1, width, space+1)
    # vert = range(1, height, space+1)
    # horz = [range(x,x+space) for x in horz]
    # vert = [range(x,x+space) for x in vert]
    # a = xp.delete(a, vert, axis=0)
    # a = xp.delete(a, horz, axis=1)
    # return a
    # if space == 0:
    #     return X
    # new_height, new_width = X.shape
    # height = (new_height + space) // (1 + space)
    # width = (new_width + space) // (1 + space)
    # Y = xp.zeros((height, width))
    # for row in range(height):
    #     new_row = row + row*space
    #     for col in range(width):
    #         new_col = col + col*space
    #         Y[row, col] = X[new_row, new_col]
    # return Y

def zero_pad(x, pad, axes=None):
    if axes is None:
        axes = range(x.ndim)
    if pad == 0:
        return x

    padding = (pad, pad)
    pads = [(0,0)] * x.ndim
    for axis in axes:
        pads[axis] = padding

    x_pad = xp.pad(x, pads, mode='constant', constant_values=0)
    return x_pad

def zero_pad2(X, pad):
    """
    Pad with zeros all images of the dataset X. 
    The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_C, n_H, n_W) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_C, n_H + 2*pad, n_W + 2*pad)
    """
    return zero_pad(X, pad, axes=(2,3))
    # if pad == 0:
    #     return X
    
    # ### START CODE HERE ### (≈ 1 line)
    # X_pad = xp.pad(X, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant', constant_values = (0,0))
    # ### END CODE HERE ###
    
    # return X_pad


def train(model, x_train, y_train, number_epochs=1, learning_rate=0.1, batch_size=50):
    x_train_batches = create_batches(x_train, batch_size)
    y_train_batches = create_batches(y_train, batch_size)
    for epoch in range(number_epochs):
        error = 0
        for x, y in zip(x_train_batches, y_train_batches):
            pred = model.forward(x)
            loss, grad = model.loss(pred, y)
            error += loss
            model.backward(grad)

        error /= len(x_train)
        print(f"Epoch {epoch+1}/{number_epochs} - Error: {error}")