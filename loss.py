from devices import xp

def sigmoid(X):
    return 1 / (1 + xp.exp(-X))

def sigmoid_derivative(X):
    return sigmoid(X) * (1 - sigmoid(X))

def relu(x):
    return (x + xp.abs(x)) * 0.5

def relu_derivative(x):
    return xp.zeros(x.shape) + (x > 0)

def softmax_internal(X):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = X - xp.max(X)
    exps = xp.exp(shiftx)
    return exps / xp.sum(exps)
    # return xp.exp(X) / xp.sum(xp.exp(X))

def softmax_derivative_vectorized(z):
    """Computes the gradient of the softmax function.

    z: (T, 1) array of input values where the gradient is computed. T is the
       number of output classes.

    Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
    is DjSi - the partial derivative of Si w.r.t. input j.
    """
    Sz = softmax_internal(z)
    # -SjSi can be computed using an outer product between Sz and itself. Then
    # we add back Si for the i=j cases by adding a diagonal matrix with the
    # values of Si on its diagonal.
    D = -xp.outer(Sz, Sz) + xp.diag(Sz.flatten())
    return D

def softmax_derivative_iterative(X):
    """Unvectorized computation of the gradient of softmax.

    z: (T, 1) column array of input values.

    Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
    is DjSi - the partial derivative of Si w.r.t. input j.
    """
    S = softmax_internal(X)
    grads = xp.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            if i == j:
                grads[i][j] = S[i][0] * (1 - S[i][0])
            else:
                grads[i][j] = -S[i][0] * S[j][0]
    return grads


def softmax(X):
    return xp.array([softmax_internal(x) for x in X])

def softmax_derivative(X):
    return xp.array([softmax_derivative_iterative(x) for x in X])
    # return softmax_derivative_vectorized(X)

def cross_entropy_loss(y_true, y_pred):
    # cross entropy loss
    loss = 0
    num_batches = y_true.shape[0]
    for b in range(num_batches):
        for i in range(len(y_true[b])):
            if y_pred[b,i] <= 0.0:
                continue
            loss -= y_true[b,i] * xp.log(y_pred[b,i])
    loss /= num_batches
    return loss

def cross_entropy_loss_derivative(y_true, y_pred):
    output = xp.zeros(y_pred.shape, dtype=xp.float64)
    num_batches = y_true.shape[0]
    for b in range(num_batches):
        for i in range(len(y_true[b])):
            if y_pred[b][i][0] <= 0.0:
                continue
            output[b][i][0] += -(y_true[b][i][0] / y_pred[b][i][0])
    return output

def mean_square_error(y_true, y_pred):
    # return xp.mean(7xp.power(y_true - y_pred, 2))
    err = xp.mean(xp.power(y_true - y_pred, 2))
    return err

def mean_square_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / xp.size(y_true)
