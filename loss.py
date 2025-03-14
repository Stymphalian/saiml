import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_derivative(X):
    return sigmoid(X) * (1 - sigmoid(X))

def softmax_internal(X):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = X - np.max(X)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)
    # return np.exp(X) / np.sum(np.exp(X))

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
    D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
    return D

def softmax_derivative_iterative(X):
    """Unvectorized computation of the gradient of softmax.

    z: (T, 1) column array of input values.

    Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
    is DjSi - the partial derivative of Si w.r.t. input j.
    """
    S = softmax_internal(X)
    grads = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            if i == j:
                grads[i][j] = S[i][0] * (1 - S[i][0])
            else:
                grads[i][j] = -S[i][0] * S[j][0]
    return grads


def softmax(X):
    return np.array([softmax_internal(x) for x in X])

def softmax_derivative(X):
    return np.array([softmax_derivative_iterative(x) for x in X])
    # return softmax_derivative_vectorized(X)

def cross_entropy_loss(y_true, y_pred):
    # cross entropy loss
    loss = 0
    num_batches = y_true.shape[0]
    for b in range(num_batches):
        for i in range(len(y_true[b])):
            if y_pred[b,i] <= 0.0:
                continue
            loss -= y_true[b,i] * np.log(y_pred[b,i])
    loss /= num_batches
    return loss

def cross_entropy_loss_derivative(y_true, y_pred):
    output = np.zeros(y_pred.shape, dtype=np.float64)
    num_batches = y_true.shape[0]
    for b in range(num_batches):
        for i in range(len(y_true[b])):
            if y_pred[b][i][0] <= 0.0:
                continue
            output[b][i][0] += -(y_true[b][i][0] / y_pred[b][i][0])
    return output

def mean_square_error(y_true, y_pred):
    # return np.mean(7np.power(y_true - y_pred, 2))
    err = np.mean(np.power(y_true - y_pred, 2))
    return err

def mean_square_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)
