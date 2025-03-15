import numpy as np

def numericalGradientCheck(fn, parameters: np.array, predictedGradient, h=1e-7):
    if predictedGradient.shape != parameters.shape:
        raise Exception("Gradients and parameters must be of the same shape")

    numericGradient = np.zeros(parameters.shape, dtype=np.float64)
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

    numerator = np.linalg.norm(numericGradient - predictedGradient)
    denominator = np.linalg.norm(numericGradient) + np.linalg.norm(predictedGradient)
    diff = numerator / denominator
    return numericGradient, diff

def covariance_matrix(data):
    """
    Return the covariance matrix of the data.
    data should be of shape (num_samples, num_features)
    """
    n = data.shape[0]
    means = np.mean(data, axis=0)
    x = data - means
    return np.dot(x.T, x) / n
    # np.cov(data, rowvar=False)
    # def covariance(data, means, i, j):
    #     mean_i = means[i]
    #     mean_j = means[j]
    #     mean_ij = np.mean(data[:,i] * data[:, j])
    #     return mean_ij - mean_i * mean_j
    # n = data.shape[1]
    # cm = np.zeros((n,n))
    # means = np.mean(data, axis=0)
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
        # min_value = np.min(data[:,y])
        # max_value = np.max(data[:,y])
        # data[:,y] = (data[:,y] - min_value) / (max_value - min_value)
        xmean = np.mean(data[:,y])
        xstd = np.std(data[:,y])
        data[:,y] = (data[:,y] - xmean) / xstd
    return data


def onehot_encode(y, size):
    onehot = np.zeros(size)
    onehot[y] = 1
    return onehot.reshape(size, 1)

def onehot_decode(y):
    return np.argmax(y)

def create_batches(data, batch_size):
    n = data.shape[0]
    if n % batch_size != 0:
        raise Exception("Number of samples is not divisible by batch size")
    m = n // batch_size
    batches = np.vsplit(data, m)
    return np.array(batches)


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