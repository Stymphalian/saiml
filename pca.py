
import utils
import numpy as np

def pca(data, num_components=None):
    num_features = data.shape[1]
    if num_components is None:
        num_components = num_features
    if num_components > num_features:
        raise ValueError("num_components cannot be greater than num_features")

    cm = utils.covariance_matrix(data)
    eigen_values, eigen_vectors = np.linalg.eig(cm)

    eigs = [(abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(eigen_values.shape[0])]
    eigs = sorted(eigs, key=lambda x: x[0], reverse=True)
    eigen_values = np.array([x[0] for x in eigs])
    eigen_vectors = np.array([x[1] for x in eigs])
    return eigen_values, eigen_vectors

def pca_transform(X, components):
    X = utils.standardize_dataset(X.copy())
    X = np.dot(X, components.T)
    return X
    # return np.dot(data, components.T)

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.variance_ratio = None
        self.cumulative_variance_ratio = None

    def fit(self, X):
        # Standardize data
        X = utils.standardize_dataset(X.copy())

        # Eigendecomposition of covariance matrix
        eigen_values, eigen_vectors = pca(X)

        self.components = eigen_vectors[:self.n_components,:]
        self.variance_ratio = eigen_values / np.sum(eigen_values)
        self.cumulative_variance_ratio = np.cumsum(self.variance_ratio)
        return self

    def transform(self, X):
        return pca_transform(X, self.components)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.random.seed(0)
    
    # simulated data
    data = []
    e1 = 1.5
    e2 = -2.5
    data = np.array([
        [0*100,0,-0.5], [0*100,1,-0.5], [0*100,2,-0.5], [0*100,3,-0.5], [0*100,4,-0.5], [0*100,5,-0.5],
        [1*100,0,-0.5], [1*100,1,-0.5], [1*100,2,-0.5], [1*100,3,-0.5], [1*100,4,-0.5], [1*100,5,-0.5],
        [2*100,0,-0.5], [2*100,1,-0.5], [2*100,2,-0.5], [2*100,3,-0.5], [2*100,4,-0.5], [2*100,5,-0.5],
        [3*100,0,-0.5], [3*100,1,-0.5], [3*100,2,-0.5], [3*100,3,-0.5], [3*100,4,-0.5], [3*100,5,-0.5],
        [4*100,0,-0.5], [4*100,1,-0.5], [4*100,2,-0.5], [4*100,3,-0.5], [4*100,4,-0.5], [4*100,5,-0.5],
        [5*100,0,-0.5], [5*100,1,-0.5], [5*100,2,-0.5], [5*100,3,-0.5], [5*100,4,-0.5], [5*100,5,-0.5],
        [0*100+e1,0+e2,0.5], [0*100+e1,1+e2,0.5], [0*100+e1,2+e2,0.5], [0*100+e1,3+e2,0.5], [0*100+e1,4+e2,0.5], [0*100+e1,5+e2,0.5],
        [1*100+e1,0+e2,0.5], [1*100+e1,1+e2,0.5], [1*100+e1,2+e2,0.5], [1*100+e1,3+e2,0.5], [1*100+e1,4+e2,0.5], [1*100+e1,5+e2,0.5],
        [2*100+e1,0+e2,0.5], [2*100+e1,1+e2,0.5], [2*100+e1,2+e2,0.5], [2*100+e1,3+e2,0.5], [2*100+e1,4+e2,0.5], [2*100+e1,5+e2,0.5],
        [3*100+e1,0+e2,0.5], [3*100+e1,1+e2,0.5], [3*100+e1,2+e2,0.5], [3*100+e1,3+e2,0.5], [3*100+e1,4+e2,0.5], [3*100+e1,5+e2,0.5],
        [4*100+e1,0+e2,0.5], [4*100+e1,1+e2,0.5], [4*100+e1,2+e2,0.5], [4*100+e1,3+e2,0.5], [4*100+e1,4+e2,0.5], [4*100+e1,5+e2,0.5],
        [5*100+e1,0+e2,0.5], [5*100+e1,1+e2,0.5], [5*100+e1,2+e2,0.5], [5*100+e1,3+e2,0.5], [5*100+e1,4+e2,0.5], [5*100+e1,5+e2,0.5],
    ], dtype=np.float32)
    data_std = utils.standardize_dataset(data)

    # Run the PCA
    pca2 = PCA(n_components=2).fit(data)
    proj2 = pca2.transform(data)

    # Plot the Data
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_std[:, 0], data_std[:, 1], data_std[:, 2], c='red', edgecolor='k', s=40)
    for i, comp in enumerate(pca2.components):
        # comp = comp * var  # scale component by its variance explanation power
        plt.plot(
            [0, comp[0]],
            [0, comp[1]],
            label=f"Component {i}",
            linewidth=5,
            color=f"C{i + 2}",
        )
    ax.set_title('3D Projection of Data')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.show()

    # Plot the 2d Projection from the PCA
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(proj2[:, 0], proj2[:, 1], c='blue', edgecolor='k', s=40)
    plt.title('MAIML PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show(block=True)