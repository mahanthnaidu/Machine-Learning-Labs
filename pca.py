import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def centered(X):
    X_std = X - X.mean(axis=0)
    return X_std

def pca(X, num_components):
    # TODO: Implemnt PCA
    # Compute covariance matrix
    # Calculate the eigenvalues and eigenvectors
    # Sort by eigenvalues and select top-k; here k = num_components
    # Project the data onto the selected eigenvectors
    # Return the projected data, the top-k eigenvalues and eigenvectors
    # cov_matrix = np.cov(X,rowvar=False)
    cov_matrix = (X.T)@(X)/(X.shape[0])
    eigen_values,eigen_vectors = np.linalg.eigh(cov_matrix)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalues = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    top_k_eigenvectors = sorted_eigenvectors[:,0:num_components]
    top_k_eigenvalues = sorted_eigenvalues[0:num_components]
    X_reduced = (np.dot(top_k_eigenvectors.T,X.T)).transpose()
    return X_reduced,top_k_eigenvalues,top_k_eigenvectors

def kernel_pca(X, num_components=2, gamma=15):
    # TODO: Compute a kernel matrix using an RBF kernel
    # Center the kernel matrix as described in the assignment
    # Find the top-k eigenvectors from the centered kernel matrix; here k = num_components
    # Project the data onto the selected eigenvectors
    # Return the projected data, the top-k eigenvalues and eigenvectors
    pairwise_sq_dists = np.sum((X[:, np.newaxis] - X) ** 2, axis=2)
    kernel_matrix = np.exp((pairwise_sq_dists)*(-1)*(gamma))
    n = X.shape[0]
    one_n_matrix = np.ones((n,n))
    one_n_matrix = one_n_matrix/n 
    centered_kernel_matrix = kernel_matrix - (one_n_matrix)@kernel_matrix - (kernel_matrix)@one_n_matrix + (one_n_matrix@kernel_matrix)@one_n_matrix 
   # cov_matrix = np.cov(centered_kernel_matrix,rowvar=False)
    eigen_values,eigen_vectors = np.linalg.eigh(centered_kernel_matrix)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalues = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    top_k_eigenvectors = sorted_eigenvectors[:,0:num_components]
    top_k_eigenvalues = sorted_eigenvalues[0:num_components]
    X_reduced = (np.dot(top_k_eigenvectors.T,centered_kernel_matrix.T)).transpose()
    # print(X_reduced)
    # print(X_reduced.shape)
    return X_reduced,top_k_eigenvalues,top_k_eigenvectors

#Load dataset
X, y = make_moons(n_samples=100, noise=0.05, random_state=1)

#Call PCA and Kernel PCA
X_centered = centered(X)
X_pca,eigenvals_pca,eigenvecs_pca = pca(X_centered, 2)
X_kernel_pca,eigenvals_kpca,eigenvecs_kpca = kernel_pca(X_centered, num_components=2, gamma=15)

# Visualize the principal components (project to PC1, PC2)
plt.scatter(X[:, 0], X[:, 1],c=y)
plt.title('Half Moon dataset')
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('dataset.png')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1],c=y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA')

plt.subplot(1, 2, 2)
plt.scatter(X_kernel_pca[:, 0], X_kernel_pca[:, 1],c=y)
plt.title('Kernel PCA (RBF)')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.savefig('pca_kpca.png')
