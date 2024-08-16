import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

np.random.seed(0)

# Decide the mean and covariance of the three Gaussian clusters
mean1 = [0, 0]
cov1 = [[50, 0], [0, 1]]

mean2 = [30, 5]
cov2 = [[50, 0], [0, 1]]

mean3 = [30, -20]
cov3 = [[50, 0], [0, 1]]

# Generate data for the three Gaussian clusters
num_samples = 500
data1 = np.random.multivariate_normal(mean1, cov1, num_samples)
data2 = np.random.multivariate_normal(mean2, cov2, num_samples)
data3 = np.random.multivariate_normal(mean3, cov3, num_samples)

# Combine the generated data into one dataset
X = np.vstack((data1, data2, data3))

# Create true labels to evaluate clustering performance
true_labels = np.concatenate([np.zeros(num_samples), np.ones(num_samples), np.full(num_samples, 2)])

# TODO: Fit K-Means with k=3
# Store cluster indices from KMeans in kmeans_labels
k_means = KMeans(n_clusters=3,n_init='auto')
k_means.fit(X)
kmeans_labels = k_means.predict(X)

# TODO: Fit Gaussian Mixture Model with 3 components
# Store cluster indices from GMM in gmm_labels
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
gmm_labels = gmm.predict(X)

# TODO: Plot the data and cluster assignments for K-Means and GMM using subplots in a single figure
# TODO: Save the plot in fig.png

# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=10)
# plt.title('K-Means Clustering')
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")

# plt.subplot(1, 2, 2)
# plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', s=10)
# plt.title('GMM Clustering')
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")

# plt.savefig('fig.png')

# plt.show()

def calculate_error(true_labels, predicted_labels):
    # TODO: Compute the average entropy metric described in the pdf
    # TODO: return the average entropy error
    total = 0 
    num_clusters = 3
    for i in range(num_clusters):
        labels = true_labels[predicted_labels == i]
        unique_elements,count_unique_elements = np.unique(labels,return_counts=True)
        probabilities = count_unique_elements/len(labels)
        total += entropy(probabilities,base=2)
    return total/num_clusters

# Print the average entropy for each clustering algorithm
print("Average Entropy for K-Means:", calculate_error(true_labels, kmeans_labels))
print("Average Entropy for GMM:", calculate_error(true_labels, gmm_labels))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=10)
plt.title('Clustering Using K-Means')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', s=10)
plt.title('Clustering Using GMM')
plt.ylabel("Y-axis")
plt.xlabel("X-axis")


plt.savefig('fig.png')
plt.show()