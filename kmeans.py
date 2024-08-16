import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters for the clusters
num_samples = 300  # Total number of data points
num_clusters = 3   # Number of clusters
cluster_means = np.array([[-5, -5], [5, 5], [0, 10]])
cluster_stddevs = np.array([1.0, 1.0, 1.0])

# Generate random data for each cluster
data = []
for i in range(num_clusters):
    cluster_data = np.random.normal(cluster_means[i], cluster_stddevs[i], (num_samples // num_clusters, 2))
    data.append(cluster_data)

print(cluster_data.shape)
data = np.vstack(data)
np.random.shuffle(data)

# Plot the generated dataset
plt.scatter(data[:, 0], data[:, 1], s=10)
plt.title("Generated Dataset for K-Means Clustering")
plt.show()

# Save the dataset to a CSV file
np.savetxt('kmeans_dataset.csv', data, delimiter=',')

def load_dataset(filename):
    data = pd.read_csv(filename, header=None)
    X = data.values
    return X

def compute_clustering_error(data, centroids, labels):
    '''
    compute the clustering error 
    inputs:
        data - numpy array of data points having shape (300, 2)
        centroids - array of shape (3, 2) containing cluster centers
        labels - array of shape (300, 1) containing assignment of point to index of closest cluster
    '''
    # TODO: Compute the sum of squared distances from each data point to its closest centroid
    # TODO: return clustering_error that contains the sum of these squared distances
    clustering_error = 0.0
    for i in range(num_samples):
        clustering_error += (np.linalg.norm(data[i] - centroids[labels[i]]))**2
    return clustering_error

def initialize_centroids(data, k, default=True):
    '''
    initialize the centroids for K-means and K-means++
    inputs:
        data - numpy array of data points having shape (300, 2)
        k - number of clusters
        default - if True use random initialization else use kmeans++ initialization
    '''
    if default:
      # Randomly initialize cluster centroids
        initial_centroids = data[np.random.choice(data.shape[0], k, replace=False)]
        return initial_centroids
    else:
      # kmeans++ initialization
      # TODO: initialize the centroids using the k-means++ algorithm specified in the pdf 
        centroids = []
        num_selected  = 0
        data1 = data.copy()
        random_ind = np.random.randint(num_samples)
        centroids.append(data1[random_ind])
        data1 = np.delete(data1,random_ind,axis=0)
        num_selected = num_selected + 1 
        while num_selected < k:
            probabilities = [0.0 for _ in range(data1.shape[0])]
            probabilities = np.array(probabilities)
            for i in range(data1.shape[0]):
                min_ind = np.argmin(np.linalg.norm((data1[i] - np.array(centroids))**2,axis = 1),axis = 0)
                probabilities[i] = (np.linalg.norm(data1[i] - np.array(centroids[min_ind])))**2
            probabilities = probabilities/np.sum(probabilities)
            random_ind = np.random.choice(len(data1), p = probabilities)
            centroids.append(data1[random_ind])
            data1 = np.delete(data1,random_ind,axis=0)
            num_selected = num_selected + 1
        centroids = np.array(centroids)
        return centroids

def k_means(X, k, max_iterations=200, default=True):
    centroids = initialize_centroids(X, k, default=default)
    # TODO: Implement both the assignment and update steps
    # TODO: If the assignment has not changed across consecutive steps, terminate 
    # TODO: Run upto a max of max_iterations
    labels = [0 for _ in range(num_samples)]
    for i in range(max_iterations):
        new_centroids = np.zeros((k,X.shape[1]))
        num_points = [0 for _ in range(num_clusters)]
        for j in range(num_samples):
            max_index = np.argmin(np.linalg.norm((X[j] - centroids),axis=1),axis=0)
            num_points[max_index] = num_points[max_index] + 1 
            labels[j] = max_index
            new_centroids[max_index] = new_centroids[max_index] + X[j]
        num_points = np.array(num_points)
        for j in range(k):
            new_centroids[j] = new_centroids[j]/num_points[j]
        if(np.array_equal(new_centroids,centroids)):
            break
        else:
            centroids = new_centroids

    # TODO: labels stores the nearest cluster index for each datapoint
    # TODO: centroids stores the cluster centroids
    labels = np.array(labels)
    return labels,centroids

def plot_clusters(X, labels, centroids, k):
    plt.figure(figsize=(8, 6))

    for i in range(k):
        cluster_data = X[labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1,], label=f'Cluster {i+1}')

    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='x', label='Centroids')
    plt.title("K-means Clustering")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    from statistics import mean

    filename = "kmeans_dataset.csv"
    k = 3

    X = load_dataset(filename)
    error_list = []

    runs=50
    for i in range(runs):
      np.random.seed(i)
      labels, centroids = k_means(X, k, default=True)
      #uncomment the line below to generate a plot
      # plot_clusters(X, labels, centroids, k)
      kmeans_score = compute_clustering_error(X, centroids, labels)

      labels, centroids = k_means(X, k, default=False)
      #uncomment the line below to generate a plot
      # plot_clusters(X, labels, centroids, k)
      kmeans_plus_score = compute_clustering_error(X, centroids, labels)
      error_list.append(kmeans_score-kmeans_plus_score)

    print("Mean error across " + str(runs) + " runs: " + str(mean(error_list)))

