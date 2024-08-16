import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from copy import deepcopy

def visualize_stepwise_adaboost(X, y, classifier,i,sample_weights=None, annotate=False, ax=None,n_estimators=10):
    assert set(y) == {-1, 1}, 'Expecting response labels to be ±1'

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        fig.set_facecolor('white')

    pad = 1
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    sizes = sample_weights * X.shape[0] * 100 if sample_weights is not None else np.ones(X.shape[0]) * 100

    # Plotting positive points scaled according to sample_weights
    X_pos, sizes_pos = X[y == 1], sizes[y == 1]
    ax.scatter(*X_pos.T, s=sizes_pos, marker='+', color='red')

    # Plotting negative points scaled according to sample_weights
    X_neg, sizes_neg = X[y == -1], sizes[y == -1]
    ax.scatter(*X_neg.T, s=sizes_neg, marker='.', c='blue')

    # TODO: Plot the decision boundary learned by the Adaboost classifier 
    x,y = np.meshgrid(np.linspace(x_min,x_max,500),np.linspace(x_min,x_max,500))
    X_new = np.c_[x.ravel(),y.ravel()]
    n_samples, n_features = X_new.shape
    final_prediction = np.zeros(n_samples)
    for j in range(i+1):
        final_prediction += (classifier.learner_weights[j])*(classifier.weak_learners[j].predict(X_new))
    z = np.sign(final_prediction)
    z = z.reshape(x.shape)
    ax.contourf(x,y,z,alpha=0.3)
    ax.set_title(f"strong_{i}")
    if(annotate==False):
        plt.show()

def plot_adaboost(X, y, clf=None, sample_weights=None, annotate=False, ax=None,i=0):
    assert set(y) == {-1, 1}, 'Expecting response labels to be ±1'

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        fig.set_facecolor('white')

    pad = 1
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    sizes = sample_weights * X.shape[0] * 100 if sample_weights is not None else np.ones(X.shape[0]) * 100

    # Plotting positive points scaled according to sample_weights
    X_pos, sizes_pos = X[y == 1], sizes[y == 1]
    ax.scatter(*X_pos.T, s=sizes_pos, marker='+', color='red')

    # Plotting negative points scaled according to sample_weights
    X_neg, sizes_neg = X[y == -1], sizes[y == -1]
    ax.scatter(*X_neg.T, s=sizes_neg, marker='.', c='blue')

    # TODO: Plot the decision boundary learned by the Adaboost classifier 
    if clf != None:
        x_points, y_points = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
        Z = clf.predict(np.c_[x_points.ravel(), y_points.ravel()])
        Z = Z.reshape(x_points.shape)
        ax.contourf(x_points, y_points, Z , alpha=0.3) 
    if(annotate==False):
        plt.show()
    else:
        ax.set_title(f"weak_{i}")

def make_dataset(n: int = 100):
    """ Generate a dataset for AdaBoost classifiers """

    n_per_class = int(n/2)

    np.random.seed(0)

    X, y = make_gaussian_quantiles(n_samples=n, n_features=2, n_classes=2)

    return X, y*2-1 #make y values ±1

class AdaBoostClassifier:
    def __init__(self):
        """
        Initialize an AdaBoost Classifier.

        Attributes:
        - weak_learners: List of weak learner models.
        - learner_weights: Weights assigned to weak learners (alpha_t's).
        - errors: Error rates of each iteration. (epsilon_t's)
        - sample_weights: Weight distribution over training samples. (w_t(i))
        """
        self.weak_learners = []
        self.learner_weights = []
        self.errors = []
        self.sample_weights = None

    def fit(self, X, y, n_estimators):
        """
        Fit the AdaBoost model with n_estimators iterations.

        Parameters:
        - X: 2D array, shape (n_samples, n_features), Input features.
        - y: 1D array, shape (n_samples,), Response labels (±1).
        - n_estimators: Number of boosting iterations.

        Returns:
        - The fitted AdaBoostClassifier.
        """

        n_samples, n_features = X.shape

        # TODO: Initialize arrays for sample weights, weak learners, learner weights, and errors

        # TODO: Initialize sample weights uniformly for the first iteration
        
        self.sample_weights = np.full(n_samples,1/n_samples)

        # TODO: For each iteration
        for t in range(n_estimators):
            # TODO: Create a weak learner (stump) and fit it with weighted samples
            model = DecisionTreeClassifier(max_depth = 1)
            model.fit(X,y,sample_weight = self.sample_weights)
            # TODO: Make predictions with the weak learner
            y_pred = model.predict(X)

            # TODO: Calculate weighted error and learner weight
            epsilon_t = 0.00
            for j in range(n_samples):
                if y[j] != y_pred[j]:
                    epsilon_t = epsilon_t + self.sample_weights[j]

            alpha_t = (np.log((1 - epsilon_t + 1e-10)/(epsilon_t + 1e-10)))/2

            # TODO: Update sample weights based on the weighted error
            arr = np.exp((-1)*y*y_pred*(alpha_t))
            arr = arr*(self.sample_weights)
            w_update = arr/np.sum(arr)

            # TODO: If not the final iteration, update sample weights for t+1
            if t != n_estimators-1:
                self.sample_weights = w_update

            # TODO: Save the results of the current iteration
            self.weak_learners.append(model)
            self.errors.append(epsilon_t)
            self.learner_weights.append(alpha_t)

        return self

    def predict(self, X):
        """
        Make predictions using the already fitted AdaBoost model.

        Parameters:
        - X: 2D array, shape (n_samples, n_features), Input features for predictions.

        Returns:
        - Predicted class labels, 1D array of shape (n_samples,).
        """
        # TODO: 
        final_prediction = np.zeros((X.shape[0]))
        for i in range(len(self.learner_weights)):
            final_prediction += (self.learner_weights[i])*(self.weak_learners[i].predict(X))
        return np.sign(final_prediction)


if __name__=="__main__":

  #Create a sample dataset
  X, y = make_dataset(n=20)
  plot_adaboost(X, y)

  #Train Adaboost classifier
  classifier = AdaBoostClassifier().fit(X, y, n_estimators=10)

  #TODO (OPTIONAL): Visualize the decision boundary by weak and strong learners (ensembled until the current iteration) at each iteration
  # visualize_stepwise_adaboost(X, y, classifier)
  plot_adaboost(X, y, classifier)
  train_err = (classifier.predict(X) != y).mean()
  print(f'Training error: {train_err:.1%}')

  fig,axis = plt.subplots(2,10,figsize=(200,16),dpi=100)
  for i in range(len(classifier.learner_weights)):
    plot_adaboost(X,y,classifier.weak_learners[i],ax=axis[0,i],annotate=True,i=i)
    visualize_stepwise_adaboost(X, y, classifier,i,n_estimators=10,ax=axis[1,i],annotate=True)
plt.show()