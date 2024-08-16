import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def load_data(file_name):
    """
    Load data from a CSV file.

    Parameters:
    file_name (str): The name of the CSV file containing the data.

    Returns:
    X (numpy.ndarray): Feature matrix containing data columns "X1" and "X2".
    y (numpy.ndarray): Target labels (Class).
    """
    df = pd.read_csv(file_name)
    X = np.column_stack([df["X1"], df["X2"]])
    y = np.array(df["Class"])
    return X, y

def plot_data(X, y, title, file_name, task=None, classifier=None):
    """
    TODO: Plot the data points and decision boundary and save it with same name as title.

    Parameters:
    X (numpy.ndarray): Feature matrix.
    y (numpy.ndarray): Target labels.
    title (str): Title for the plot.
    file_name (str): Save your plot as file_name
    task (str): task1 or task2
    classifier: Trained classifier for decision boundary.
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c = 'blue', label = "Class 0")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c = 'red', label = "Class 1")
    
    if task == "task1":
        # TODO: Find the slope, intercept, support_vectors using sklearn's in-built features namely classifier.support_vectors_, classifier.coef_, classifier.intercept_. Compute the margin. Recall margin = 1/||w||_2.  
        # TODO: Plot the decision boundary, the margin lines and the support vectors.

        #  margin = 1 / np.linalg.norm(w)
        margin = 1 
        w = classifier.coef_.flatten()
        m = (-1)*w[0]/w[1]
        b = classifier.intercept_[0]
        supp_vec = classifier.support_vectors_
        x_points = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
        y_points = m * x_points - (b / w[1])
        plt.plot(x_points, y_points, color = 'red', label = "Decision Boundary")
        plt.plot(x_points, y_points - (margin / w[1]), color = 'green', label = "Margin Line", linestyle = 'dotted')
        plt.plot(x_points, y_points + (margin / w[1]), color = 'blue', label = "Margin Line",linestyle = 'dotted')
        plt.scatter(supp_vec[:, 0], supp_vec[:, 1], c = 'black', marker = 'o', label = "Support Vectors")
        
    elif task == "task2":
        # TODO: Plot the non-linear decision boundary using plt.contour. No need to plot the margin lines. 
        x_points, y_points = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500), np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500))
        Z = classifier.decision_function(np.c_[x_points.ravel(), y_points.ravel()])
        Z = Z.reshape(x_points.shape)
        plt.contour(x_points, y_points, Z, levels = [0],linewidths = 2)

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(title)
    plt.legend()
    plt.savefig('images/' + file_name)

def svm_task_1():
    """
    Perform SVM Task 1: Load data, train Perceptron and SVM, and plot decision boundaries.
    """
    X, y = load_data("data1.csv")

    # Train SVM using sklearn and a linear kernel
    # TODO Write code here to train the SVM

    C_values = [1,10,100]
    for c in C_values:
        clf = SVC(C = c,kernel = 'linear')
        clf.fit(X,y)
        plot_data(X,y,f"SVM Linear for C = {c}",f"svm-C-{c}.png","task1",clf)
        

    # TODO Call plot_data below and produce three plots for C = 1, 10, 100. Save these plots as svm-C-1.png, svm-C-10.png and svm-C-100.png.

def svm_task_2():
    """
    Perform SVM Task 2: Load data, train SVM with RBF kernel, and plot decision boundary.
    """
    X, y = load_data("data2.csv")

    # Train SVM using sklearn with non-linear kernels and different C values
    # TODO Write code here to train the SVM
    c = 2000
    clf = SVC(C = c,kernel = 'rbf')
    clf.fit(X,y)
    plot_data(X,y,f"SVM RBF for C = {c}","svm-kernel.png","task2",clf)

    # Call plot_data below to plot a non-linear SVM decision boundary. Save the file in svm-kernel.png.

if __name__ == "__main__":
    """
    Save all the plots generated in this question in q2/images directory.
    """
    os.makedirs('images', exist_ok=True)
    
    # SVM Task 1
    svm_task_1()

    # SVM Task 2
    svm_task_2()
