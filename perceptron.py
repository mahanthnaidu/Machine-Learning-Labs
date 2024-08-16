import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os

def seed_everything(seed = 42):
    random.seed(seed)
    np.random.seed(seed)

def make_directory_structure():
    os.makedirs('./images/average', exist_ok=True)
    os.makedirs('./images/vanilla', exist_ok=True)

def produce_sample_dataset(input_dim, data_size):
    y = np.random.randint(0,2, size=(data_size,))
    y = 2*y-1
    x = (y==-1)[:,None]*np.random.normal(loc=0, scale=1, size=(data_size, input_dim)) + (y==1)[:,None]*np.random.normal(loc=2, scale=1, size=(data_size, input_dim))
    return x,y

def plot_decision_boundary(x,y,w,name="boundary"):
    plt.figure()
    plt.scatter(x[y==-1][:,0],x[y==-1][:,1], c=['blue'])
    plt.scatter(x[y==1][:,0],x[y==1][:,1], c=['red'])
    plt.axline((0,-w[-1]/w[1]),(1,-(w[0]+w[-1])/w[1]), c='black', marker='o')
    plt.savefig(f"{name}.png")


def test_train_split(x, y, frac=0.8):
    '''
    Input: x: np.ndarray: features of shape (data_size, input_dim)
           y: np.ndarray: labels of shape (data_size,)
           frac: float: fraction of dataset to be used for test set
    Output: x_train, y_train, x_test, y_test
    '''
    cut = math.trunc(frac*x.shape[0])
    return x[:cut], y[:cut], x[cut:], y[cut:]

class Perceptron():
    def __init__(self, input_dim, lam=0.8):
        '''
            Input: input_dim: int: size of input
                   lam: float: parameter of geometric moving average. Moving average is calculated as
                            a_{t+1} = lam*a_t + (1-lam)*w_{t+1}
        '''
        self.weights = np.random.randn(input_dim+1) # the last weight is for the bias term
        self.running_avg_weights = self.weights
        self.lam = lam
    
    def fit(self, x, y, lr = 0.001, epochs = 100):
        '''
            Input: x: np.ndarray: training features of shape (data_size, input_dim)
                   y: np.ndarray: training labels of shape (data_size,)
                   lr: float: learning rate
                   epochs: int: number of epochs
        '''
        data_size = x.shape[0]
        x_new = np.copy(x)
        ones_column = np.ones((data_size, 1), dtype=x.dtype)
        x_new = np.hstack((x_new, ones_column))
        for e in range(epochs):
            # TODO concatenate 1's at the end of x to make it of the shape (data_size, input_dim+1) so that w[-1] can be the bias term
            # TODO calculate y_pred directly using x and w. Do not use the predict() function here, that is only for test
            # TODO perform the weight update
            # TODO update the running average of weights
            # plotting the decision boundary at this epoch
            y_pred = np.sign(np.matmul(x_new,self.weights))
            update = lr*(np.matmul(x_new.T,(y-y_pred)/2))
            self.weights = self.weights + update
            self.running_avg_weights = (self.lam)*(self.running_avg_weights) + (1-self.lam)*(self.weights)
            plot_decision_boundary(x,y,p.get_decision_boundary(False),f"images/vanilla/{e:05d}")
            plot_decision_boundary(x,y,p.get_decision_boundary(True),f"images/average/{e:05d}")

    def predict(self, x, running_avg = False):
        '''
            Input: x: np.ndarray: test features of shape (data_size, input_dim)
                   running_avg: bool: choose whether to use the running average weights for prediction
            Output: y_pred: np.ndarray: predicted labels of shape (data_size,)
        '''
        # TODO concatenate 1's at the end of x to make it of the shape (data_size, input_dim+1) so that w[-1] can be the bias term
        # TODO make y_pred using either the final weight vector or the moving average of the weights
        # TODO use np.sign to compute y_pred
        data_size = x.shape[0]
        x_new = np.copy(x)
        ones_column = np.ones((data_size, 1), dtype=x.dtype)
        x_new = np.hstack((x_new, ones_column))
        if running_avg:
            y_pred = np.sign(np.matmul(x_new,self.running_avg_weights))
        else:
            y_pred = np.sign(np.matmul(x_new,self.weights))
        return y_pred
    
    def get_decision_boundary(self, running_avg = False):
        '''
            Input: running_avg: bool: choose whether to use the running average weights for prediction
            Output: np.ndarray of shape (input_dim+1,) representing the decision boundary
        '''
        if running_avg:
            return self.running_avg_weights
        else:
            return self.weights

if __name__ == "__main__":
    seed_everything()
    make_directory_structure()

    input_dim = 2
    data_size = 1000
    x, y = produce_sample_dataset(input_dim, data_size)
    x_train, y_train, x_test, y_test = test_train_split(x, y)
    
    p = Perceptron(input_dim)
    # TODO fit the perceptron on the train set
    p.fit(x_train,y_train)

    # TODO predict on the test set using the last weight vector and print accuracy
    y_last_weight = p.predict(x_test,False)
    s = np.count_nonzero(y_last_weight - y_test)
    print(100 - (s/y_test.shape[0])*100)

    # TODO predict on the test set using the average weight vector and print accuracy
    y_average_weight = p.predict(x_test,True)
    s = np.count_nonzero(y_average_weight - y_test)
    print(100 - (s/y_test.shape[0])*100)
    
