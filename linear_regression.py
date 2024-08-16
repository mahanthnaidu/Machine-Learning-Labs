import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from tqdm import tqdm
from time import time
import os 

np.random.seed(42)


class LinearRegressionBatchGD:
    def __init__(self, learning_rate=0.01, max_epochs=200, batch_size=None, regularization=None, reg_strength=0.01):
        '''
        Initializing the parameters of the model

        Args:
          learning_rate : learning rate for batch gradient descent
          max_epochs : maximum number of epochs that the batch gradient descent algorithm will run for
          batch-size : size of the batches used for batch gradient descent.
          regularization : str ('none','l1','l2','ElasticNet')
          reg_strength : float / Tuple(float) : It is float for l1 or l2 regularization. If reg_strength is $\lambda$, then the 
                      regularizer is $\lambda ||w||$
                      For ElasticNet, it is a pair of floats. The first float is the scaling parameter for the regularizer, and the
                      second parameter interpolates between l1 and l2 regularization. Thus, if reg_strength is $(\lambda, \alpha)$, 
                      then regularizer is $\lambda (\alpha ||w||_{1} + (1-\alpha) ||w||_{2})$

        Returns: 
          None 
        '''
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.weights = None
        # Added Regularization based parameters
        self.regularization = regularization
        self.reg_strength = reg_strength

    def fit(self, X, y, X_dev, y_dev):
        '''
        This function is used to train the model using batch gradient descent.

        Args:
          X : 2D numpy array of training set data points. Dimensions (n x (d+1))
          y : 2D numpy array of target values in the training dataset. Dimensions (n x 1)
          X_dev : 2D numpy array of development set data points. Dimensions (n x (d+1))
          y_dev : 2D numpy array of target values in the training dataset. Dimensions (n x 1)
          early_stopping_patience : Maximum number of increases in the loss values that can be tolerated while model training.

        Returns : 
          None
        '''
        if self.batch_size is None:
            self.batch_size = X.shape[0]

        # Initialize the weights
        self.weights = np.zeros((X.shape[1], 1))

        prev_weights = self.weights

        self.error_list = []
        for epoch in tqdm(range(self.max_epochs), desc="Training Progress"):

            batches = create_batches(X, y, self.batch_size)
            for batch in batches:
                X_batch, y_batch = batch
                dW = self.compute_gradient(X_batch, y_batch, self.weights)
                prev_weights = self.weights
                self.weights = self.weights - self.learning_rate * dW

            # Calculate validation loss
            val_loss = self.compute_rmse_loss(X_dev, y_dev, self.weights)
            self.error_list.append(val_loss[0][0])

            if np.linalg.norm(self.weights - prev_weights) < 1e-5:
                print(f" Stopping at epoch {epoch}.")
                break

        print("Training complete.")
        print("Mean validation RMSE loss : ", np.mean(self.error_list))
        print("Batch size: ", self.batch_size)
        print("learning rate: ", self.learning_rate)

        # plot_loss(self.error_list, self.batch_size, self.regularization, self.reg_strength)

    def predict(self, X):
        '''
        This function is used to predict the target values for the given set of feature values

        Args:
          X: 2D numpy array of data points. Dimensions (n x (d+1)) 

        Returns:
          2D numpy array of predicted target values. Dimensions (n x 1)
        '''
        return X @ self.weights

    def compute_rmse_loss(self, X, y, weights):
        '''
        This function computes the Root Mean Square Error (RMSE) loss

        Args:
          X : 2D numpy array of data points. Dimensions (n x (d+1))
          y : 2D numpy array of target values. Dimensions (n x 1)
          weights : 2D numpy array of weights of the model. Dimensions ((d+1) x 1)

        Returns:
          loss : 2D numpy array of RMSE loss. Dimensions (1x1)
        '''
        y_pred = X @ weights
        difference = y_pred - y
        loss = np.sqrt(np.dot(difference.T, difference)/X.shape[0])
        return loss

    def compute_gradient(self, X, y, weights):
        '''
        This function computes the gradient of the regularized MSE loss w.r.t the weights

        Args:
          X : 2D numpy array of data points. Dimensions (n x (d+1))
          y : 2D numpy array of target values. Dimensions (n x 1)
          weights : 2D numpy array of weights of the model. Dimensions ((d+1) x 1)

        Returns:
          dw : 2D numpy array of gradients w.r.t weights. Dimensions ((d+1) x 1)
        '''
        # TODO compute the prediction y_pred
        y_pred = None
        # TODO find the gradient descent update of w arising from the loss

        # size of the dataset 
        n = X.shape[0]  
        # Number of features
        # d = X.shape[1] - 1  
        # print(d)
        #print(n)
        
        #calculating predictions 
        y_pred = X @ weights  

        # print(y_pred.shape)
        # print(y_pred)
        
        # Gradient of the loss wrt the weights
        gradient_loss = 2*X.T @ (y_pred - y)/n

        # reg_term = None 

        # If no regularization 
        if self.regularization == "none":

            dw = gradient_loss

        # If regularization is L1 
        elif self.regularization == "l1":

            regularization_term = self.reg_strength*np.sign(weights)
            dw = gradient_loss + regularization_term

        # If regularization is L2 
        elif self.regularization == "l2":
            regularization_term = 2*self.reg_strength*weights
            dw = gradient_loss + regularization_term
        
        # If regularization is ElasticNet
        elif self.regularization == "ElasticNet":
            alpha = self.reg_strength[1]
            lambda_param = self.reg_strength[0]
            l1_regularization_term = lambda_param*alpha*np.sign(weights)
            l2_regularization_term = (1 - alpha)*2*lambda_param*weights
            regularization_term = l1_regularization_term + l2_regularization_term
            dw = gradient_loss + regularization_term

        return dw


def plot_loss(error_list, batch_size, regularization, reg_strength):
    '''
    This function plots the loss for each epoch.

    Args:
      error_list : list of validation loss for each epoch
      batch_size : size of one batch

    Returns:
      None
    '''

    plt.plot(error_list)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.savefig(f"./figures/loss_"+str(batch_size)+"_"+regularization+"_"+str(reg_strength) +
                ".png" if regularization != "none" else f"./figures/loss_"+str(batch_size)+".png")
    plt.close()


def create_batches(X, y, batch_size):
    '''
    This function is used to create the batches of randomly selected data points.

    Args:
      X : 2D numpy array of data points. Dimensions (n x (d+1))
      y : 2D numpy array of target values. Dimensions (n x 1)

    Returns:
      batches : list of tuples with each tuple of size batch size.
    '''
    batches = []
    data = np.hstack((X, y))
    np.random.shuffle(data)
    num_batches = data.shape[0]//batch_size
    i = 0
    for i in range(num_batches+1):
        if i < num_batches:
            batch = data[i * batch_size:(i + 1)*batch_size, :]
            X_batch = batch[:, :-1]
            Y_batch = batch[:, -1].reshape((-1, 1))
            batches.append((X_batch, Y_batch))
        if data.shape[0] % batch_size != 0 and i == num_batches:
            batch = data[i * batch_size:data.shape[0]]
            X_batch = batch[:, :-1]
            Y_batch = batch[:, -1].reshape((-1, 1))
            batches.append((X_batch, Y_batch))
    return batches


def standard_scaler(data, mean=None, std=None):
    """
    Scales the given data using the provided mean and standard deviation. 
    If no mean and standard deviation are provided, they are computed from the data.

    Args:
    - data (numpy array): The data to be scaled.
    - mean (numpy array, optional): The mean value used for scaling.
    - std (numpy array, optional): The standard deviation used for scaling.

    Returns:
    - Scaled data, mean, standard deviation.
    """
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

    # Ensure standard deviation is not zero to avoid division errors
    std = np.where(std == 0, 1, std)

    return (data - mean) / std, mean, std


def scaling(X_train, y_train, X_dev, y_dev):
    """
    Scales feature and target data for training and dev sets.

    Args:
    - X_train (numpy array): Training feature data.
    - y_train (numpy array): Training target data.
    - X_dev (numpy array): Development feature data.
    - y_dev (numpy array): Development target data.

    Returns:
    - Scaled X_train, y_train, X_dev, y_dev, and the mean and standard deviation of X_train.
    """
    X_train, X_mean, X_std = standard_scaler(X_train)
    y_train, y_mean, y_std = standard_scaler(y_train)
    X_dev, _, _ = standard_scaler(X_dev, X_mean, X_std)
    y_dev, _, _ = standard_scaler(y_dev, y_mean, y_std)

    return X_train, y_train, X_dev, y_dev, X_mean, X_std, y_mean, y_std


def load_train_dev_dataset(extend=0):
    """
    Loads training and development datasets, scales them, and optionally extends features.

    Args:
    - extend (int, optional): Whether to extend features using two-way interactions.

    Returns:
    - Scaled and optionally extended X_train, y_train, X_dev, y_dev, and scaling parameters of X_train.
    """
    train_set = pd.read_csv(f"./splits/train_data.csv", header=None)
    dev_set = pd.read_csv(f"./splits/dev_data.csv", header=None)

    X_train = train_set.iloc[:, 1:].to_numpy()
    if extend:
        X_train = get_two_way_interactions(X_train)
    y_train = train_set.iloc[:, 0].to_numpy().reshape(-1, 1)

    X_dev = dev_set.iloc[:, 1:].to_numpy()
    if extend:
        X_dev = get_two_way_interactions(X_dev)
    y_dev = dev_set.iloc[:, 0].to_numpy().reshape(-1, 1)

    X_train, y_train, X_dev, y_dev, X_mean, X_std, y_mean, y_std = scaling(
        X_train, y_train, X_dev, y_dev)

    X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_dev = np.c_[np.ones((X_dev.shape[0], 1)), X_dev]

    return X_train, y_train, X_dev, y_dev, X_mean, X_std, y_mean, y_std


def load_test_dataset(X_mean, X_std, extend=0):
    """
    Loads test dataset, scales it using provided parameters, and optionally extends features.

    Args:
    - X_mean (numpy array): Mean of training feature data used for scaling.
    - X_std (numpy array): Standard deviation of training feature data used for scaling.
    - extend (int, optional): Whether to extend features using two-way interactions.

    Returns:
    - Scaled and optionally extended X_test and y_test.
    """
    X_test = pd.read_csv(f"./splits/test_data.csv", header=None).to_numpy()
    if extend:
        X_test = get_two_way_interactions(X_test)
    y_test = pd.read_csv(f"./splits/test_labels.csv",
                         header=None).to_numpy().reshape(-1, 1)

    X_test, _, _ = standard_scaler(X_test, X_mean, X_std)
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    return X_test, y_test


def evaluate_model(weights, X, y, ymean, ystd):
    '''
    This function is used to calculate the RMSE loss on test dataset.

    Args:
      weights : 2D numpy array of weights of the model. Dimensions ((d+1) x 1)
      X : 2D numpy array of test set data points. Dimensions (n x (d+1))
      y : 2D numpy array of target values in the test dataset. Dimensions (n x 1)
      y_min : minimum value of target labels in the train set.
      y_max : maximum value of target labels in the train set.
    '''
    y_pred_scaled = X @ weights
    y_pred_actual = (y_pred_scaled * ystd + ymean)
    difference = (y_pred_actual) - y
    rmse = np.sqrt(np.mean(difference**2))

    return rmse


def grid_search(X_train, y_train, X_dev, y_dev, y_train_mean, y_train_std, reg_strengths, regularizations, learning_rate=0.08, max_epochs=200, batch_size=None):
    # dictionary mapping each regularizer to an list of losses on the dev set for each possible reg_strength
    dev_losses = {reg: [] for reg in regularizations}
    best_dev_loss = np.inf
    best_params = {}

    # TODO For each regularizer, grid search over the regularization strength. To do this, get the loss on the dev set for each
    # value of the regularizer strength and output the parameter that minimizes this loss

    # for each regularization and for each strength calculating the loss and taking the small one 

    for reg_type in regularizations:
        for reg_strength in reg_strengths:

            # taking the model 
            model = LinearRegressionBatchGD(learning_rate=learning_rate,max_epochs=max_epochs,batch_size=batch_size,regularization=reg_type,reg_strength=reg_strength)

            # print("regularization : ", reg)
            # print("regularization strength :", reg_strength)
            
            # Fitting the model on the training data
            model.fit(X_train, y_train, X_dev, y_dev)


            # computing the loss using given function
            # loss = model.compute_rmse_loss(X_dev, y_dev, model.weights)
            y_dev_pred = model.predict(X_dev)
            dev_loss = np.sqrt(np.mean((y_dev_pred - y_dev)**2))
            # print("My loss calculation :", dev_loss)
            # print("function loss values", loss[0,0])

            # if the loss is less taking that as the best parameter 
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_params = {"regularization": reg_type, "reg_strength": reg_strength}

            dev_losses[reg_type].append(dev_loss)
            
            


    return dev_losses, best_dev_loss, best_params


def grid_search_ElasticNet(X_train, y_train, X_dev, y_dev, y_train_mean, y_train_std, lambdas, alphas, learning_rate=0.08, max_epochs=200, batch_size=None):
    # list of dictionaries. Each element will be of the form {'alpha': ,'lambda': , 'dev_loss': }
    dev_losses = []
    # TODO grid search over alpha and lambda values. For each (alpha, lambda) pair you try, get the dev loss and
    # append it to dev_losses as a dictionary {'alpha': ,'lambda': , 'dev_loss': }
    # This grid search may be either the full grid search or the heuristic-based version described in the lab description for Q3.
    for alpha in alphas:
        for reg_strength in lambdas:
            # print("alpha: ", alpha)
            # print("reg_strength :", reg_strength)
            # print("ElasticNet")
            model = LinearRegressionBatchGD(learning_rate=learning_rate,max_epochs=max_epochs,batch_size=batch_size,regularization="ElasticNet",reg_strength=(reg_strength, alpha))

            #fitting the model 
            model.fit(X_train, y_train, X_dev, y_dev)   

            y_dev_pred = model.predict(X_dev)
            dev_loss = np.sqrt(np.mean((y_dev_pred - y_dev)**2))

            # loss = model.compute_rmse_loss(X_dev, y_dev, model.weights)


            # dev_losses.append({'alpha': alpha, 'lambda': reg_strength, 'dev_loss': loss[0,0]})

            dev_losses.append({'alpha': alpha, 'lambda': reg_strength, 'dev_loss': dev_loss})

    return dev_losses


def get_two_way_interactions(X):
    """
    Compute unique two-way interactions for one-third of the initial features in a dataset X using numpy.

    Args:
    - X (numpy.ndarray): A 2-dimensional array of shape (number_of_samples, number_of_features).

    Returns:
    - X_extended (numpy.ndarray): Extended dataset with interactions, shape (number_of_samples, number_of_features + number_of_interactions).
    """

    # 1. Select one-third of the features
    subset_size = X.shape[1] // 3
    X_subset = X[:, :subset_size]
    # After this step, X_subset shape: (number_of_samples, subset_size)

    # Follow step by step to get X_extended, also follow the example detailed in Two_Way_Interactions.pdf  

    # 2. Compute all two-way interactions for the subset
    temp = X_subset[:, :, np.newaxis]*X_subset[:, np.newaxis, :]


    temp = temp.reshape(-1, temp.shape[1]*temp.shape[2])
    # After broadcasting and multiplying, the shape: (number_of_samples, subset_size, subset_size)

    # 3. Create a mask for the upper triangle to exclude duplicates
    mask = np.triu_indices(subset_size, k=1)
    # mask shape: (subset_size, subset_size)

    # 4. Apply mask and reshape
    result = temp[:, mask[0]*subset_size + mask[1]]
    # After masking and reshaping, interactions shape: (number_of_samples, (subset_size * (subset_size + 1)) // 2)

    # 5. Combine the original dataset with the interactions
    X_extended = np.hstack((X, result))
    # After horizontal stacking, X_extended shape: (number_of_samples, number_of_features + (subset_size * (subset_size + 1)) // 2)

    return X_extended


def plot_reg_strength_vs_dev_loss(reg_strengths, dev_losses, regularizations):
    '''
    Plots dev loss vs reg strength for each of the regularizations. Just for illustration, not to be evaluated.
        Inputs: reg_strengths: List(float) : list of regularization strength
                dev_losses : dict(String, List(float)) : maps each regulization to its list of dev losses for the different reg_strengths
                regularization : List(str) : list of different regularizations
    '''
    if not os.path.exists("figures"):
        os.makedirs("figures")
    plt.figure(figsize=(10, 6))
    for reg in regularizations:
        plt.plot(reg_strengths, dev_losses[reg], marker='o',label=f"Regularization: {reg.upper()}")

    plt.xlabel("Regularization Weight(reg_Weight)")
    plt.ylabel("Dev Loss")
    plt.title("Regularization Strength vs. Development Loss")
    plt.legend()
    plt.savefig("figures/grid_search_reg_strength.png")


def plot_reg_strength_vs_dev_loss_ElasticNet(dev_losses):
    '''
    Makes a 3D plot of dev loss vs alpha and lambda. Just for illustration, not to be evaluated.
    Input : List(dict(str, float)) : Takes in a list with each element being a dictionary of form {'alpha': , 'lambda': , 'dev_loss': }
    '''
    plt.figure(figsize=(6, 6))
    ax = plt.axes(projection='3d')
    alphas = []
    lambdas = []
    losses = []
    for e in dev_losses:
        alphas.append(e['alpha'])
        lambdas.append(e['lambda'])
        losses.append(e['dev_loss'])

    ax.scatter3D(np.log10(np.array(alphas)), np.array(lambdas), np.array(
        losses), c=np.array(losses), cmap='viridis')
    plt.savefig("figures/grid_search_ElasticNet.png")


def save_prediction(Xmean, Xstd, ymean, ystd, weights, extend=0):
    """
    Function to save the model predictions on hidden_test_dataset.

    Args:
    - Xmean (numpy array): Mean of the features from the training set.
    - Xstd (numpy array): Standard deviation of the features from the training set.
    - ymean (float): Mean value of target labels of the training dataset.
    - ystd (float): Standard deviation of target labels of the training dataset.
    - weights (numpy array): Trained model weights.
    - extend (int): If 1, two way interactions are also included.

    Returns:
    - None.
    """

    # Load the hidden test data
    X = pd.read_csv(f"./splits/hidden_test_data.csv", header=None).to_numpy()

    if extend:
        X = get_two_way_interactions(X)

    # Scale the X values using Xmean and Xstd
    X = (X - Xmean)/Xstd

    # Add a bias term (1) to each feature vector in X
    X = np.c_[np.ones((X.shape[0], 1)), X]

    # Predict using the trained weights
    predictions = (X @ weights)

    # Reverse the standard scaling for the predicted values
    y_pred_hidden_test = (predictions * ystd + ymean)

    # Save the predictions to a CSV file
    pd.DataFrame(y_pred_hidden_test, columns=['Year']).to_csv(
        '210050161.csv', index=True, header=True, index_label="ID")


if __name__ == '__main__':

    learning_rate = 1e-4
    batch_size = 128
    max_epochs = 50
    # Play around and tune these values to see which strength improves over unregularized GD.
    reg_strengths = [0.000008, 0.000016, 0.000032, 0.000064, 0.000128]
    regularizations = ["none","l1", "l2"]  # Types of regularization

    # ----------------------- Q1 -----------------------------

    X_train, y_train, X_dev, y_dev, X_mean, X_std, y_train_mean, y_train_std = load_train_dev_dataset()
    X_test, y_test = load_test_dataset(X_mean, X_std)

    dev_losses, best_dev_loss, best_params = grid_search(X_train, y_train, X_dev, y_dev, y_train_mean, y_train_std,reg_strengths, regularizations, learning_rate, max_epochs, batch_size)
    print(best_params)
    #print(dev_losses)
    plot_reg_strength_vs_dev_loss(reg_strengths, dev_losses, regularizations)

    best_model = LinearRegressionBatchGD(learning_rate=learning_rate, max_epochs=max_epochs, batch_size=batch_size,regularization=best_params["regularization"], reg_strength=best_params["reg_strength"])
    best_model.fit(X_train, y_train, X_dev, y_dev)

    test_loss = evaluate_model(best_model.weights, X_test, y_test, y_train_mean, y_train_std)
    print("Test Loss using Best Parameters:", test_loss)

    #----------------------- Q2 -----------------------------

    #Uncomment the following, complete the code for two-way interactions and play around with different values of reg_strength and regularization

    X_train_extended, y_train, X_dev_extended, y_dev, X_mean_extended, X_std_extended, y_train_mean, y_train_std = load_train_dev_dataset(1)
    X_test_extended, y_test = load_test_dataset(X_mean_extended, X_std_extended, 1)

    model = LinearRegressionBatchGD(learning_rate=learning_rate, max_epochs=max_epochs,batch_size=batch_size, regularization='none', reg_strength=0.1)
    model.fit(X_train_extended, y_train, X_dev_extended, y_dev)

    test_loss = evaluate_model(model.weights, X_test_extended, y_test, y_train_mean, y_train_std)
    print("Test Loss using the given parameters:", test_loss)

    # ----------------------- Q3 -----------------------------

    lambdas = [0.001, 0.05, 0.01, 0.05, 0.1]
    alphas = [0.2, 0.4, 0.6, 0.8]

    X_train, y_train, X_dev, y_dev, X_mean, X_std, y_train_mean, y_train_std = load_train_dev_dataset()
    X_test, y_test = load_test_dataset(X_mean, X_std)

    dev_losses = grid_search_ElasticNet(X_train, y_train, X_dev, y_dev, y_train_mean, y_train_std,lambdas, alphas, learning_rate, max_epochs, batch_size)

    plot_reg_strength_vs_dev_loss_ElasticNet(dev_losses)

    # print(dev_losses)
    t = sorted(dev_losses,key=lambda x : x['dev_loss'])
    print("Minimum dev loss:",t[0]['dev_loss'])

    best_model = LinearRegressionBatchGD(learning_rate=learning_rate, max_epochs=max_epochs, batch_size=batch_size,regularization="ElasticNet", reg_strength=(t[0]['lambda'],t[0]['alpha']))
    best_model.fit(X_train, y_train, X_dev, y_dev)

    test_loss = evaluate_model(best_model.weights, X_test, y_test, y_train_mean, y_train_std)
    print("Test Loss using Best Parameters:", test_loss)


    # -----------------------   Kaggle -----------------------------
    # Save predictions for the best model you get.
    # save_prediction(X_mean, X_std,
    #                 y_train_mean, y_train_std, model.weights, 0)
