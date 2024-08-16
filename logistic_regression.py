import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

class LogisticRegression:
    def __init__(self, epochs=1000, learning_rate=0.01, lambda_value=0):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lambda_value = lambda_value
        
    
    def sigmoid(self, z):
        """
        function to compute sigmoid of a vector z
        
        args:    z ---> nx1
        returns: nx1
        """
        z = np.array(z)           #z ---> n x 1
        return 1/(1+np.exp(-z))

    def loss_grad_function(self, X, y, theta, m):
        """
        function to compute the loss and gradient
        
        args:    X ---> nxd, y ---> nx1, theta ---> dx1, m ---> n
        returns: J ---> 1x1, grad ---> 1xd
        """
        #TODO: Write the code for the loss/grad function below
        epsilon = 1e-10
        sig_arr = self.sigmoid(X@theta)
        J = (-1/m)*((y.T)@(np.log((sig_arr+epsilon)/(1-sig_arr+epsilon)))) + np.mean(np.log(1-sig_arr+epsilon))*(-1)
        grad = ((X.T)@(sig_arr - y))
       # grad = grad/np.linalg.norm(grad)
        if(self.lambda_value != 0):
            J += self.lambda_value*((theta.T)@theta)
            grad += 2*self.lambda_value*(theta)
            grad = grad/np.linalg.norm(grad)
        else:
            grad = grad/np.linalg.norm(grad)
        return J,grad
    
    def gradient_descent(self, X, y, theta, m):
        """
        function to implement gradient descent
        
        args:    X ---> nxd, y ---> nx1, theta ---> dx1, m ---> n   
        returns: theta ---> dx1
        """
        #TODO: Write the code for full gradient descent over a 
        ## fixed number of training epochs in self.epochs by making calls to loss_grad_function  
        ## Print the losses returned by loss_grad_function over
        ## epochs to keep track of whether gradient descent is converging
        theta = np.zeros((X.shape[1],1))
        for ti in range(self.epochs):
            # prev_theta = theta
            loss,dw = self.loss_grad_function(X,y,theta,m)
            theta -= (self.learning_rate)*(dw) 
            # if np.linalg.norm(prev_theta - theta) <= 1e-10:
            #     break
        return theta
    
def map_feature_vectorized(df):
    #TODO: Write the code for creating additional features from X1 and X2
    m = len(df)
    df['X1X2'] = df['X1']*df['X2']
    df['X1**2'] = df['X1']*df['X1'] 
    df['X2**2'] = df['X2']*df['X2'] 
    df['X1**3'] = df['X1']*df['X1']*df['X1']
    df['X2**3'] = df['X2']*df['X2']*df['X2']
    X = np.hstack((np.ones((m,1)),df[['X1', 'X2' , 'X1X2' , 'X1**2','X2**2','X1**3','X2**3']].values))
    return X

def load_dataset(file_name):
    """
    function to load data file
    """
    df = pd.read_csv(file_name, sep=",", header=None)
    df.columns = ["X1", "X2", "label"]
    return df


def visualize_data(df, task, theta=None):
    """
    function to plot data points and decision boundaries
    """
    plt.figure(figsize=(7,5))
    ax = sns.scatterplot(x='X1', y='X2', hue='label', data=df, style='label', s=80)
    plt.title('Scatter plot of training data')
    
    #TODO: Insert code specific to each toy task (T1, T2, T3) to plot decision boundaries
    if task==1:
        x_vals = np.array([df['X1'].min(), df['X1'].max()])
        y_vals = -(theta[0][0] + theta[1][0]*x_vals)/(theta[2][0])
        plt.plot(x_vals, y_vals, label='Decision Boundary')
        
    elif task==2:
        u = np.linspace(0, 1.0, 50)
        v = np.linspace(0, 1.0, 50)

        U, V = np.meshgrid(u, v)
        X1X2 = U * V

        df = pd.DataFrame({'X1': U.flatten(), 'X2': V.flatten(), 'X1X2': X1X2.flatten()})
        m = len(df)
        X = np.hstack((np.ones((m, 1)), df[['X1', 'X2', 'X1X2']].values))
        z = X.dot(theta)
        Z = z.reshape(U.shape)

        ax.contour(u, v, Z, levels=[0], colors='green')
    
    elif task==3:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        U, V = np.meshgrid(u, v)
        UV_matrix = np.column_stack((U.flatten(), V.flatten()))
        df1 = pd.DataFrame(UV_matrix,columns=['X1','X2'])
        z_flat = np.dot(map_feature_vectorized(df1), theta)
        Z = z_flat.reshape(U.shape)

        ax.contour(U, V, Z, levels=[0], colors='green')
    
    path = "plot_"+str(task)+".png"
    plt.savefig(path)
    print("Data plot with decision boundary saved at: ", path)


def task_1():
    #-----------------------------------Q1---------------------------------------------
    #Complete the loss_grad_function, gradient_descent, and visualize_data task==1 code 
    #----------------------------------------------------------------------------------
    
    #Play around with the following hyperparameters to see which one works the best!
    learning_rate = 0.01
    epochs = 100000

    file_name = "task1_data.csv"
    df = load_dataset(file_name) 

    m = len(df)                                                             #get the number of training examples ---> n
    X = np.hstack((np.ones((m,1)),df[['X1', 'X2']].values))                 #dimension of X                      ---> n x d
    y = np.array(df.label.values).reshape(-1,1)                             #dimension of y                      ---> n x 1
    initial_theta = np.zeros(shape=(X.shape[1],1))                          #initialize theta value              ---> d x 1

    model = LogisticRegression(epochs, learning_rate)                       #initialize model
    theta = model.gradient_descent(X, y, initial_theta, m)               #call for the gradient descent function

    print('Theta found by gradient descent:\n', theta)
    visualize_data(df, 1, theta)                                            #plot the decision boundary on the data points

def task_2():
    #-----------------------------------Q2---------------------------------------------
    #Complete the missing parts below and the visualize_data function task==2 part
    #----------------------------------------------------------------------------------

    #Play around with the following hyperparameters to see which one works the best!
    learning_rate = 0.01
    epochs = 100000

    file_name = "task2_data.csv"
    df = load_dataset(file_name) 

    #TODO: Add the extra x1 \times x2 feature below to make the dataset linearly separable 
    df["X1X2"] = df['X1']*df['X2']                                                #add additional feature X1 x X2      

    m = len(df)                                                             #get the number of training examples ---> n
    #TODO: Create X and y, with the specified dimensions, to pass to gradient_descent
    X = np.hstack((np.ones((m,1)),df[['X1', 'X2' , 'X1X2']].values))                                                         #dimension of X                      ---> n x d
    y = np.array(df.label.values).reshape(-1,1)                                                      #dimension of y                      ---> n x 1
    initial_theta = np.zeros(shape=(X.shape[1],1))                          #initialize theta value              ---> d x 1

    model = LogisticRegression(epochs, learning_rate)                       #initialize model
    theta = model.gradient_descent(X, y, initial_theta, m)                  #call the gradient descent function

    print('Theta found by gradient descent:\n', theta)
    visualize_data(df, 2, theta)                                            #plot the decision boundary on the data points

    
def task_3():
    #-----------------------------------Q3---------------------------------------------------------
    #Complete the map_feature_vectorized function and the visualize_data function task==3 part
    #----------------------------------------------------------------------------------------------
    
    #Play around with the following hyperparameters to see which one works the best!
    #Try for different values of lambda ---> 0, 1, 10
    learning_rate = 0.01
    epochs = 100000
    lambda_value = 0

    file_name = "task3_data.csv"
    df = load_dataset(file_name)

    m = len(df)       
                                                          #get the number of training examples ---> n
    #TODO: Fill in map_feature_vectorized with the required transformation so that the dataset is separated using an LR classifier
    # df['X1X2'] = df['X1']*df['X2']
    # df['X1**2'] = df['X1']*df['X1'] 
    # df['X2**2'] = df['X2']*df['X2'] 
    # df['X1**3'] = df['X1']*df['X1']*df['X1']
    # df['X2**3'] = df['X2']*df['X2']*df['X2']
    X = map_feature_vectorized(df)                                      #dimension of X                      ---> n x d
    y = np.array(df.label.values).reshape(-1,1)                             #dimension of y                      ---> n x 1
    initial_theta = np.zeros(shape=(X.shape[1],1))                          #initialize theta value              ---> d x 1

    model = LogisticRegression(epochs, learning_rate, lambda_value)         #initialize model
    theta = model.gradient_descent(X, y, initial_theta, m)                  #call the gradient descent function 

    print('Theta found by gradient descent:\n', theta)
    visualize_data(df, 3, theta)                                            #plot the decision boundary on the data points

    
if __name__=='__main__':
    
    #TODO: Uncomment the line below to run task 1 
  #  task_1()
    
    #TODO: Uncomment the line below to run task 2 
  # task_2()
    
    #TODO: Uncomment the line below to run task 3 
   task_3()
    
