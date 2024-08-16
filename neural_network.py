import numpy as np

np.random.seed(42)

"""
Sigmoid activation applied at each node.
"""
def sigmoid(x):
    # cap the data to avoid overflow?
    x[x>100] = 100
    x[x<-100] = -100
    return 1/(1+np.exp(-x))

"""
Derivative of sigmoid activation applied at each node.
"""
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

class NN:
    def __init__(self, input_dim, hidden_dim, activation_func = sigmoid, activation_derivative = sigmoid_derivative):
        """
        Parameters
        ----------
        input_dim : TYPE
            DESCRIPTION.
        hidden_dim : TYPE
            DESCRIPTION.
        activation_func : function, optional
            Any function that is to be used as activation function. The default is sigmoid.
        activation_derivative : function, optional
            The function to compute derivative of the activation function. The default is sigmoid_derivative.

        Returns
        -------
        None.

        """
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative
        # TODO: Initialize weights and biases for the hidden and output layers

        self.w1 = np.random.normal(0.0,1.0,(input_dim, hidden_dim))
        self.b2 = np.random.normal(0,1,(1,1))
        self.w2 = np.random.normal(0.0,1.0,(hidden_dim, 1))
        self.b1 = np.random.normal(0,1,(1, hidden_dim))
        self.yhat = None
        return
        
    def forward(self, X):
        # Forward pass
        # TODO: Compute activations for all the nodes with the activation function applied 
        # for the hidden nodes, and the sigmoid function applied for the output node
        # TODO: Return: Output probabilities of shape (N, 1) where N is number of examples
        return sigmoid((sigmoid((X @ self.w1) + self.b1) @ self.w2) + self.b2)
    
    def backward(self, X, y, learning_rate):
        # Backpropagation
        # TODO: Compute gradients for the output layer after computing derivative of sigmoid-based binary cross-entropy loss
        # TODO: When computing the derivative of the cross-entropy loss, don't forget to divide the gradients by N (number of examples)  
        # TODO: Next, compute gradients for the hidden layer
        # TODO: Update weights and biases for the output layer with learning_rate applied
        # TODO: Update weights and biases for the hidden layer with learning_rate applied
        y_new = np.copy(y)
        y_new.resize(X.shape[0],1)
        dLdyhat = (-1)*(((y_new)/(self.yhat)) - ((1 - y_new)/(1 - self.yhat)))
        dLdo = (self.yhat)*(1 - self.yhat)

        dw2 = ((sigmoid((X @ self.w1) + self.b1)).T) @ (dLdyhat * dLdo)
        dw1 = (X.T) @ (((dLdo*dLdyhat) @ ((self.w2).T)) * (sigmoid_derivative((X @ self.w1) + self.b1)))
        db2 = np.mean(dLdo*dLdyhat)
        db1 = np.mean(((dLdo*dLdyhat) @ ((self.w2).T)) * (sigmoid_derivative((X @ self.w1) + self.b1)),axis=0)

        self.w1 = self.w1 - (learning_rate/X.shape[0])*(dw1)
        self.w2 = self.w2 - (learning_rate/X.shape[0])*(dw2)
        self.b1 = self.b1 - (learning_rate)*(db1)
        self.b2 = self.b2 - (learning_rate)*(db2)

        return 

    def train(self, X, y, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            # Forward pass
            self.yhat = self.forward(X)
            # Backpropagation and gradient descent weight updates
            self.backward(X, y, learning_rate)
            # TODO: self.yhat should be an N times 1 vector containing the final
            # sigmoid output probabilities for all N training instances 
            # TODO: Compute and print the loss (uncomment the line below)
            epsilon = 1e-10
            loss = np.mean(-y*np.log(self.yhat + epsilon) - (1-y)*np.log(1-self.yhat + epsilon))
            # TODO: Compute the training accuracy (uncomment the line below)
            accuracy = np.mean((self.yhat > 0.5).reshape(-1,) == y)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            if accuracy >= 1.0000 :
                print(f"Stopping at Epoch {epoch+1}/{num_epochs}")
                break 
            self.pred('pred_train.txt')
            
    def pred(self,file_name='pred.txt'):
        pred = self.yhat > 0.5
        with open(file_name,'w') as f:
            for i in range(len(pred)):
                f.write(str(self.yhat[i]) + ' ' + str(int(pred[i])) + '\n')

# Example usage:
if __name__ == "__main__":
    # Read from data.csv 
    csv_file_path = "data_train.csv"
    eval_file_path = "data_eval.csv"
    
    data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=0)
    data_eval = np.genfromtxt(eval_file_path, delimiter=',', skip_header=0)
    # Separate the data into X (features) and y (target) arrays
    X = data[:, :-1]
    y = data[:, -1]
    X_eval = data_eval[:, :-1]
    y_eval = data_eval[:, -1]

    # Create and train the neural network
    input_dim = X.shape[1]
    hidden_dim = 4
    learning_rate = 0.05
    num_epochs = 100
    
    model = NN(input_dim, hidden_dim)
    model.train(X**2, y, learning_rate, num_epochs) #trained on concentric circle data 

    test_preds = model.forward(X_eval**2)
    model.pred('pred_eval.txt')

    test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
    print(f"Test accuracy: {test_accuracy:.4f}")
