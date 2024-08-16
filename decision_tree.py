import numpy as np

def entropy(targets):
    """
    Calculate the entropy of a set of labels.

    Args:
        targets (numpy.ndarray): numpy array of labels (number_of_sets_of_labels x number_of_examples)
    
    Returns:
        numpy.ndarray: An array of entropies where each value corresponds to the entropy of one set of labels.   
    """
    ### remove below

    # Get unique labels and their counts in the target dataset
    _, label_counts =  np.unique(targets, return_counts=True)

    # Calculate the probabilities of each unique label occurring
    probabilities = label_counts / len(targets)

    # Calculate the entropy value using the formula: -Σ(p_i * log2(p_i))
    # Add a small constant (1e-12) to prevent logarithm of zero
    entropy_value = -np.sum(probabilities * np.log2(probabilities + (1e-12)))
    
    return entropy_value


def information_gain(examples, targets, attr):
    """
    Calculate the information gain from splitting the dataset on a given attribute.

    Args:
        examples (numpy.ndarray): A numpy array of shape (number_of_examples, number_of_features) containing input examples.
        targets (numpy.ndarray): A numpy array of shape (number_of_examples) containing target labels.
        attr (int): An integer specifying the attribute to split on, ranging from 0 to (number_of_features - 1) inclusive.
    
    Returns:
        float: The information gain obtained from splitting the dataset on the specified attribute.
    """
    
    ### remove below

    # Get unique values and their counts for the specified attribute
    unique_attribute_values, attribute_value_counts = np.unique(examples[:, attr], return_counts=True)

    # Calculate the entropy of the entire dataset before splitting
    entropy_before_split = entropy(targets)

    ## Calculate the entropy after splitting for all unique values of the attribute

    # Calculate the weighted sum of entropies for each attribute value
    H_after = 0
    for i in range(unique_attribute_values.size):
        entropy_input = targets[examples[:, attr] == unique_attribute_values[i]]
        weighted_entropies = (attribute_value_counts[i] / len(entropy_input)) * entropy(entropy_input)
        H_after += weighted_entropies
    
    # Calculate and return the information gain
    information_gain_value = entropy_before_split - H_after

    return information_gain_value


class IDTNode:
    """Class describing a decision node on the tree."""

    def __init__(self, default, attr_to_split = -1, children = {}):
        """
        Initialize an IDTNode object.

        Args:
            default: The default prediction value for this node.
            attr_to_split: The index of the attribute to split on (-1 for leaf nodes).
            children: A dictionary containing child nodes.
        """

        self.attr_to_split = attr_to_split
        self.children = children
        self.default = default
    
    def predict(self, attributes):
        """
        Predict the outcome based on the input attributes.

        Args:
            attributes: A list of attribute values to make a prediction.

        Returns:
            The predicted outcome.
        """

        # If this node is a leaf node (no attribute to split on), return the default prediction.
        if (self.attr_to_split == -1):
            return self.default
        
        # Retrieve the value of the splitting attribute for this example.
        key = attributes[self.attr_to_split]

        # Check if there is a child node associated with the attribute value.
        if (key in self.children):
            # Recursively call the prediction for the child node.
            return self.children[key].predict(attributes)
        else:
            # If no child node found, return the default prediction.
            return self.default

class IDTree:
    """Class describing a decision tree."""

    def constructID(self, examples, targets, attributes):
        """
        Recursively construct an ID3 decision tree.

        Args:
            examples (numpy.ndarray): A numpy array of shape (number_of_examples, number_of_features) containing input examples.
            targets (numpy.ndarray): A numpy array of shape (number_of_examples) containing target labels.
            attributes (list): A list of attribute indices representing the available features.

        Returns:
            IDTNode: The root node of the constructed decision tree.
        """
        ## Handle base cases here
        # Check if all examples belong to the positive class
        if (targets.sum() == targets.size):
            # Return a leaf node with prediction of class 1
            return IDTNode(1)
        
        # Check if all examples belong to the positive class
        if (targets.sum() == 0):
            # Return a leaf node with prediction of class 0
            return IDTNode(0)
        
        # Check if there are no remaining attributes to split on.
        if (len(attributes) == 0):
            # Make a prediction based on majority class.
            if (targets.sum()*2 >= targets.size):
                return IDTNode(1)
            else:
                return IDTNode(0)
        
        ### remove below

        # Find the best attribute to split on based on information gain.
        best_attr = attributes[np.argmax(np.array([information_gain(examples, targets, attr) for attr in attributes]))]

        # Get unique values of the best attribute for splitting.
        possible_values = np.unique(examples[:, best_attr])
        
        # Create a dictionary to hold child nodes.
        children = {}

        # Remove the best attribute from the list of attributes to pass to child nodes.
        attributes_to_pass = attributes[:]
        attributes_to_pass.remove(best_attr)

        # Split the dataset based on the unique values of the best attribute.
        for i in range(possible_values.size):
            split_examples = examples[examples[:, best_attr] == possible_values[i]]
            split_targets = targets[examples[:, best_attr] == possible_values[i]]
            children[possible_values[i]] = self.constructID(split_examples, split_targets, attributes_to_pass)

        # Determine the default prediction for empty nodes.
        default_for_empty = 1 if targets.sum()*2 >= targets.size else 0

        # Return the root node of the current subtree.
        return IDTNode(default_for_empty, best_attr, children)

    def __init__(self, examples, targets, attributes):
        """
        Initialize an IDTree object.

        Args:
            examples (numpy.ndarray): A numpy array of shape (number_of_examples, number_of_features) containing input examples.
            targets (numpy.ndarray): A numpy array of shape (number_of_examples) containing target labels.
            attributes (list): A list of attribute indices representing the available features.

        This constructor builds the decision tree using the ID3 algorithm.
        """

        self.root = self.constructID(examples, targets, attributes)
    
    def predict(self, attributes):
        """
        Predict the outcome based on the input attributes.

        Args:
            attributes (list): A list of attribute values corresponding to the features of an example.

        Returns:
            The predicted outcome.
        """
        return self.root.predict(attributes)


def get_precision(model, input, target):
    """Given the model, data and targets, calculates precision."""
    correct_preds = 0
    for index in range(len(input)):
        correct_preds += (model.predict(input[index]) == target[index])
    precision = correct_preds / len(input)
    return precision


if __name__ == "__main__":
    ## Load all the relevant data
    
    # Training inputs
    cancer_train_inputs = np.loadtxt("data_q1/X_train.csv", delimiter=',', dtype=np.dtype(str), ndmin=2)[1:]

    # Training targets: we convert the string labels to integer
    cancer_train_targets = np.loadtxt("data_q1/Y_train.csv", delimiter=',', dtype=np.dtype(str), ndmin=2)[1:].squeeze(1).astype('int')

    # Test inputs
    cancer_test_inputs = np.loadtxt("data_q1/X_test.csv", delimiter=',', dtype=np.dtype(str), ndmin=2)[1:]

    # Tets targets: we convert the string labels to integer
    cancer_test_targets = np.loadtxt("data_q1/Y_test.csv", delimiter=',', dtype=np.dtype(str), ndmin=2)[1:].squeeze(1).astype('int')
    
    # Logging relevant statistics
    num_train_data = cancer_train_inputs.shape[0]
    print("Number of Training Data:", num_train_data)

    num_test_data = cancer_test_inputs.shape[0]
    print("Number of Test Data:", num_test_data)
    
    num_features = cancer_train_inputs.shape[1]
    print("Number of Features:", num_features)
    
    # Initialize the Descision Tree Classifier using Training Instances
    DTree = IDTree(cancer_train_inputs, cancer_train_targets, list(range(num_features)))
    
    # Print the precision for training and test instances
    print("Training precision:", get_precision(DTree, cancer_train_inputs, cancer_train_targets))
    print("Test precision:", get_precision(DTree, cancer_test_inputs, cancer_test_targets))
    