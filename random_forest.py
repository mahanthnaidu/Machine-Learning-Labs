import time
import concurrent.futures
import numpy as np
from decision_tree import IDTree, get_precision
from sklearn.datasets import make_gaussian_quantiles
from multiprocessing import Process
import multiprocessing
import numpy as np
from functools import partial

class RandomForest:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.tree_estimators = []

    def fit(self, examples, targets, attributes, num_attribute_sample):
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            partial_fit = partial(self._fit_tree, examples, targets, attributes, num_attribute_sample)
            self.tree_estimators = list(pool.map(partial_fit, range(self.n_estimators)))
        return self

    def _fit_tree(self, examples, targets, attributes, num_attribute_sample,_):
        # TODO: Randomly select data points with replacement (bootstrapping)
        sample_indices = np.random.choice(range(len(examples)),len(examples),replace=True)
        sample_examples = [examples[i] for i in sample_indices]
        sample_targets = [targets[i] for i in sample_indices]
        # TODO: Randomly sample a subset of attributes
        sample_attributes = np.random.choice(attributes,num_attribute_sample,replace=False)
        # TODO: Extract the sampled data
        # TODO: Create and train a decision tree on the sampled data with a call to IDTree
        model = IDTree(np.array(sample_examples),np.array(sample_targets),list(sample_attributes))
        return model

    def predict(self, attributes):
        # TODO: Make predictions using each tree in the ensemble
        # TODO: return majority vote as the final prediction
        predictions = [tree.predict(attributes) for tree in self.tree_estimators]
        final_prediction = np.mean(predictions,axis=0)
        final_prediction = np.where(final_prediction>=0.5,1,0)
        return final_prediction
    
def main():
    # Load your training and test data
    # Training inputs
    cancer_train_inputs = np.loadtxt("data_q2/X_train.csv", delimiter=',', dtype=np.dtype(str), ndmin=2)[1:]

    # Training targets: we convert the string labels to integer
    cancer_train_targets = np.loadtxt("data_q2/Y_train.csv", delimiter=',', dtype=np.dtype(str), ndmin=2)[1:].squeeze(1).astype('int')

    # Test inputs
    cancer_test_inputs = np.loadtxt("data_q2/X_test.csv", delimiter=',', dtype=np.dtype(str), ndmin=2)[1:]

    # Tets targets: we convert the string labels to integer
    cancer_test_targets = np.loadtxt("data_q2/Y_test.csv", delimiter=',', dtype=np.dtype(str), ndmin=2)[1:].squeeze(1).astype('int')
    
    # Logging relevant statistics
    num_train_data = cancer_train_inputs.shape[0]
    print("Number of Training Instances:", num_train_data)

    num_test_data = cancer_test_inputs.shape[0]
    print("Number of Test Instances:", num_test_data)
    
    num_features = cancer_train_inputs.shape[1]
    print("Number of Features:", num_features)

    # Seed
    np.random.seed(0)

    # Number of trees in the random forest
    n_estimators = 50

    # Choose the number of attributes to sample
    num_attribute_sample = 7
    assert(num_attribute_sample <= num_features)
    print("Number of Features Sampled:", num_attribute_sample)
    
    # Create and train the Random Forest
    rf_classifier = RandomForest(n_estimators=n_estimators)
    
    start_time = time.time()
    rf_classifier.fit(cancer_train_inputs, cancer_train_targets, list(range(num_features)), num_attribute_sample=num_attribute_sample)
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")

    #print(rf_classifier.tree_estimators)
    
    print("Random Forest Training Precision:", get_precision(rf_classifier, cancer_train_inputs, cancer_train_targets))
    print("Random Forest Test Precision:", get_precision(rf_classifier, cancer_test_inputs, cancer_test_targets))

if __name__ == "__main__":
    main()
