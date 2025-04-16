import numpy as np
import pandas as pd
from math import pi, exp, sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
"""
Naive Bayes Classifier implementation using Gaussian Naive Bayes for continuous data.

Usage:
    $ python naive_bayes_classifier.py
"""
class NaiveBayesClassifier:
    """Naive Bayes Classifier object."""
    def __init__(self):
        """Initializes the Naive Bayes Classifier to use the Gaussian model for continuous features."""
        self.priors = {}
        self.likelihoods = {}
        self.classes = []
        
    def fit(self, X, y):
        """Fits the Naive Bayes classifier to the training data by calculating
        the prior probabilities and likelihoods (mean and variance) for each class."""
        self.classes = np.unique(y)  # Get unique class labels.
        # Calculate prior probabilities P(class) for each class.
        for cls in self.classes:
            self.priors[cls] = np.sum(y == cls) / len(y)    
        # Calculate likelihoods P(feature|class) for each class.
        self.likelihoods = {}
        for cls in self.classes:
            X_class = X[y == cls] # Get features for the current class.
            self.likelihoods[cls] = []
            # Gaussian Naive Bayes. Calculate mean and variance for each feature.
            for feature in X_class.T:
                mean = np.mean(feature)
                var = np.var(feature)
                self.likelihoods[cls].append((mean, var))

    def predict(self, X):
        """Predict the class for each instance in the test data based on the
        calculated likelihoods and prior probabilities."""
        predictions = []
        for instance in X:
            class_probs = {} # Dictionary to store the probability for each class.
            for cls in self.classes:
                # Calculate the likelihood for each class.
                likelihood = self.calculate_likelihood(instance, cls)
                # Apply Bayes' Theorem. P(class|data) = P(data|class) * P(class).
                class_probs[cls] = likelihood * self.priors[cls]
            # Choose the class with the highest posterior probability.
            predicted_class = max(class_probs, key=class_probs.get)
            predictions.append(predicted_class)
        return np.array(predictions)

    def calculate_likelihood(self, instance, cls):
        """Calculate the likelihood for a given instance and class."""
        likelihood = 1
        for i, feature in enumerate(instance):
            mean, var = self.likelihoods[cls][i]
            probability = self.gaussian_probability(feature, mean, var)
            likelihood *= probability # Multiply all feature probabilities.
        return likelihood

    def gaussian_probability(self, x, mean, var):
        """Gaussian probability density function."""
        exponent = exp(-(x - mean) ** 2 / (2 * var)) 
        return (1 / sqrt(2 * pi * var)) * exponent

def main():
    """Main driver function."""
    UNIVERSAL_ANSWER = 42
    # Load the Iris dataset from sklearn.
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    # Split the dataset into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=UNIVERSAL_ANSWER)
    nb_classifier = NaiveBayesClassifier() # Instantiate the NaiveBayesClassifier with a Gaussian model.
    nb_classifier.fit(X_train.values, y_train.values) # Train the model using the training data.
    y_pred = nb_classifier.predict(X_test.values) # Make predictions on the test set.
    # Calculate and print the model's accuracy.
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    # Display predictions and actual class labels for the first 5 test samples.
    print("\n\n--------Predictions for the first 5 test samples--------")
    for i in range(5):
        print(f"\nSample {i+1}: \n\t Predicted Class: {y_pred[i]}\n\t Actual Class: {y_test.values[i]}")
    print("\n--------------------------------------------------------")

# Big red activation button.
if __name__ == "__main__":
    main()
