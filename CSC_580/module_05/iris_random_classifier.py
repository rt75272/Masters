import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.tree import plot_tree
# -------------------------------------------------------------------------------------
# Iris Random Forest Classifier
# 
# Loads the Iris dataset, splits it into training and test sets, trains a Random Forest 
# classifier, makes predictions, and prints results including a confusion matrix and 
# feature importances.
#
# Usage:
#   python iris_random_classifier.py
# -------------------------------------------------------------------------------------
def load_data(seed: int = 0):
    """Loads the Iris dataset and returns a DataFrame and the original dataset 
    object."""
    np.random.seed(seed)
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris

def split_data(df: pd.DataFrame, train_frac: float = 0.75):
    """Splits the DataFrame into training and test sets."""
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= train_frac
    train = df[df['is_train'] == True]
    test = df[df['is_train'] == False]
    return train, test

def preprocess_data(train: pd.DataFrame, features):
    """Factorizes the species column for model training."""
    y = pd.factorize(train['species'])[0]
    return y

def train_classifier(train: pd.DataFrame, features, y):
    """Trains a Random Forest classifier."""
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf.fit(train[features], y)
    return clf

def predict(clf: RandomForestClassifier, test: pd.DataFrame, features):
    """Makes predictions on the test set."""
    preds = clf.predict(test[features])
    return preds

def plot_confusion_matrix(test, preds, iris):
    """Plots a confusion matrix as a heatmap."""
    conf_matrix = pd.crosstab(
        test['species'],
        pd.Categorical.from_codes(preds, iris.target_names),
        rownames=['Actual Species'],
        colnames=['Predicted Species'])
    plt.figure(figsize=(6, 4))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Species")
    plt.ylabel("Actual Species")
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.xticks(ticks=range(len(iris.target_names)), labels=iris.target_names)
    plt.yticks(ticks=range(len(iris.target_names)), labels=iris.target_names)
    for i in range(len(iris.target_names)):
        for j in range(len(iris.target_names)):
            plt.text(j, i, 
                conf_matrix.values[i, j], 
                ha='center', va='center', color='black')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def plot_random_forest_classifier(clf, features, class_names):
    """Plots the first tree of the trained Random Forest classifier."""
    plt.figure(figsize=(20, 10))
    plot_tree(
        clf.estimators_[0],
        feature_names=features,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10)
    plt.title("First Decision Tree in the Random Forest")
    plt.show()

def print_results(train, test, preds, iris, features, clf):
    """Prints the results: dataset sizes, predictions, confusion matrix, and 
    feature importances."""
    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))
    print("\nPredicted species (first 10):", preds[:10])
    print("\nActual species:", list(test['species'].head()))
    print("Predicted species:",
        list(pd.Categorical.from_codes(preds, iris.target_names)[:5]))
    conf_matrix = pd.crosstab(
        test['species'],
        pd.Categorical.from_codes(preds, iris.target_names),
        rownames=['Actual Species'],
        colnames=['Predicted Species'])
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nFeature importances:")
    print(list(zip(features, clf.feature_importances_)))
    plot_confusion_matrix(test, preds, iris)
    plot_random_forest_classifier(clf, features, iris.target_names)

def main():
    """Main driver function to execute the workflow of loading data, 
    training the model, and making predictions."""
    df, iris = load_data()
    train, test = split_data(df)
    features = df.columns[:4]
    y = preprocess_data(train, features)
    clf = train_classifier(train, features, y)
    preds = predict(clf, test, features)
    print_results(train, test, preds, iris, features, clf)

# The big red activation button.
if __name__ == "__main__":
    main()
