import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.tree import plot_tree # type: ignore
from sklearn.datasets import load_iris # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
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
def load_data(seed: int = None):
    """Loads the Iris dataset and returns a DataFrame and the original dataset 
    object."""
    # Sets random seed for reproducible results if provided.
    if seed is not None:
        np.random.seed(seed)
    # Loads the famous Iris dataset from sklearn.
    iris = load_iris()
    # Creates a DataFrame with feature columns using descriptive names.
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # Adds species names as categorical data for easier interpretation.
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris

def split_data(df: pd.DataFrame, train_frac: float = 0.75):
    """Splits the DataFrame into training and test sets."""
    # Uses uniform distribution to randomly assign each row.
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= train_frac
    # Splits data based on the random assignment.
    train = df[df['is_train'] == True]
    test = df[df['is_train'] == False]
    return train, test

def preprocess_data(train: pd.DataFrame, features):
    """Factorizes the species column for model training."""
    y = pd.factorize(train['species'])[0]
    return y

def train_classifier(train: pd.DataFrame, features, y):
    """Trains a Random Forest classifier."""
    # Initializes Random Forest with 2 parallel jobs for faster training.
    # No fixed random_state allows for varied results across runs.
    clf = RandomForestClassifier(n_jobs=2)
    # Trains the model on feature data (X) and target labels (y).
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
    # Set up the plot.
    plt.figure(figsize=(6, 4))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Species")
    plt.ylabel("Actual Species")
    # Displays matrix as heatmap with blue color scheme.
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    # Sets axis labels to show species names instead of numbers.
    plt.xticks(ticks=range(len(iris.target_names)), labels=iris.target_names)
    plt.yticks(ticks=range(len(iris.target_names)), labels=iris.target_names)
    # Adds text annotations showing the actual count values.
    for i in range(len(iris.target_names)):
        for j in range(len(iris.target_names)):
            plt.text(j, i, 
                conf_matrix.values[i, j], 
                ha='center', va='center', color='black')
    # Adds colorbar and finalize layout.
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def plot_random_forest_classifier(clf, features, class_names):
    """Plots the first tree of the trained Random Forest classifier."""
    # Creates large figure to accommodate decision tree visualization.
    plt.figure(figsize=(20, 10))
    plot_tree(
        clf.estimators_[0], # First tree in the forest.
        feature_names=features, # Uses descriptive feature names.
        class_names=class_names, # Shows species names, not numbers.
        filled=True, # Fills nodes with colors.
        rounded=True, # Rounded node corners.
        fontsize=10) # Readable font size.
    plt.title("Decision Tree in the Random Forest")
    plt.show()

def print_results(train, test, preds, iris, features, clf):
    """Prints the results: dataset sizes, predictions, confusion matrix, and feature 
    importances."""
    # Displays basic dataset information.
    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))
    # Shows sample of the actual data structure.
    print("\nTraining Data (first 5 rows):")
    print(train.head())
    print("\nTest Data (first 5 rows):")
    print(test.head())
    # Gets random sample of predictions for display.
    random_indices = np.random.choice(len(preds), size=min(10, len(preds)), 
                                      replace=False)
    random_preds = preds[random_indices]
    random_actual = test.iloc[random_indices]['species'].tolist()
    random_predicted_names = list(pd.Categorical.from_codes(random_preds, 
                                                            iris.target_names))
    # Shows both numeric codes and species names for predictions.
    print(f"\nRandom sample of predictions ({len(random_preds)}):", random_preds)
    print("\nActual species (random sample):", random_actual)
    print("\nPredicted species (random sample):", random_predicted_names)
    # Creates and display confusion matrix.
    conf_matrix = pd.crosstab(
        test['species'],
        pd.Categorical.from_codes(preds, iris.target_names),
        rownames=['Actual Species'],
        colnames=['Predicted Species'])
    print("\n\nConfusion Matrix:\n", conf_matrix)
    print("\n\nFeature Importances:")
    print("-" * 40)
    for feature, importance in zip(features, clf.feature_importances_):
        print(f"{feature:<30} {importance*100:>6.2f}%")
    # Generates visualization plots.
    plot_confusion_matrix(test, preds, iris)
    plot_random_forest_classifier(clf, features, iris.target_names)

def main():
    """Main driver function to execute the workflow of loading data, training the model,
    and making predictions."""
    # Loads and prepares the Iris dataset.
    df, iris = load_data()
    # Splits data into training and testing sets.
    train, test = split_data(df)
    # Extracts feature columns (first 4 columns are measurements).
    features = df.columns[:4]
    # Converts species names to numeric labels for training.
    y = preprocess_data(train, features)
    # Trains the Random Forest classifier.
    clf = train_classifier(train, features, y)
    # Makes predictions on test data.
    preds = predict(clf, test, features)
    # Displays comprehensive results and visualizations.
    print_results(train, test, preds, iris, features, clf)

# The big red activation button.
if __name__ == "__main__":
    main()
