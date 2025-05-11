import numpy as np
import pandas as panda # type: ignore
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import PolynomialFeatures # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore
# ------------------------------------------------------------------------------------------------------------
# Polynomial Regression Model.
#
# Implements a polynomial regression model to predict an employee's salary based on their years of experience.
#
# Usage:
#   $ python polynomial.py
# ------------------------------------------------------------------------------------------------------------
DEGREE = 5 # Degree of the polynomial regression model.

def load_data():
    """Loads and returns the dataset."""
    data = panda.read_csv('Salary_Data.csv') # Read the dataset from the CSV file.
    return data

def preprocess_data(data):
    """Prepares the features and target variables, and splits the data into training and testing sets."""
    # Separate features and the target.
    X = data[['YearsExperience']].values
    y = data['Salary'].values
    # Split the dataset into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def train_polynomial_regression(X_train, y_train):
    """Trains the polynomial regression model and returns the trained model and polynomial transformer."""
    # Create polynomial features.
    poly = PolynomialFeatures(degree=DEGREE)
    X_poly_train = poly.fit_transform(X_train)
    # Train the polynomial regression model.
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    return model, poly

def printer(train_rmse, test_rmse, train_r2, test_r2):
    """Displays the evaluation results in the terminal."""
    print("\n\n\t===================================")
    print("\tEvaluation Results")
    print("\t===================================")
    print(f"\tPolynomial Degree: {DEGREE}")
    print(f"\tTrain RMSE: {train_rmse:.2f}")
    print(f"\tTest RMSE: {test_rmse:.2f}")
    print(f"\tTrain R²: {train_r2:.2f}")
    print(f"\tTest R²: {test_r2:.2f}")
    print("\t===================================\n")

def evaluate_model(model, poly, X_train, X_test, y_train, y_test):
    """Evaluates the model and displays the RMSE and R² scores."""
    # Transform the features using the polynomial transformer.
    X_poly_train = poly.transform(X_train)
    X_poly_test = poly.transform(X_test)
    # Make predictions.
    y_train_pred = model.predict(X_poly_train)
    y_test_pred = model.predict(X_poly_test)
    # Calculate evaluation stats.
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    printer(train_rmse, test_rmse, train_r2, test_r2) # Display the evaluation stats.
    return train_rmse, test_rmse, train_r2, test_r2

def visualize_results(model, poly, X, y):
    """Visualizes the actual data and the polynomial regression curve."""
    plt.scatter(X, y, color='blue', label='Actual Data') # Scatter plot of actual data.
    # Generate a range of values for plotting the regression curve.
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_range_pred = model.predict(X_range_poly)
    plt.plot(X_range, y_range_pred, color='cyan', label=f'Polynomial Regression (degree={DEGREE})')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.title('Polynomial Regression')
    plt.legend()
    plt.show()

def predict_salary(model, poly, years_of_experience=10.5):
    """Predicts the salary for a given number of years of experience."""
    # Transform the input years of experience into polynomial features.
    years_of_experience_poly = poly.transform([[years_of_experience]])
    # Predict the salary using the trained model.
    predicted_salary = model.predict(years_of_experience_poly)[0]
    return predicted_salary

def main():
    """Main driver function to load data, train the model, evaluate it, and visualize the results."""
    data = load_data() # Load the dataset.
    X_train, X_test, y_train, y_test = preprocess_data(data) # Preprocess the data.
    model, poly = train_polynomial_regression(X_train, y_train) # Train the model.
    try: # Gather user input for prediction.
        working_years = float(input("\nEnter the number of years of experience: "))
        predicted_salary = predict_salary(model, poly, working_years)
        print(f"Predicted Salary for {working_years} years of experience: ${predicted_salary:,.2f}")
    except ValueError:
        print("Invalid input. Please enter a numeric value for years of experience.")
    evaluate_model(model, poly, X_train, X_test, y_train, y_test) # Evaluate the model.
    visualize_results(model, poly, data[['YearsExperience']].values, data['Salary'].values) # Graphical representation.

# The big red activation button.
if __name__ == "__main__":
    main()