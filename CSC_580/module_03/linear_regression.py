import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
import matplotlib.pyplot as plt # type: ignore
# --------------------------------------------------------------------------------------
# TensorFlow & Keras Linear Regression.
#
# Demonstrates a linear regression model using TensorFlow and Keras. Predicts a 
# linear relationship between two variables.
#
# Usage:
#    $ python linear_regression.py
# --------------------------------------------------------------------------------------

def generate_data(seed=101, num_points=50):
    """Generates synthetic linear data with noise and scales it."""
    np.random.seed(seed)
    x = np.linspace(0, 50, num_points)
    y = np.linspace(0, 50, num_points)
    # Add noise to both x and y.
    x += np.random.uniform(-4, 4, num_points)
    y += np.random.uniform(-4, 4, num_points)
    # Scale data.
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    return x, y

def plot_data(x, y, title="Training Data"):
    """Plots the training data as a scatter plot."""
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def build_model():
    """Builds a TensorFlow linear regression model."""
    model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),loss='mse')
    return model

def train_model(model, x, y, epochs=1000):
    """Trains the linear regression model using Tensorflow and Keras."""
    # Reshape both x and y.
    x_train = x.reshape(-1, 1)
    y_train = y.reshape(-1, 1)
    print("\nTraining...")
    print("Please hold...")
    history = model.fit(x_train, y_train, epochs=epochs, verbose=0)
    # Print progress every 100 epochs.
    for epoch in range(0, epochs, 100):
        loss = history.history['loss'][epoch]
        weights = model.layers[0].get_weights()
        W = weights[0][0][0]
        b = weights[1][0]
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {loss:.6f} | "
            f"Weight (W): {W:.6f} | "
            f"Bias (b): {b:.6f}")
    final_W = model.layers[0].get_weights()[0][0][0]
    final_b = model.layers[0].get_weights()[1][0]
    final_loss = history.history['loss'][-1]
    print("\nTraining finished!")
    print(f"Final Loss: {final_loss:.6f}")
    print(f"Final Weight (W): {final_W:.6f}")
    print(f"Final Bias (b): {final_b:.6f}")
    return final_W, final_b, final_loss

def plot_fitted_line(x, y, W, b):
    """Plots the original data and the fitted regression line."""
    y_fit = W * x + b
    plt.plot(x, y_fit, color='red', label='Fitted Line')
    plt.scatter(x, y, label='Training Data')
    plt.title("Linear Regression Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def main():
    """Main driver function to run the linear regression workflow."""
    x, y = generate_data()
    plot_data(x, y)
    model = build_model()
    final_W, final_b, final_loss = train_model(model, x, y)
    print("\nTraining completed.")
    print(f"Final training loss: {final_loss:.4f}")
    print(f"Learned weight: {final_W:.4f}")
    print(f"Learned bias: {final_b:.4f}")
    plot_fitted_line(x, y, final_W, final_b)

# The big red activation button.
if __name__ == "__main__":
    main()
