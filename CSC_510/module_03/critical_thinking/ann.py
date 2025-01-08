import numpy as np
import matplotlib.pyplot as plt
###########################################################################################################
# 2-layer Artificial Neural Network.
# 
# Uses a 2-layer ANN implementation in order to predict the next number in a series.
# 
# Usage:
#     $ python ann.py
###########################################################################################################

"""
Sigmoid function to introduce non-linearity in the network.

It outputs values between 0 and 1.
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

"""
Derivative of the sigmoid function, used during backpropagation.

The derivative is computed as x * (1 - x) where x is the output of the sigmoid function.
"""
def sigmoid_derivative(x):
    return x * (1 - x)

"""
Mean Squared Error (MSE) loss function used to evaluate the performance of the model.

It calculates the average squared difference between actual and predicted values.
"""
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

"""
Initialize the neural network with random weights and biases.

input_size: Number of input features.
hidden_size: Number of neurons in the hidden layer.
output_size: Number of output neurons (1 for regression).
"""
class ANN:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)  # Weights between input and hidden layer.
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)  # Weights between hidden and output layer.
        self.bias_hidden = np.random.randn(hidden_size)  # Bias for hidden layer.
        self.bias_output = np.random.randn(output_size)  # Bias for output layer.

    """
    Forward pass through the network.

    Computes the hidden layer's output and the final output.
    """
    def forward(self, X):
        # Input to the hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)  # Apply sigmoid activation function to hidden layer.
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.output_input  # No activation function for output layer (regression).
        return self.output

    """
    Backpropagation to update weights and biases based on the error.

    The error is computed for the output layer, then propagated back to the hidden layer.
    """
    def backward(self, X, y, learning_rate=0.01):
        # Calculate the error at the output layer.
        output_error = y - self.output
        output_delta = output_error  # No activation function in output layer, so no derivative is needed.
        # Calculate the error at the hidden layer.
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)  # Derivative of sigmoid for hidden layer.
        # Update weights and biases for output layer using gradient descent.
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        # Update weights and biases for input layer using gradient descent.
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate

    """
    Train the network using the given data.
    
    Loops through the training data for a set number of epochs.
    """
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        loss_history = []  # List to store the loss at each epoch.
        # Training loop.
        for epoch in range(epochs):
            self.forward(X)  # Perform forward pass.
            self.backward(X, y, learning_rate)  # Perform backward pass to adjust weights.
            # Calculate and store the loss value at each epoch.
            loss = mean_squared_error(y, self.output)
            loss_history.append(loss)
            # Print the loss every 100 epochs.
            if epoch % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        return loss_history

# Define the input data (simple sequence data).
X = np.array([[1], [2], [3], [4], [5]])  # Input numbers (sequence).
y = np.array([[2], [3], [4], [5], [6]])  # Target numbers (next number in sequence).

# Define the network architecture (input_size=1, hidden_size=5, output_size=1).
input_size = 1
hidden_size = 5
output_size = 1

# Initialize the Artificial Neural Network.
ann = ANN(input_size, hidden_size, output_size)

# Train the model and store the loss history.
loss_history = ann.train(X, y, epochs=1000, learning_rate=0.01)

# Plotting the Loss Curve to visualize the training progress.
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Testing the trained model with new data (e.g., input 6).
test_input = np.array([[6]])
predicted_output = ann.forward(test_input)  # Get the model's prediction for input 6.
print(f"Predicted output for input {test_input[0][0]}: {predicted_output[0][0]:.4f}")

# Extending the input range for prediction (testing on more values).
extended_X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
predictions = ann.forward(extended_X)  # Get predictions for the extended range.

# Plotting the predictions vs actual target values.
plt.figure(figsize=(10, 6))
plt.plot(X, y, label='Actual', color='blue', marker='o')
plt.plot(extended_X, predictions, label='Predicted', color='red', linestyle='dashed', marker='x')
plt.title("Predicted vs Actual Values (Extended Range)")
plt.xlabel("Input (X)")
plt.ylabel("Output (y)")
plt.legend()
plt.grid(True)
plt.show()
