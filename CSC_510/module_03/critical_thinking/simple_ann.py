import numpy as np
###########################################################################################################
# 2-layer Artificial Neural Network.
# 
# Uses a 2-layer ANN implementation in order to predict the next number in a series.
# 
# Usage:
#     $ python ann.py
###########################################################################################################

"""
The sigmoid function is used as an activation function.

It maps input values to a range between 0 and 1, and is commonly used for binary classification tasks.
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

"""
The derivative of the sigmoid function is used during backpropagation to update the weights and biases.

The derivative is calculated as: sigmoid(x) * (1 - sigmoid(x)).
"""
def sigmoid_derivative(x):
    return x * (1 - x)

"""
The MSE loss function calculates the average squared difference between the true values (y_true)
and the predicted values (y_pred). 

It's commonly used for regression tasks.
"""
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

"""
Initializes the artificial neural network with random weights and biases.

Parameters:
- input_size: The number of input features.
- hidden_size: The number of neurons in the hidden layer.
- output_size: The number of output neurons.
"""
class ANN:
    def __init__(self, input_size, hidden_size, output_size):
        # Randomly initialize weights between input and hidden layer.
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        # Randomly initialize weights between hidden and output layer.
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        # Randomly initialize biases for the hidden layer.
        self.bias_hidden = np.random.randn(hidden_size)
        # Randomly initialize biases for the output layer.
        self.bias_output = np.random.randn(output_size)

    """
    The feedforward step of the neural network where input data passes through the network layers.

    Parameters:
    - X: Input data matrix.
    
    Returns:
    - The output of the network after passing through the hidden and output layers.
    """
    def feedforward(self, X):
        # Hidden layer calculations.
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)  # Apply the sigmoid activation function.
        # Output layer calculations.
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)  # Apply the sigmoid activation function.
        return self.output

    """
    The backpropagation step of the neural network where the gradients are calculated 
    and the weights/biases are updated.
    
    Parameters:
    - X: Input data.
    - y: Actual output (desired output).
    - learning_rate: Learning rate used for gradient descent.
    """
    def backpropagate(self, X, y, learning_rate):
        # Calculate the error at the output layer (difference between actual and predicted output).
        output_error = y - self.output
        # Calculate the delta for the output layer (error term).
        output_delta = output_error * sigmoid_derivative(self.output)  # Derivative of sigmoid used for the update.
        # Backpropagate the error to the hidden layer.
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)  # Derivative of sigmoid for hidden layer.
        # Update weights and biases using gradient descent.
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        # Fixing bias update to match the shape of the output layer.
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate  # Update the bias for output layer.
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        # Fixing bias update to match the shape of the hidden layer.
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate  # Update the bias for hidden layer.

    """
    Train the neural network using the provided training data.
    
    Parameters:
    - X: Input data.
    - y: Target (desired output).
    - epochs: Number of iterations to train the model.
    - learning_rate: The rate at which the model learns during training.
    """
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Feedforward step to calculate predicted output.
            self.feedforward(X)
            # Backpropagate step to adjust weights and biases.
            self.backpropagate(X, y, learning_rate)
            # Print loss every 100 epochs.
            if epoch % 100 == 0:
                loss = mse_loss(y, self.output)  # Calculate the mean squared error (MSE) loss.
                print(f'Epoch {epoch}, Loss: {loss}')  # Print the loss for the current epoch.

# Example data (X: input data, y: expected output).
X = np.array([[0], [1], [2], [3]])  # Simple input data (a series of numbers: 0, 1, 2, 3).
y = np.array([[1], [2], [3], [4]])  # Expected output: next number in the series (1, 2, 3, 4).

# Parameters for the neural network.
input_size = X.shape[1]  # Number of input features (in this case, just 1 feature).
hidden_size = 5         # Number of neurons in the hidden layer (arbitrary choice).
output_size = 1         # Output size (just 1 output, the next number in the sequence).
epochs = 1000           # Number of training iterations.
learning_rate = 0.1     # Learning rate for gradient descent.

# Initialize and train the neural network.
ann = ANN(input_size, hidden_size, output_size)
ann.train(X, y, epochs, learning_rate)

# After training, test the neural network.
test_input = np.array([[4]])  # Test with input 4 (next number in the series).
predicted_output = ann.feedforward(test_input)  # Get the network's prediction.
print(f"Predicted output for input {test_input}: {predicted_output}")
