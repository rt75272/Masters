import numpy as np

# Sigmoid activation function and its derivative
# Sigmoid squashes the output between 0 and 1, often used in output layers for binary classification.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function used during backpropagation to compute gradients
def sigmoid_derivative(x):
    return x * (1 - x)

# ReLU (Rectified Linear Unit) activation function and its derivative
# ReLU is used to introduce non-linearity in the hidden layers, and helps avoid vanishing gradients.
def relu(x):
    return np.maximum(0, x)  # Applies ReLU by zeroing out negative values.

# Derivative of the ReLU function
# ReLU's derivative is 1 for positive values and 0 for non-positive values.
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Mean squared error (MSE) loss function
# Used to calculate the error between the predicted output and actual target output.
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)  # A simple way to calculate average squared error.

# Neural Network class definition
class simple_ann:
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier Initialization for better weight starting points, especially for deep networks.
        # This approach helps to scale weights depending on the size of input and hidden layer.
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2. / (input_size + hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2. / (hidden_size + output_size))
        
        # Biases initialized to zero; they shift the activation function, controlling the output.
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

    def forward(self, X):
        # Forward pass through the network:
        # First, calculate input to the hidden layer, add bias, and apply activation function (ReLU).
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = relu(self.hidden_input)  # Apply ReLU activation in the hidden layer
        
        # Calculate input to the output layer, add bias, and apply sigmoid activation for binary output.
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)  # Output is squashed between 0 and 1 for binary classification
        return self.output

    def backward(self, X, y, learning_rate=0.1):
        # Backpropagation step to adjust weights and biases:
        
        # Compute the error between the actual and predicted output at the output layer.
        output_error = y - self.output  # Difference between actual and predicted values
        output_delta = output_error * sigmoid_derivative(self.output)  # Apply sigmoid derivative for gradient calculation

        # Compute error at the hidden layer based on the output layer's error.
        hidden_error = output_delta.dot(self.weights_hidden_output.T)  # Backpropagate the error
        hidden_delta = hidden_error * relu_derivative(self.hidden_output)  # Apply ReLU derivative for gradient calculation

        # Update weights and biases for the output layer using the calculated gradients.
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate

        # Update weights and biases for the input layer using the gradients from the hidden layer.
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate

    def train(self, X, y, epochs=10000, learning_rate=0.1):
        # Training loop:
        for epoch in range(epochs):
            self.forward(X)  # Perform forward pass (calculating predicted output)
            self.backward(X, y, learning_rate)  # Perform backpropagation to update weights and biases
            
            # Print the loss (error) every 1000 epochs for monitoring progress
            if epoch % 1000 == 0:
                loss = mean_squared_error(y, self.output)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

# Define the input and target data (for binary regression)
# Inputs: A simple set of 0 and 1 values, representing a binary classification task.
X = np.array([[0], [1]])  # Input values: [0], [1]
y = np.array([[0], [1]])  # Expected output: [0], [1] (i.e., output should match input)

# Define the network architecture:
# input_size: 1 (only one feature),
# hidden_size: 5 (a larger hidden layer to capture more complex patterns),
# output_size: 1 (binary output).
input_size = 1
hidden_size = 5
output_size = 1

# Initialize the neural network with the above architecture
ann = simple_ann(input_size, hidden_size, output_size)

# Train the model with the given input data for 10,000 epochs and a learning rate of 0.1
ann.train(X, y, epochs=10000, learning_rate=0.1)

# Testing the trained model on both input values (0 and 1)
test_input = np.array([[0], [1]])  # Test with input values [0] and [1]
predicted_output = ann.forward(test_input)  # Perform forward pass to get predictions

# Print the predicted output for each test input
print(f"Predicted output for input {test_input[0][0]}: {predicted_output[0][0]:.4f}")
print(f"Predicted output for input {test_input[1][0]}: {predicted_output[1][0]:.4f}")
