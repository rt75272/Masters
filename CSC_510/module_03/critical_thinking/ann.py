#-------------------------------------------------------------------------------------------------------
#
# CLEAN DATA
#
#
#-------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Mean squared error loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Neural Network class definition
class ANN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)  # Weights between input and hidden layer
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)  # Weights between hidden and output layer
        self.bias_hidden = np.random.randn(hidden_size)  # Bias for hidden layer
        self.bias_output = np.random.randn(output_size)  # Bias for output layer

    def forward(self, X):
        # Forward pass through the network
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.output_input  # No activation function for output layer (regression)
        return self.output

    def backward(self, X, y, learning_rate=0.01):
        # Backpropagation to update weights and biases
        # Compute the error at the output layer
        output_error = y - self.output
        output_delta = output_error  # No activation function in output layer, so no derivative is needed

        # Compute the error at the hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases for output layer
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate

        # Update weights and biases for input layer
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        # To track the loss over time
        loss_history = []
        
        # Training loop
        for epoch in range(epochs):
            self.forward(X)  # Perform forward pass
            self.backward(X, y, learning_rate)  # Perform backward pass
            
            # Store the loss value every epoch
            loss = mean_squared_error(y, self.output)
            loss_history.append(loss)
            
            if epoch % 100 == 0:  # Print the loss every 100 epochs
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

        return loss_history

# Define the input data (simple sequence data)
X = np.array([[1], [2], [3], [4], [5]])  # Input numbers (sequence)
y = np.array([[2], [3], [4], [5], [6]])  # Target numbers (next number in sequence)

# Define the network architecture (input_size=1, hidden_size=5, output_size=1)
input_size = 1
hidden_size = 5
output_size = 1

# Initialize the Artificial Neural Network
ann = ANN(input_size, hidden_size, output_size)

# Train the model
loss_history = ann.train(X, y, epochs=1000, learning_rate=0.01)

# Plotting the Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Testing the trained model
test_input = np.array([[6]])
predicted_output = ann.forward(test_input)
print(f"Predicted output for input {test_input[0][0]}: {predicted_output[0][0]:.4f}")

# Extending the input range for prediction
extended_X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# Get predictions for the extended input range
predictions = ann.forward(extended_X)

# Plotting the predictions vs actual target values
plt.figure(figsize=(10, 6))
plt.plot(X, y, label='Actual', color='blue', marker='o')
plt.plot(extended_X, predictions, label='Predicted', color='red', linestyle='dashed', marker='x')
plt.title("Predicted vs Actual Values (Extended Range)")
plt.xlabel("Input (X)")
plt.ylabel("Output (y)")
plt.legend()
plt.grid(True)
plt.show()








#-------------------------------------------------------------------------------------------------------
#
# MESSY DATA
#
#
#-------------------------------------------------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt

# # Define the activation function (ReLU) and its derivative
# def relu(x):
#     return np.maximum(0, x)

# def relu_derivative(x):
#     return np.where(x > 0, 1, 0)

# # Define the mean squared error loss function
# def mean_squared_error(y_true, y_pred):
#     return np.mean((y_true - y_pred) ** 2)

# # Neural Network Class
# class ANN:
#     def __init__(self, input_size, hidden_size, output_size):
#         self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
#         self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
#         self.bias_hidden = np.zeros(hidden_size)
#         self.bias_output = np.zeros(output_size)

#     def forward(self, X):
#         self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
#         self.hidden_output = relu(self.hidden_input)  # Apply ReLU to hidden layer
#         self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
#         self.output = self.output_input  # Linear output (no activation)
#         return self.output

#     def backward(self, X, y, learning_rate=0.001):  # Increased learning rate for faster convergence
#         output_error = y - self.output
#         output_delta = output_error  # Linear output so no additional activation derivative needed

#         hidden_error = output_delta.dot(self.weights_hidden_output.T)
#         hidden_delta = hidden_error * relu_derivative(self.hidden_output)

#         # Update weights and biases
#         self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
#         self.bias_output += np.sum(output_delta, axis=0) * learning_rate
#         self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
#         self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate

#     def train(self, X, y, epochs=10000, learning_rate=0.001, patience=100):
#         previous_loss = np.inf
#         loss_history = []

#         for epoch in range(epochs):
#             self.forward(X)
#             self.backward(X, y, learning_rate)

#             if epoch % 100 == 0:
#                 loss = mean_squared_error(y, self.output)
#                 loss_history.append(loss)
#                 print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

#                 if abs(previous_loss - loss) < 1e-6:
#                     print(f"Early stopping at epoch {epoch+1}")
#                     break
#                 previous_loss = loss

#         return loss_history

#     def denormalize(self, X, X_max):
#         return X * X_max

# # Generate noisy non-linear data (e.g., sine wave with added noise)
# X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)  # 100 points from 0 to 2*pi
# y = np.sin(X) + 0.2 * np.random.randn(*X.shape)  # Add some Gaussian noise to the sine wave

# # Prepare the data for time-series prediction: predict the next number in the sequence
# X_seq = X[:-1]  # Use all but the last value as inputs
# y_seq = y[1:]  # The next value is the target

# # Normalize the data
# X_max = np.max(X)
# y_max = np.max(np.abs(y))  # Use the max of absolute y-values for normalization
# X_seq = X_seq / X_max
# y_seq = y_seq / y_max

# # Define the network parameters
# input_size = 1
# hidden_size = 10  # Adjust hidden layer size for more complex data
# output_size = 1

# # Initialize the neural network
# ann = ANN(input_size, hidden_size, output_size)

# # Train the model and track the loss history
# loss_history = ann.train(X_seq, y_seq, epochs=100000, learning_rate=0.001)

# # Plot the Loss vs Epochs graph
# plt.figure(figsize=(10, 5))
# plt.plot(loss_history)
# plt.title('Loss vs Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss (MSE)')
# plt.grid(True)
# plt.show()

# # Test the trained model on the original data
# predictions = ann.forward(X_seq)

# # Denormalize the predictions
# predictions_denormalized = ann.denormalize(predictions, y_max)
# y_seq_denormalized = ann.denormalize(y_seq, y_max)

# # Plot Predicted vs Actual values
# plt.figure(figsize=(10, 5))
# plt.plot(X_seq * X_max, y_seq_denormalized, label="Actual Values", marker='o')
# plt.plot(X_seq * X_max, predictions_denormalized, label="Predicted Values", marker='x')
# plt.title('Predicted vs Actual Values')
# plt.xlabel('Input (X)')
# plt.ylabel('Output (y)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Predict the next value in the sequence (e.g., after input 2*pi)
# test_input = np.array([[2]])  # Test input for the next value in the sequence
# test_input = test_input / X_max  # Normalize the input

# predicted_output = ann.forward(test_input)
# predicted_output_denormalized = ann.denormalize(predicted_output, y_max)

# print(f"Predicted next value for input {2:.4f}: {predicted_output_denormalized[0][0]:.4f}")







#-------------------------------------------------------------------------------------------------------
#
# BACKUP
#
#-------------------------------------------------------------------------------------------------------
# import numpy as np

# # Define the activation function (ReLU) and its derivative
# def relu(x):
#     # ReLU (Rectified Linear Unit) activation function: returns max(0, x)
#     return np.maximum(0, x)

# def relu_derivative(x):
#     # Derivative of ReLU: 1 for positive values, 0 for non-positive values
#     return np.where(x > 0, 1, 0)

# # Define the mean squared error loss function
# def mean_squared_error(y_true, y_pred):
#     # Calculates the mean squared error (MSE) loss between true values and predicted values
#     return np.mean((y_true - y_pred) ** 2)

# # Neural Network Class
# class ANN:
#     def __init__(self, input_size, hidden_size, output_size):
#         # Initialize weights and biases with random values and zeros respectively
#         self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1  # Random weights for input to hidden layer
#         self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1  # Random weights for hidden to output layer
#         self.bias_hidden = np.zeros(hidden_size)  # Bias for the hidden layer initialized to zeros
#         self.bias_output = np.zeros(output_size)  # Bias for the output layer initialized to zeros

#     def forward(self, X):
#         # Perform the forward pass: calculates the output of the network
#         self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden  # Input to hidden layer
#         self.hidden_output = relu(self.hidden_input)  # Apply ReLU activation function to hidden layer output
#         self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output  # Input to output layer
#         self.output = self.output_input  # No activation function at the output layer (linear output)
#         return self.output

#     def backward(self, X, y, learning_rate=0.0001):  # Learning rate controls how fast we learn
#         # Perform the backward pass (gradient descent) to update weights and biases based on the error
#         output_error = y - self.output  # Calculate the error at the output layer (difference between true and predicted values)
#         output_delta = output_error  # Delta for output layer, proportional to the error

#         # Backpropagate the error to the hidden layer
#         hidden_error = output_delta.dot(self.weights_hidden_output.T)  # Error at hidden layer
#         hidden_delta = hidden_error * relu_derivative(self.hidden_output)  # Apply the derivative of ReLU to the hidden layer error

#         # Update weights and biases using the gradients calculated above
#         self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate  # Update hidden-output weights
#         self.bias_output += np.sum(output_delta, axis=0) * learning_rate  # Update output biases

#         self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate  # Update input-hidden weights
#         self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate  # Update hidden biases

#     def train(self, X, y, epochs=10000, learning_rate=0.0001, patience=100):
#         # Train the neural network using gradient descent
#         previous_loss = np.inf  # Initialize previous loss as infinity (for early stopping)
#         for epoch in range(epochs):
#             # Perform forward and backward passes for each epoch
#             self.forward(X)  # Forward pass to calculate predictions
#             self.backward(X, y, learning_rate)  # Backward pass to adjust weights and biases

#             if epoch % 100 == 0:  # Print loss every 100 epochs
#                 loss = mean_squared_error(y, self.output)  # Calculate the loss (MSE)
#                 print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

#                 # Early stopping if the loss does not improve significantly
#                 if abs(previous_loss - loss) < 1e-6:
#                     print(f"Early stopping at epoch {epoch+1}")  # Stop training if loss change is negligible
#                     break
#                 previous_loss = loss

#     def denormalize(self, X, X_max):
#         # Denormalize the output back to the original scale
#         return X * X_max

# # Sample Data (Input and Output)
# X = np.array([[1], [2], [3], [4], [5]])  # Sample input data
# y = np.array([[1], [2], [3], [4], [5]])  # Expected output is the same as input (identity mapping)

# # Normalize the data (scaling to the range [0, 1])
# X_max = np.max(X)  # Find the maximum value in the input data for normalization
# y_max = np.max(y)  # Find the maximum value in the output data for normalization
# X = X / X_max  # Normalize the input data
# y = y / y_max  # Normalize the output data

# # Define the network parameters
# input_size = 1  # One input feature
# hidden_size = 10  # Increased size of hidden layer to capture more complex patterns
# output_size = 1  # One output value

# # Initialize the neural network
# ann = ANN(input_size, hidden_size, output_size)

# # Train the model with specified parameters
# ann.train(X, y, epochs=150000, learning_rate=0.001)  # Reduced learning rate for finer convergence

# # Testing the trained model
# test_input = np.array([[6]])  # Test with input 6
# test_input = test_input / X_max  # Normalize the input using the same maximum value as during training

# predicted_output = ann.forward(test_input)  # Get the model's prediction

# # Denormalize the output to get the value on the original scale
# predicted_output_denormalized = ann.denormalize(predicted_output, y_max)

# # Print the final prediction for input 6
# print(f"Predicted output for input 6.0000: {predicted_output_denormalized[0][0]:.4f}")
