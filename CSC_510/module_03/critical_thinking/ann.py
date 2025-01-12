import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar for training loop.
###########################################################################################################
# 2 Layer Artificial Neural Network.
# 
# Uses a 2-layer ANN implementation in order to predict the next number in a series. Can work with integers
# or floats as user input. 
# 
# Usage:
#   $ python ann.py [--number n]
#   ex.  
#       $ python ann.py --number 8
###########################################################################################################
def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function that allows small negative values instead of zero.
    This prevents the 'dead neuron' problem that can occur with traditional ReLU.
    
    Parameters:
        x (numpy.ndarray): Input values to the activation function.
        alpha (float): The slope for negative values (default is 0.01).
    
    Returns:
        numpy.ndarray: Output after applying the Leaky ReLU function.
    """
    positive_part = x * (x > 0)  # Keep x when x > 0, else 0.
    negative_part = alpha * x * (x <= 0)  # Keep alpha * x when x <= 0, else 0.
    both_parts = positive_part + negative_part
    return both_parts

def leaky_relu_derivative(x, alpha=0.01):
    """Derivative of the Leaky ReLU function, used during backpropagation to update weights.
    
    Parameters:
        x (numpy.ndarray): Output of the Leaky ReLU activation function.
        alpha (float): The slope for negative values (default is 0.01).
    
    Returns:
        numpy.ndarray: The derivative of the Leaky ReLU function.
    """
    # If x > 0, derivative is 1; otherwise, it's alpha.
    positive_part = (x > 0).astype(float)  # 1 when x > 0, otherwise 0.
    negative_part = (x <= 0).astype(float) * alpha  # alpha when x <= 0, otherwise 0.
    final_result = positive_part + negative_part
    return final_result

def mean_squared_error(y_true, y_pred):
    """Calculate the Mean Squared Error (MSE) between the actual and predicted values.
    MSE is commonly used for regression problems.
    
    Parameters:
        y_true (numpy.ndarray): Actual target values.
        y_pred (numpy.ndarray): Predicted values from the model.
    
    Returns:
        float: The mean squared error between the true and predicted values.
    """
    difference = y_true - y_pred # Calculate the difference between true values and predicted values.
    squared_difference = difference ** 2 # Square the differences.
    mse = np.mean(squared_difference) # Compute the mean of the squared differences.
    return mse

class ANN:
    """Artificial Neural Network.
    
    An Artificial Neural Network (ANN) is a computational model inspired by the way 
    biological neural networks in the brain work. ANNs are used to model and solve complex 
    problems such as classification, regression, pattern recognition, and prediction.   

    An ANN mimics how the human brain processes information, allowing it to perform complex 
    tasks such as prediction, classification, and regression. The network "learns" from data, 
    adjusts its weights and biases during training, and can be used to make predictions on 
    new, unseen data. 
    """
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the artificial neural network with random weights and biases.
        
        Parameters:
            input_size (int): Number of input features (e.g., 1 for single input).
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output neurons (1 for regression).
        """
        # Random initialization of weights with small values to avoid large gradients initially.
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        # Bias initialization (starting with zeros).
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

    def forward(self, X):
        """Perform a forward pass through the network.
        
        This computes the activations of the hidden layer and output layer.
        
        Parameters:
            X (numpy.ndarray): Input data for the network (e.g., training inputs).
        
        Returns:
            numpy.ndarray: Predicted output from the network.
        """
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden # Hidden layer computation.
        self.hidden_output = leaky_relu(self.hidden_input) # Apply Leaky ReLU activation to the hidden layer.
        # Output layer computation (linear output, no activation function).
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.output_input  # No activation on output layer for regression.
        return self.output

    def backward(self, X, y, learning_rate=0.01):
        """Perform backpropagation to update the weights and biases based on the error.
        
        Parameters:
            X (numpy.ndarray): Input data used in forward pass.
            y (numpy.ndarray): Actual target values.
            learning_rate (float): The learning rate to control how much the weights change.
        """
        output_error = y - self.output # Compute the error.
        output_delta = output_error # Gradient for output layer.
        hidden_error = output_delta.dot(self.weights_hidden_output.T) # Backpropagate the error to the hidden layer.
        hidden_delta = hidden_error * leaky_relu_derivative(self.hidden_output) # Compute the gradient for the hidden layer.
        # Update the weights and biases using the gradients and the learning rate.
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """Train the neural network using the given training data.
        
        Parameters:
            X (numpy.ndarray): Training data (inputs).
            y (numpy.ndarray): Target values (outputs).
            epochs (int): Number of training iterations.
            learning_rate (float): Rate at which weights are updated during training.
        
        Returns:
            list: List of loss values for each epoch, to monitor training progress.
        """
        loss_history = []
        # Training loop with progress bar.
        for epoch in tqdm(range(epochs), desc="Training Epochs", ncols=100, dynamic_ncols=True, position=0):
            self.forward(X)   # Perform a forward pass.
            self.backward(X, y, learning_rate)  # Perform a backward pass (update weights).
            # Calculate and store the loss for this epoch.
            loss = mean_squared_error(y, self.output)
            loss_history.append(loss)
            # Optionally print the loss every 200 epochs (can be commented out if not needed).
            if epoch % 200 == 0:
                sys.stdout.flush()  # Ensure the terminal output is immediately visible.
        return loss_history

def main():
    """Main driver function.
    """
    # Command-line argument parser.
    parser = argparse.ArgumentParser(description="Predict the next number in the sequence.")
    parser.add_argument('--number', type=float, help="Input value to predict the next number in the sequence.")
    args = parser.parse_args()

    # Determine the test_input_value either from command line argument or user input.
    if args.number is not None:
        test_input_value = args.number
    else:
        # If no command-line input is provided, prompt the user for input.
        user_input = input("Enter a value to predict the next number in the sequence (Press Enter for default 8): ")
        test_input_value = float(user_input) if user_input else 8  # Default to 8 if no input is given.
    print(f"Using input value: {test_input_value}")

    # Define the input and output data (sequence of numbers).
    X = np.array([[1], [2], [3], [4], [5], [6], [7]])
    y = np.array([[1], [2], [3], [4], [5], [6], [7]])

    # Network architecture settings (1 input neuron, 5 hidden neurons, 1 output neuron).
    input_size = 1
    hidden_size = 5
    output_size = 1
    ann = ANN(input_size, hidden_size, output_size) # Initialize the Artificial Neural Network (ANN).
    loss_history = ann.train(X, y, epochs=90000, learning_rate=0.01) # Train the ANN with sequence data.

    # Plot the training loss curve to visualize how the model is learning over time.
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show() # Uncomment to display training graph.

    # Testing the trained model with the input value provided by the user.
    test_input = np.array([[test_input_value]])
    predicted_output = ann.forward(test_input)
    print(f"Predicted output for input {test_input[0][0]}: {predicted_output[0][0]:.4f}")

    # Extend the range of inputs to test predictions for new values (testing generalization). For graph.
    extended_X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [14], [15]])
    predictions = ann.forward(extended_X)

    # Plot the predictions vs actual target values.
    plt.figure(figsize=(10, 6))
    plt.plot(X, y, label='Actual', color='blue', marker='o')
    plt.plot(extended_X, predictions, label='Predicted', color='red', linestyle='dashed', marker='x')
    plt.title("Predicted vs Actual Values (Extended Range)")
    plt.xlabel("Input (X)")
    plt.ylabel("Output (y)")
    plt.legend()
    plt.grid(True)
    plt.show() # Uncomment to display testing graph.

# Big red activation button.
if __name__ == "__main__":
    main()
