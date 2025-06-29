import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# -------------------------------------------------------------------------------------
# Logistic Regression.
#
# Implements a logistic regression model using TensorFlow.
#
# Usage:
#    $ python logistic_regression.py
# -------------------------------------------------------------------------------------
TWO = 2

def generate_synthetic_data(N=99):
    """Generates synthetic 2D data for binary classification."""
    cov = 0.42 * np.eye(TWO)
    mean_zeros = np.array([-1, -1])
    size_zeros = N // TWO
    x_zeros = np.random.multivariate_normal(
        mean=mean_zeros, 
        cov=cov, 
        size=size_zeros)
    y_zeros = np.zeros((N//TWO,))
    mean_ones = np.array([1, 1])
    size_ones = N // TWO
    x_ones = np.random.multivariate_normal(
        mean=mean_ones, 
        cov=cov, 
        size=size_ones)
    y_ones = np.ones((N//TWO,))
    x_np = np.vstack([x_zeros, x_ones])
    y_np = np.concatenate([y_zeros, y_ones])
    return x_np, y_np, x_zeros, x_ones

def plot_data(x_zeros, x_ones):
    """Plots the synthetic data points for both classes."""
    plt.figure(figsize=(8, 6))
    plt.scatter(
        x_zeros[:, 0],
        x_zeros[:, 1],
        color='blue', 
        label='Class 0')
    plt.scatter(
        x_ones[:, 0],
        x_ones[:, 1], 
        color='red', 
        label='Class 1')
    plt.title('Synthetic Data')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True)
    plt.show()

def build_graph(N):
    """Builds the TensorFlow graph for logistic regression."""
    tf.compat.v1.disable_eager_execution() # Disable eager execution for compatibility.
    with tf.name_scope("placeholders"):
        # Input features.
        x = tf.compat.v1.placeholder(
            tf.float32, 
            (N, TWO))
        # Labels.
        y = tf.compat.v1.placeholder(
            tf.float32, 
            (N,))
    with tf.name_scope("weights"):
        W = tf.Variable(tf.random.normal((TWO, 1))) # Weights.
        b = tf.Variable(tf.random.normal((1,))) # Bias.
    with tf.name_scope("prediction"):
        y_logit = tf.squeeze(tf.matmul(x, W) + b)
        y_one_prob = tf.sigmoid(y_logit) # Sigmoid for probability.
        y_pred = tf.round(y_one_prob) # Predicted class (0 or 1).
    with tf.name_scope("loss"):
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y_logit, 
            labels=y)
        l = tf.reduce_sum(entropy) # Total loss.
    with tf.name_scope("optim"):
        train_op = tf.compat.v1.train.AdamOptimizer(0.01).minimize(l)
    return x, y, W, b, y_pred, l, train_op

def calculate_accuracy(y_true, y_pred):
    """Calculate the accuracy of predictions."""
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    return accuracy

def demonstrate_prediction(W_np, b_np):
    """Demonstrate prediction on a few sample points."""
    # Sample test points
    test_points = np.array([[-1.5, -1.5], [1.5, 1.5], [0.01, 0.01], [-0.5, 0.5]])
    
    print("\nDemonstration Predictions:")
    print("Point\t\tLogit\t\tProbability\tPrediction")
    print("-" * 60)
    
    for point in test_points:
        # Calculate logit: W1*x1 + W2*x2 + b
        logit = W_np[0] * point[0] + W_np[1] * point[1] + b_np[0]
        # Calculate probability using sigmoid
        probability = 1 / (1 + np.exp(-logit))
        # Make prediction
        prediction = 1 if probability > 0.5 else 0
        
        print(f"{point}\t{logit:.3f}\t\t{probability:.3f}\t\t{prediction}")

def train_model(x, y, W, b, y_pred, l, train_op, x_np, y_np, N, num_steps=1000):
    """Trains the logistic regression model using TensorFlow."""
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(num_steps):
            _, loss = sess.run([train_op, l], feed_dict={x: x_np, y: y_np})
            if i % 100 == 0:
                print(f'Iteration {i}, loss: {loss}')
        
        # Retrieve trained weights and predictions
        y_pred_np, W_np, b_np = sess.run([y_pred, W, b], feed_dict={x: x_np, y: y_np})
    

    
    # Convert to numpy arrays and flatten for safe indexing
    W_np = np.array(W_np).flatten()
    b_np = np.array(b_np).flatten()
    
    # Print model weights and accuracy
    print(f"\nModel Weights:")
    print(f"W1 (weight for X1): {W_np[0]:.4f}")
    print(f"W2 (weight for X2): {W_np[1]:.4f}")
    print(f"Bias: {b_np[0]:.4f}")
    
    accuracy = calculate_accuracy(y_np, y_pred_np)
    print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Demonstrate predictions
    demonstrate_prediction(W_np, b_np)
    
    return y_pred_np, W_np, b_np

def plot_predictions(x_np, y_pred_np, y_np, W_np=None, b_np=None):
    """Plots predicted vs actual labels for the dataset."""
    plt.figure(figsize=(8, 6))
    # Plot predicted labels.
    plt.scatter(
        x_np[:, 0],
        x_np[:, 1],
        c=y_pred_np,
        cmap='bwr',
        alpha=0.6,
        edgecolor='k',
        label='Predictions')
    # Plot actual labels.
    plt.scatter(
        x_np[:, 0],
        x_np[:, 1],
        c=y_np,
        cmap='cool',
        marker='+',
        s=100,
        label='Actual Labels')
    plt.title('Predicted vs Actual Labels')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """Main driver function to run the logistic regression workflow."""
    N = 142
    x_np, y_np, x_zeros, x_ones = generate_synthetic_data(N)
    plot_data(x_zeros, x_ones)
    x, y, W, b, y_pred, l, train_op = build_graph(N)
    y_pred_np, W_np, b_np = train_model(x, y, W, b, y_pred, l, train_op, x_np, y_np, N)
    plot_predictions(x_np, y_pred_np, y_np, W_np, b_np)

# The big red activation button.
if __name__=="__main__":
    main()
