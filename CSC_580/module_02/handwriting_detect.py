import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow output.
import matplotlib.pyplot as plt
import tensorflow as tf # type: ignore
from tensorflow.keras import layers, models # type: ignore
# --------------------------------------------------------------------------------------
# Handwriting Detection with the MNIST Dataset.
#
# Loads the MNIST dataset, preprocesses it, builds a neural network model, trains the
# model on the training set, and evaluates it on the test set. It also includes
# functionality to display a sample image from the dataset. The MNIST dataset consists
# of 70,000 images of handwritten digits (0-9), each image is 28x28 pixels, and the
# task is to classify these images into one of the 10 digit classes.
#
# Usage:
#    $ python handwriting_detect.py
# --------------------------------------------------------------------------------------
def load_and_prepare_data():
    """Loads the MNIST dataset, flattens and normalizes the images,and one-hot encodes 
    the labels."""
    # Load MNIST dataset.
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Flatten and normalize images.
    train_images = x_train.reshape(60000, 784).astype('float32') / 255.0
    test_images = x_test.reshape(10000, 784).astype('float32') / 255.0
    # One-hot encode labels.
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return train_images, y_train, test_images, y_test

def display_sample(images, labels, num):
    """Displays a single sample image from the dataset."""
    print("Label (one-hot):", labels[num])
    label = labels[num].argmax()
    image = images[num].reshape([28, 28])
    plt.title(f'Sample: {num}, Label: {label}')
    plt.imshow(image, cmap='gray_r')
    plt.show()

def build_model(hidden_nodes=512):
    """Builds and compiles a simple neural network model for MNIST digits."""
    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(hidden_nodes, activation='relu'),
        layers.Dense(10, activation='softmax')])
    model.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

def train_and_evaluate(
    model,
    train_images,
    y_train,
    test_images,
    y_test,
    epochs=20,
    batch_size=100):
        """Trains the model and evaluates its performance on the test set."""
        history = model.fit(
            train_images, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(test_images, y_test))
        test_loss, test_acc = model.evaluate(test_images, y_test)
        print(f"Test accuracy: {test_acc:.4f}")
        return history, test_loss, test_acc

def main():
    """Main driver function to run the MNIST handwriting detection pipeline. Loads data,
    displays a sample, builds the model, and trains/evaluates it."""
    train_images, y_train, test_images, y_test = load_and_prepare_data()
    # display_sample(train_images, y_train, 1242)
    model = build_model(hidden_nodes=512)
    train_and_evaluate(
        model, 
        train_images, 
        y_train, 
        test_images, 
        y_test, 
        epochs=42, 
        batch_size=100)
    # Show a test image.
    idx = 42
    display_sample(test_images, y_test, idx)
    # Predict and show result.
    prediction = model.predict(test_images[idx].reshape(1, 784))
    predicted_label = prediction.argmax()
    true_label = y_test[idx].argmax()
    print(f"Predicted label: {predicted_label}")
    print(f"True label: {true_label}")

# The big red activation button.
if __name__ == "__main__":
    main()