import os
import glob
import numpy as np
import seaborn as sns
import tensorflow as tf
from collections import defaultdict
from tensorflow.keras import layers # type: ignore
from tensorflow import keras
from PIL import Image
# -------------------------------------------------------------------------------------
# Cat VS Dog Classifier
# 
# Implements a simple image classifier to distinguish between cats and dogs.
#
# Usage:
#   python cat_vs_dog.py
# -------------------------------------------------------------------------------------
IMG_SIZE = (94, 125) # Image size for resizing all images.
# Number of images to use for training and validation.
SAMPLE_SIZE = 2048
VALID_SIZE = 512

def pixels_from_path(file_path):
    """Loads an image from a file path, resizes it, and returns it as a numpy array."""
    im = Image.open(file_path)
    im = im.resize(IMG_SIZE)
    return np.array(im)

def analyze_image_shapes(cat_dir, sample_count=1000):
    """Analyzes the shapes of the first sample_count images in the given directory.
    Prints progress every 100 images. Returns a dictionary with shape counts."""
    shape_counts = defaultdict(int)
    for i, cat in enumerate(glob.glob(os.path.join(cat_dir, '*'))[:sample_count]):
        if i % 100 == 0:
            print(f"Processed {i} images")
        shape_counts[str(pixels_from_path(cat).shape)] += 1
    return shape_counts

def load_dataset(cat_dir, dog_dir, sample_size):
    """Loads cat and dog images from the specified directories. Returns concatenated 
    image arrays and corresponding labels."""
    print(f"Loading cat images from {cat_dir}...")
    cat_paths = glob.glob(os.path.join(cat_dir, '*'))[:sample_size]
    cat_images = []
    for cat in cat_paths:
        img = pixels_from_path(cat)
        cat_images.append(img)
    cat_set = np.asarray(cat_images)
    print(f"Loading dog images from {dog_dir}...")
    dog_paths = glob.glob(os.path.join(dog_dir, '*'))[:sample_size]
    dog_images = []
    for dog in dog_paths:
        img = pixels_from_path(dog)
        dog_images.append(img)
    dog_set = np.asarray(dog_images)
    x = np.concatenate([cat_set, dog_set])
    y = np.asarray([1]*sample_size + [0]*sample_size) # 1 for cats, 0 for dogs.
    return x, y

def build_dense_model(input_shape, fc_size=512):
    """Builds a simple dense (fully connected) neural network model."""
    inputs = keras.Input(shape=input_shape, name='ani_image')
    x = layers.Flatten(name='flattened_img')(inputs)
    x = layers.Dense(fc_size, activation='relu', name='first_layer')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='class')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mean_squared_error", 
            metrics=["binary_crossentropy", 
            "mean_squared_error"])
    return model

def build_cnn_model(input_shape, fc_layer_size=128, extra_conv=False):
    """Builds a convolutional neural network (CNN) model. If extra_conv is True, 
    adds an extra convolutional layer."""
    inputs = keras.Input(shape=input_shape, name='ani_image')
    x = layers.Conv2D(24, kernel_size=3, activation='relu')(inputs)
    if extra_conv:
        x = layers.Conv2D(48, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten(name='flattened_features')(x)
    x = layers.Dense(fc_layer_size, activation='relu', name='first_layer')(x)
    x = layers.Dense(fc_layer_size, activation='relu', name='second_layer')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='class')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-6),
            loss="binary_crossentropy", 
            metrics=["binary_crossentropy", 
            "mean_squared_error"])
    return model

def train_and_evaluate(model, x_train, y_train, x_valid, y_valid, epochs=10, 
                    batch_size=32):
    """Trains the given model and evaluates it on the validation set. Prints the 
    Pearson correlation between predictions and true labels. Returns the training 
    history and predictions."""
    print(f'# Fit model: {model.name}')
    history = model.fit(x_train, y_train, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    epochs=epochs,
                    validation_data=(x_valid, y_valid))
    preds = model.predict(x_valid)
    preds = np.asarray([pred[0] for pred in preds])
    pearson = np.corrcoef(preds, y_valid)[0][1]
    print("Pearson correlation:", pearson)
    return history, preds

def scatterplot_analysis(preds, labels_valid):
    """Plots a scatterplot of predictions vs. true labels. Prints accuracy for 
    various thresholds."""
    sns.scatterplot(x=preds, y=labels_valid)
    for i in range(1, 10):
        threshold = 0.1 * i
        selected = labels_valid[preds > threshold]
        if len(selected) > 0:
            acc = sum(selected) / len(selected)
        else:
            acc = 0
        print(f"Threshold: {threshold:.1f}, Accuracy: {acc}")

def save_model(model, path):
    """Saves the trained model to the specified path."""
    model.save(path)
    print(f"Model saved to {path}")

def animal_pic(x_valid, index):
    """Returns a PIL Image object for the image at the given index in the 
    validation set."""
    img_array = x_valid[index]
    img = Image.fromarray(img_array)
    return img

def cat_index(model, x_valid, index):
    """Returns the predicted probability that the image at the given index is a cat."""
    img_array = np.asarray([x_valid[index]])
    prediction = model.predict(img_array)
    prob = prediction[0][0]
    return prob

def main():
    """Main driver function."""
    num_epochs = 10  # Number of epochs for training.
    # Define data directories with 'data/' prefix.
    TRAIN_CAT_DIR = os.path.join('data', 'train', 'cats')
    TRAIN_DOG_DIR = os.path.join('data', 'train', 'dogs')
    TEST_CAT_DIR = os.path.join('data', 'test', 'cats')
    TEST_DOG_DIR = os.path.join('data', 'test', 'dogs')
    analyze_image_shapes(TRAIN_CAT_DIR) # Analyze image shapes (for debugging).
    # Load training and validation datasets.
    x_train, labels_train = load_dataset(TRAIN_CAT_DIR, TRAIN_DOG_DIR, SAMPLE_SIZE)
    x_valid, labels_valid = load_dataset(TEST_CAT_DIR, TEST_DOG_DIR, VALID_SIZE)
    # Build and train a dense model.
    dense_model = build_dense_model((IMG_SIZE[1], IMG_SIZE[0], 3))
    train_and_evaluate(dense_model, 
                    x_train, 
                    labels_train, 
                    x_valid, 
                    labels_valid, 
                    epochs=num_epochs)
    # Build and train a basic CNN model.
    cnn_model = build_cnn_model((IMG_SIZE[1], IMG_SIZE[0], 3))
    _, preds = train_and_evaluate(cnn_model, 
                                x_train, 
                                labels_train, 
                                x_valid, 
                                labels_valid, 
                                epochs=num_epochs)
    # Build and train a deeper CNN model.
    cnn_model2 = build_cnn_model((IMG_SIZE[1], IMG_SIZE[0], 3), extra_conv=True)
    _, preds2 = train_and_evaluate(cnn_model2, 
                                x_train, 
                                labels_train, 
                                x_valid, 
                                labels_valid, 
                                epochs=num_epochs)
    scatterplot_analysis(preds2, labels_valid) # Analyze predictions with scatterplot.
    save_model(cnn_model2, 'conv_model_big') # Save the best model.
    index = 600 # Show prediction and image for a sample index.
    # Check if x_valid is loaded and index is valid.
    print(f"x_valid shape: {x_valid.shape}")
    if x_valid is None or len(x_valid) <= index:
        print("Validation data is missing or index is out of range.")
        return
    prob = cat_index(cnn_model2, x_valid, index)
    print(f"cat_index returned: {prob}")
    if prob is None:
        print("Model prediction failed.")
        return
    print("Probability of being a cat: {:.2f}".format(prob))
    img = animal_pic(x_valid, index)
    img.show()

# The big red activation button.
if __name__ == "__main__":
    main()
