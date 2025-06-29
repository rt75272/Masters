import os
# Suppress TensorFlow output (must be set before importing tensorflow).
# 0=all_messages, 1=info, 2=warning, 3=error.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supress TensorFlow output.
import glob
import random
import argparse
import numpy as np
import seaborn as sns
import concurrent.futures
from collections import defaultdict
from tensorflow.keras import layers # type: ignore
from tensorflow import keras
from PIL import Image
from sklearn.metrics import confusion_matrix
# -------------------------------------------------------------------------------------
# Cat VS Dog Classifier
# 
# Implements a simple image classifier to distinguish between cats and dogs.
#
# Usage:
#   # Run everything with training:
#       $ python cat_vs_dog.py
#   # Only use a previously trained model. No training:
#       $ python cat_vs_dog.py --use-model conv_model_big
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

def load_images_parallel(paths):
    """Helper to load images in parallel and reduce execution time."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        images = list(executor.map(pixels_from_path, paths))
    images = np.asarray(images)
    return images

def load_dataset(cat_dir, dog_dir, sample_size):
    """Loads a balanced set of cat and dog images."""
    cat_paths = glob.glob(os.path.join(cat_dir, '*'))
    dog_paths = glob.glob(os.path.join(dog_dir, '*'))
    min_count = min(len(cat_paths), len(dog_paths), sample_size)
    cat_paths = cat_paths[:min_count]
    dog_paths = dog_paths[:min_count]
    cat_set = load_images_parallel(cat_paths)
    dog_set = load_images_parallel(dog_paths)
    x = np.concatenate([cat_set, dog_set])
    y = np.asarray([1]*min_count + [0]*min_count)
    # Shuffle the dataset.
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
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
    """Builds a deeper convolutional neural network (CNN) model. If extra_conv is True, 
    adds an extra convolutional block."""
    inputs = keras.Input(shape=input_shape, name='ani_image')
    # First convolutional block.
    x = layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    # Second convolutional block.
    x = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    # Optional extra convolutional block.
    if extra_conv:
        x = layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.25)(x)
    # Fully connected layers.
    x = layers.Flatten()(x)
    x = layers.Dense(fc_layer_size, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='class')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", 
                keras.metrics.AUC(name="auc"), 
                "binary_crossentropy", 
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

def silent_show(img):
    """Displays an image without printing to stderr. Temporarily redirect stderr."""
    with open(os.devnull, 'w') as f, os.fdopen(os.dup(2), 'w') as old_stderr:
        os.dup2(f.fileno(), 2)
        img.show()
        os.dup2(old_stderr.fileno(), 2)

def main():
    """Main driver function."""
    parser = argparse.ArgumentParser(description="Cat vs Dog Classifier")
    parser.add_argument('--use-model', type=str, default=None,
    help="Path to a previously trained model. If set, only predictions will be shown "
    "(no training).")
    parser.add_argument('--index', type=int, default=None,
    help="Index of the validation image to display and predict. If not set, a random " \
    "image will be used.")
    args = parser.parse_args()
    num_epochs = 10  # Number of epochs for training.
    # Define data directories with 'data/' prefix.
    TRAIN_CAT_DIR = os.path.join('data', 'train', 'cats')
    TRAIN_DOG_DIR = os.path.join('data', 'train', 'dogs')
    TEST_CAT_DIR = os.path.join('data', 'test', 'cats')
    TEST_DOG_DIR = os.path.join('data', 'test', 'dogs')
    # Always load validation data for viewing predictions.
    x_valid, labels_valid = load_dataset(TEST_CAT_DIR, TEST_DOG_DIR, VALID_SIZE)
    if args.use_model:
        # Loads the previously trained model.
        print(f"Loading model from {args.use_model} ...")
        model = keras.models.load_model(args.use_model)
    else:
        # Analyze image shapes (for debugging).
        analyze_image_shapes(TRAIN_CAT_DIR)
        # Loads training data.
        x_train, labels_train = load_dataset(TRAIN_CAT_DIR, TRAIN_DOG_DIR, SAMPLE_SIZE)
        # Builds and trains a dense model.
        dense_model = build_dense_model((IMG_SIZE[1], IMG_SIZE[0], 3))
        train_and_evaluate(dense_model, 
                        x_train, 
                        labels_train, 
                        x_valid, 
                        labels_valid, 
                        epochs=num_epochs)
        # Builds and trains a basic CNN model.
        cnn_model = build_cnn_model((IMG_SIZE[1], IMG_SIZE[0], 3))
        _, preds = train_and_evaluate(cnn_model, 
                                    x_train, 
                                    labels_train, 
                                    x_valid, 
                                    labels_valid, 
                                    epochs=num_epochs)
        # Builds and trains a deeper CNN model.
        cnn_model2 = build_cnn_model((IMG_SIZE[1], IMG_SIZE[0], 3), extra_conv=True)
        _, preds2 = train_and_evaluate(cnn_model2, 
                                    x_train, 
                                    labels_train, 
                                    x_valid, 
                                    labels_valid, 
                                    epochs=num_epochs)
        scatterplot_analysis(preds2, labels_valid) # Analyze predictions w/ scatterplot.
        save_model(cnn_model2, 'conv_model_big') # Saves the best model.
        model = cnn_model2
    # Picks a random index, if not specified.
    if args.index is not None:
        index = args.index
    else:
        index = random.randint(0, len(x_valid) - 1)
    print(f"x_valid shape: {x_valid.shape}")
    if x_valid is None or len(x_valid) <= index:
        print(f"Validation data is missing or index {index} is out of range.")
        return
    prob = cat_index(model, x_valid, index)
    if prob is None:
        print(f"Model prediction failed for index {index}.")
        return
    cat_label = labels_valid[index]
    msg = f"Index {index}: "
    msg += f"Probability of being a cat: {prob:.2f} (Label: {cat_label})"
    print(msg)
    img = animal_pic(x_valid, index)
    silent_show(img)
    # Confusion Matrix.
    pred_labels = (model.predict(x_valid) > 0.5).astype(int).flatten()
    cm = confusion_matrix(labels_valid, pred_labels)
    print("Confusion Matrix:\n", cm)

# The big red activation button.
if __name__ == "__main__":
    main()
