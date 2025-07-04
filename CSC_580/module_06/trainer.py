import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from config import BATCH_SIZE, CHECKPOINT_PATH
"""Model training and evaluation utilities."""
class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self):
        """Constructor for ModelTrainer."""
        self.data_augmentation = self._create_data_augmentation()
    
    def _create_data_augmentation(self):
        """Create data augmentation pipeline."""
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),])
    
    def _augment_data(self, x, y):
        """Apply data augmentation to training data."""
        x = self.data_augmentation(x, training=True)
        return x, y
    
    def _create_callbacks(self):
        """Create training callbacks for better performance."""
        return [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=7,
                restore_best_weights=True,
                verbose=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=1e-8,
                verbose=1),
            keras.callbacks.ModelCheckpoint(
                filepath=CHECKPOINT_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1)]
    
    def train_and_evaluate(self, model, x_train, y_train, x_valid, y_valid, 
                          epochs=50, batch_size=BATCH_SIZE):
        """Trains the given model and evaluates it on the validation set."""
        print(f'# Fit model: {model.name}')
        # Create datasets.
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.map(self._augment_data)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        # Train the model
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=self._create_callbacks(),
            verbose=1)
        # Make predictions
        preds = model.predict(x_valid)
        # Handle both (n,) and (n, 1) output shapes
        if len(preds.shape) > 1 and preds.shape[1] == 1:
            preds = preds.flatten()
        # Calculate Pearson correlation
        pearson = np.corrcoef(preds, y_valid)[0][1]
        print("Pearson correlation:", pearson)
        return history, preds
    
    def save_model(self, model, path):
        """Saves the trained model to the specified path."""
        model.save(path)
        print(f"Model saved to {path}")
