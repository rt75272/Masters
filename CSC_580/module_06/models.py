import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from config import DENSE_LEARNING_RATE, LEARNING_RATE, DENSE_FC_SIZE, CNN_FC_SIZE
"""Neural network model definitions."""
class ModelBuilder:
    """Builds different types of neural network models."""
    def __init__(self, gpu_manager):
        """Constructor for ModelBuilder."""
        self.gpu_manager = gpu_manager
    
    def build_dense_model(self, input_shape, fc_size=DENSE_FC_SIZE):
        """Builds a simple dense (fully connected) neural network model."""
        with tf.device(self.gpu_manager.get_device_context()):
            inputs = keras.Input(shape=input_shape, name='ani_image')
            x = layers.Flatten(name='flattened_img')(inputs)
            x = layers.Dense(fc_size, activation='relu', name='first_layer')(x)
            outputs = layers.Dense(1, activation='sigmoid', name='class')(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=DENSE_LEARNING_RATE),
                loss="binary_crossentropy", 
                metrics=["accuracy"])
        return model
    
    def build_cnn_model(self, input_shape, fc_layer_size=CNN_FC_SIZE, extra_conv=True):
        """Builds a deeper convolutional neural network (CNN) model."""
        with tf.device(self.gpu_manager.get_device_context()):
            inputs = keras.Input(shape=input_shape, name='ani_image')
            # First convolutional block
            x = layers.Conv2D(
                64, 
                kernel_size=3, 
                activation='relu', 
                padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Dropout(0.25)(x)
            # Second convolutional block
            x = layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Dropout(0.3)(x)
            # Third convolutional block
            x = layers.Conv2D(256, kernel_size=3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(256, kernel_size=3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Dropout(0.4)(x)
            # Optional fourth convolutional block for deeper model
            x = layers.Conv2D(
                512, 
                kernel_size=3, 
                activation='relu', 
                padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(
                512,
                kernel_size=3,
                activation='relu',
                padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Dropout(0.5)(x)
            # Global Average Pooling instead of Flatten to reduce overfitting
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(fc_layer_size, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            outputs = layers.Dense(1, activation='sigmoid', name='class')(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss="binary_crossentropy",
                metrics=["accuracy", keras.metrics.AUC(name="auc")])
        return model
