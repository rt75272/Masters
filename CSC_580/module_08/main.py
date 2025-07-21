import numpy as np
import tensorflow as tf
from data_generator import DataGenerator
from encoder_decoder_model import EncoderDecoderModel
from model_trainer import ModelTrainer
from gpu_config import configure_gpu
"""Keras Encoder-Decoder Sequence-to-Sequence Model.

Main entry point for the encoder-decoder sequence-to-sequence model.
Coordinates data generation, model training, and evaluation.

Usage:
    $ python main.py
"""

def main():
    """Main driver function to run the encoder-decoder sequence-to-sequence model."""
    # Configure GPU before any other operations.
    gpu_available = configure_gpu()
    # Set random seed for reproducibility.
    np.random.seed(42)
    tf.random.set_seed(42)
    # Display system information.
    print("=" * 50)
    print("System Configuration:")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    if tf.config.list_physical_devices('GPU'):
        print(f"GPU Device: {tf.config.list_physical_devices('GPU')[0]}")
    device_type = "GPU" if gpu_available else "CPU"
    print(f"Execution Device: {device_type}")
    print("=" * 50)
    n_features = 50 + 1
    n_steps_in = 6
    n_steps_out = 3
    n_units = 256
    n_samples = 15000 
    dropout_rate = 0.2  # Regularization.
    data_generator = DataGenerator(cardinality=n_features, random_seed=42)
    model = EncoderDecoderModel(n_features, n_features, n_units, 
                               use_gpu=gpu_available, dropout_rate=dropout_rate)
    trainer = ModelTrainer(model, data_generator)
    print("Generating dataset...")
    X1, X2, y = data_generator.get_dataset(n_steps_in, 
                                        n_steps_out, 
                                        n_samples)
    X1 = X1.reshape((n_samples, n_steps_in, n_features))
    X2 = X2.reshape((n_samples, n_steps_out, n_features))
    y = y.reshape((n_samples, n_steps_out, n_features))
    print("Compiling model...")
    model.compile_model(optimizer='adam', 
                       loss='categorical_crossentropy', 
                       metrics=['accuracy'],
                       learning_rate=0.001)
    print(f"Training model on {device_type}...")
    model.train(X1, X2, y, epochs=50, batch_size=64, verbose=2, use_callbacks=True)
    print("Evaluating model accuracy...")
    trainer.evaluate_accuracy(n_steps_in, n_steps_out, n_features, n_test_samples=100)
    print("Saving predictions...")
    trainer.save_predictions(n_steps_in, n_steps_out, n_features, 
                           filename='output_predictions.txt', n_samples=10)
    print("Training and evaluation complete!")
    if not gpu_available:
        print("\nNo GPU available. Ensure CUDA drivers and libraries are installed.")

# The big red activation button.
if __name__ == '__main__':
    main()
