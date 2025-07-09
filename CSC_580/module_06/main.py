import os
import random
import argparse
import numpy as np
from tensorflow import keras
# Suppress TensorFlow warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from config import *
from gpu_utils import GPUManager
from data_loader import DataLoader
from models import ModelBuilder
from trainer import ModelTrainer
from analyzer import ModelAnalyzer
"""Cat vs Dog Classifier - Modular Implementation

A machine learning project that classifies images of cats and dogs using deep learning.
This version uses a modular, object-oriented architecture for better maintainability.

Usage:
    # Train and evaluate models:
    python main.py
    
    # Use a pre-trained model:
    python main.py --use-model conv_model_big.keras
    
    # Analyze a specific image:
    python main.py --use-model conv_model_big.keras --index 42
"""
class CatDogClassifier:
    """Main class that orchestrates the cat vs dog classification workflow."""
    def __init__(self):
        """Initialize all components and set up the classifier pipeline."""
        self.gpu_manager = GPUManager()
        self.data_loader = DataLoader() 
        self.model_builder = ModelBuilder(self.gpu_manager)
        self.trainer = ModelTrainer() 
        self.analyzer = ModelAnalyzer()
        self.x_train = None 
        self.y_train = None 
        self.x_valid = None
        self.y_valid = None 
        self.dense_model = None 
        self.cnn_model = None 
        self.best_model = None 
        
    def setup_environment(self):
        """Set up the environment and check GPU availability."""
        print("=" * 60)
        print("CAT VS DOG CLASSIFIER - MODULAR VERSION")
        print("=" * 60)
        print("GPU CONFIGURATION CHECK")
        print("=" * 60)
        # Check and display GPU availability.
        self.gpu_manager.check_gpu_usage()
        print("=" * 60)
        
    def load_data(self):
        """Load and prepare the dataset for training and validation."""
        print("Loading dataset...")
        if not self.data_loader.validate_data_directories(
                TRAIN_CAT_DIR, 
                TRAIN_DOG_DIR, 
                TEST_CAT_DIR,
                TEST_DOG_DIR):
            return False
        print("Loading training data...")
        self.x_train, self.y_train = self.data_loader.load_dataset(
                TRAIN_CAT_DIR, 
                TRAIN_DOG_DIR, 
                SAMPLE_SIZE)
        print("Loading validation data...")
        self.x_valid, self.y_valid = self.data_loader.load_dataset(
                TEST_CAT_DIR, 
                TEST_DOG_DIR, 
                VALID_SIZE)
        if (len(self.x_train) == 0 or len(self.y_train) == 0 or 
            len(self.x_valid) == 0 or len(self.y_valid) == 0):
            print("ERROR: Failed to load training or validation data!")
            return False
        print(f"Training data: {self.x_train.shape[0]} samples")
        print(f"Validation data: {self.x_valid.shape[0]} samples")
        print("Analyzing image shapes...")
        self.data_loader.analyze_image_shapes(TRAIN_CAT_DIR, 100)
        return True
    
    def train_models(self):
        """Train different model architectures and compare their performance."""
        # Define input shape for all models (height, width, channels).
        input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
        print("\n" + "=" * 50)
        print("TRAINING PHASE")
        print("=" * 50)
        print("Training Dense Neural Network...")
        self.dense_model = self.model_builder.build_dense_model(input_shape)
        self.trainer.train_and_evaluate(
            self.dense_model, self.x_train, self.y_train, 
            self.x_valid, self.y_valid, epochs=EPOCHS)
        # Uses convolutional layers to extract spatial features.
        print("\nTraining Basic CNN...")
        self.cnn_model = self.model_builder.build_cnn_model(input_shape)
        _, preds = self.trainer.train_and_evaluate(
            self.cnn_model, self.x_train, self.y_train, 
            self.x_valid, self.y_valid, epochs=EPOCHS)
        # Includes extra convolutional layers for better feature extraction.
        print("\nTraining Advanced CNN...")
        self.best_model = self.model_builder.build_cnn_model(input_shape, 
                                                           extra_conv=True)
        _, preds_best = self.trainer.train_and_evaluate(
            self.best_model, self.x_train, self.y_train, 
            self.x_valid, self.y_valid, epochs=EPOCHS)
        print("\nAnalyzing best model predictions...")
        self.analyzer.scatterplot_analysis(preds_best, self.y_valid)
        self.trainer.save_model(self.best_model, MODEL_SAVE_PATH)
        return self.best_model
    
    def load_pretrained_model(self, model_path):
        """Load a pre-trained model from disk."""
        print(f"Loading pre-trained model from {model_path}...")
        try:
            self.best_model = keras.models.load_model(model_path)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def analyze_model_performance(self):
        """Analyze model performance on validation set using various metrics."""
        if self.best_model is None:
            print("No model available for analysis!")
            return
        print("\n" + "=" * 50)
        print("MODEL ANALYSIS")
        print("=" * 50)
        self.analyzer.generate_confusion_matrix(
            self.best_model, self.x_valid, self.y_valid)
    
    def predict_single_image(self, index=None):
        """Predict and display a single image from the validation set."""
        if self.best_model is None or self.x_valid is None:
            print("Model or validation data not available!")
            return
        if index is None:
            index = random.randint(0, len(self.x_valid) - 1)
        if index >= len(self.x_valid):
            print(f"Index {index} is out of range (max: {len(self.x_valid) - 1})")
            return
        print(f"\nPredicting image at index {index}...")
        print(f"Validation set shape: {self.x_valid.shape}")
        probability = self.analyzer.get_prediction(self.best_model, 
                                                  self.x_valid, index)
        true_label = self.y_valid[index]
        result = self.analyzer.format_prediction_result(index, probability, 
                                                       true_label)
        print(result)
        img = self.analyzer.get_image_for_display(self.x_valid, index)
        self.analyzer.display_image_silently(img)
    
    def interactive_prediction_loop(self):
        """Allow user to interactively explore multiple test images."""
        if self.best_model is None or self.x_valid is None:
            print("Model or validation data not available!")
            return
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE")
        print("=" * 60)
        print(f"Available images: 0 to {len(self.x_valid) - 1}")
        print("Commands:")
        print("  - Enter a number to view that image")
        print("  - Press Enter for random image")
        print("  - Type 'quit' or 'q' to exit")
        print("=" * 60)
        while True:
            try:
                user_input = input("\nEnter image index (or 'quit'): ").strip()
                if user_input.lower() in ['quit', 'q', 'exit']:
                    print("Exiting interactive mode...")
                    break
                if user_input == "":
                    index = random.randint(0, len(self.x_valid) - 1)
                    print(f"Randomly selected index: {index}")
                else:
                    try:
                        index = int(user_input)
                        if index < 0 or index >= len(self.x_valid):
                            print(f"Index must be between 0 and {len(self.x_valid) - 1}")
                            continue
                    except ValueError:
                        print("Invalid input. Please enter a number or 'quit'.")
                        continue
                self.predict_single_image(index)
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except EOFError:
                print("\nExiting interactive mode...")
                break
    
    def run_full_pipeline(self, interactive=True):
        """Run the complete training and evaluation pipeline."""
        self.setup_environment()
        if not self.load_data():
            return
        self.train_models()
        self.analyze_model_performance()
        if interactive:
            self.interactive_prediction_loop()
        else:
            self.predict_single_image()
        print("\n" + "=" * 50)
        print("PIPELINE COMPLETE")
        print("=" * 50)
    
    def run_inference_only(self, model_path, image_index=None, interactive=True):
        """Run inference only with a pre-trained model (no training)."""
        self.setup_environment()
        print("Loading validation data...")
        self.x_valid, self.y_valid = self.data_loader.load_dataset(
            TEST_CAT_DIR, TEST_DOG_DIR, VALID_SIZE)
        # Verify validation data was loaded successfully.
        if len(self.x_valid) == 0 or len(self.y_valid) == 0:
            print("ERROR: Failed to load validation data!")
            return
        if not self.load_pretrained_model(model_path):
            return
        self.analyze_model_performance()
        if image_index is not None:
            self.predict_single_image(image_index)
        if interactive:
            self.interactive_prediction_loop()
        elif image_index is None:
            self.predict_single_image()
        print("\n" + "=" * 50)
        print("INFERENCE COMPLETE")
        print("=" * 50)

def main():
    """Main entry point - handles command line arguments and workflow selection."""
    # Set up command line argument parser with detailed help.
    parser = argparse.ArgumentParser(
        description="Cat vs Dog Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python main.py # Train and evaluate models
            python main.py --use-model conv_model_big.keras # Use pre-trained model
            python main.py --use-model model.keras --index 42 # Predict specific image
            python main.py --use-model model.keras --no-interactive # Run once and exit
        """)
    parser.add_argument(
        '--use-model', 
        type=str, 
        default=None,
        help='Path to a pre-trained model file. If provided, skips training.')
    parser.add_argument(
        '--index', 
        type=int, 
        default=None,
        help='Index of validation image to analyze. If not provided, uses random.')
    parser.add_argument(
        '--no-interactive', 
        action='store_true',
        help='Disable interactive mode. Run once and exit.')
    args = parser.parse_args()
    classifier = CatDogClassifier()
    # Determine if interactive mode should be enabled.
    interactive = not args.no_interactive
    # Choose workflow based on command line arguments.
    if args.use_model:
        classifier.run_inference_only(args.use_model, args.index, interactive)
    else:
        classifier.run_full_pipeline(interactive)

# The big red activation button.
if __name__ == "__main__":
    main()
