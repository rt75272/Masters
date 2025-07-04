#!/usr/bin/env python3
import os
import random
import argparse
import numpy as np
from tensorflow import keras
# Suppress TensorFlow warnings before importing TF-related modules.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Local module imports - our custom components.
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
        # Initialize core components - each handles a specific aspect of the pipeline
        self.gpu_manager = GPUManager()          # GPU configuration and management
        self.data_loader = DataLoader()          # Data loading and preprocessing
        self.model_builder = ModelBuilder(self.gpu_manager)  # Model architecture
        self.trainer = ModelTrainer()            # Training and evaluation logic
        self.analyzer = ModelAnalyzer()          # Analysis and visualization
        
        # Data placeholders - will be populated during execution
        self.x_train = None    # Training images
        self.y_train = None    # Training labels
        self.x_valid = None    # Validation images
        self.y_valid = None    # Validation labels
        
        # Model placeholders - will hold trained models
        self.dense_model = None    # Simple fully connected model
        self.cnn_model = None      # Basic CNN model
        self.best_model = None     # Best performing model (advanced CNN)
        
    def setup_environment(self):
        """Set up the environment and check GPU availability."""
        # Print header and GPU configuration information
        print("=" * 60)
        print("CAT VS DOG CLASSIFIER - MODULAR VERSION")
        print("=" * 60)
        print("GPU CONFIGURATION CHECK")
        print("=" * 60)
        # Check and display GPU availability - important for performance
        self.gpu_manager.check_gpu_usage()
        print("=" * 60)
        
    def load_data(self):
        """Load and prepare the dataset for training and validation."""
        print("Loading dataset...")
        # First, validate that all required directories exist
        # This prevents cryptic errors later in the pipeline
        if not self.data_loader.validate_data_directories(
                TRAIN_CAT_DIR, 
                TRAIN_DOG_DIR, 
                TEST_CAT_DIR,
                TEST_DOG_DIR):
            return False
        # Load training data - used for model training
        print("Loading training data...")
        self.x_train, self.y_train = self.data_loader.load_dataset(
                TRAIN_CAT_DIR, 
                TRAIN_DOG_DIR, 
                SAMPLE_SIZE)
        # Load validation data - used for model evaluation and testing
        print("Loading validation data...")
        self.x_valid, self.y_valid = self.data_loader.load_dataset(
                TEST_CAT_DIR, 
                TEST_DOG_DIR, 
                VALID_SIZE)
        # Verify that data was loaded successfully
        # Empty arrays indicate a problem with data loading
        if (len(self.x_train) == 0 or len(self.y_train) == 0 or 
            len(self.x_valid) == 0 or len(self.y_valid) == 0):
            print("ERROR: Failed to load training or validation data!")
            return False
        # Display data summary for user verification
        print(f"Training data: {self.x_train.shape[0]} samples")
        print(f"Validation data: {self.x_valid.shape[0]} samples")
        # Analyze image shapes for debugging purposes
        # This helps identify issues with image preprocessing
        print("Analyzing image shapes...")
        self.data_loader.analyze_image_shapes(TRAIN_CAT_DIR, 100)
        return True
    
    def train_models(self):
        """Train different model architectures and compare their performance."""
        # Define input shape for all models (height, width, channels)
        input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
        print("\n" + "=" * 50)
        print("TRAINING PHASE")
        print("=" * 50)
        # Train Dense Model - Simple baseline model
        # This provides a performance baseline for comparison
        print("Training Dense Neural Network...")
        self.dense_model = self.model_builder.build_dense_model(input_shape)
        self.trainer.train_and_evaluate(
            self.dense_model, self.x_train, self.y_train, 
            self.x_valid, self.y_valid, epochs=EPOCHS)
        # Train Basic CNN Model - More sophisticated than dense model
        # Uses convolutional layers to extract spatial features
        print("\nTraining Basic CNN...")
        self.cnn_model = self.model_builder.build_cnn_model(input_shape)
        _, preds = self.trainer.train_and_evaluate(
            self.cnn_model, self.x_train, self.y_train, 
            self.x_valid, self.y_valid, epochs=EPOCHS)
        # Train Advanced CNN Model - Most sophisticated architecture
        # Includes extra convolutional layers for better feature extraction
        print("\nTraining Advanced CNN...")
        self.best_model = self.model_builder.build_cnn_model(input_shape, 
                                                           extra_conv=True)
        _, preds_best = self.trainer.train_and_evaluate(
            self.best_model, self.x_train, self.y_train, 
            self.x_valid, self.y_valid, epochs=EPOCHS)
        # Analyze the best model's predictions using visualization
        # This helps understand model performance and behavior
        print("\nAnalyzing best model predictions...")
        self.analyzer.scatterplot_analysis(preds_best, self.y_valid)
        # Save the best performing model for future use
        # This allows inference without retraining
        self.trainer.save_model(self.best_model, MODEL_SAVE_PATH)
        return self.best_model
    
    def load_pretrained_model(self, model_path):
        """Load a pre-trained model from disk."""
        print(f"Loading pre-trained model from {model_path}...")
        try:
            # Load the saved Keras model
            # This includes architecture, weights, and optimizer state
            self.best_model = keras.models.load_model(model_path)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            # Handle various loading errors (file not found, corrupt file, etc.)
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
        # Generate and display confusion matrix
        # This shows true positives, false positives, etc.
        self.analyzer.generate_confusion_matrix(
            self.best_model, self.x_valid, self.y_valid)
    
    def predict_single_image(self, index=None):
        """Predict and display a single image from the validation set."""
        # Ensure we have both a model and validation data
        if self.best_model is None or self.x_valid is None:
            print("Model or validation data not available!")
            return
        # Choose random index if not specified by user
        # This allows for interactive exploration of the dataset
        if index is None:
            index = random.randint(0, len(self.x_valid) - 1)
        # Validate that the index is within bounds
        # Prevents array index errors
        if index >= len(self.x_valid):
            print(f"Index {index} is out of range (max: {len(self.x_valid) - 1})")
            return
        # Display prediction information
        print(f"\nPredicting image at index {index}...")
        print(f"Validation set shape: {self.x_valid.shape}")
        # Get model prediction for the specified image
        # Returns probability that the image contains a cat
        probability = self.analyzer.get_prediction(self.best_model, 
                                                  self.x_valid, index)
        true_label = self.y_valid[index]
        # Format and display the prediction result
        # Shows both prediction and ground truth for comparison
        result = self.analyzer.format_prediction_result(index, probability, 
                                                       true_label)
        print(result)
        # Display the actual image for visual verification
        # Converts normalized image back to displayable format
        img = self.analyzer.get_image_for_display(self.x_valid, index)
        self.analyzer.display_image_silently(img)
    
    def run_full_pipeline(self):
        """Run the complete training and evaluation pipeline."""
        # Initialize environment and display configuration
        self.setup_environment()
        # Load and prepare training/validation data
        # Exit if data loading fails
        if not self.load_data():
            return
        # Train multiple model architectures and compare performance
        # This provides insights into which approach works best
        self.train_models()
        # Analyze the best model's performance using various metrics
        # Helps understand model strengths and weaknesses
        self.analyze_model_performance()
        # Demonstrate prediction capability on a sample image
        # Provides visual verification of model performance
        self.predict_single_image()
        # Display completion message
        print("\n" + "=" * 50)
        print("PIPELINE COMPLETE")
        print("=" * 50)
    
    def run_inference_only(self, model_path, image_index=None):
        """Run inference only with a pre-trained model (no training)."""
        # Initialize environment and display configuration
        self.setup_environment()
        # Load validation data only - no training data needed
        # This is much faster than loading the full dataset
        print("Loading validation data...")
        self.x_valid, self.y_valid = self.data_loader.load_dataset(
            TEST_CAT_DIR, TEST_DOG_DIR, VALID_SIZE)
        # Verify validation data was loaded successfully
        if len(self.x_valid) == 0 or len(self.y_valid) == 0:
            print("ERROR: Failed to load validation data!")
            return
        # Load the pre-trained model from disk
        # Exit if model loading fails
        if not self.load_pretrained_model(model_path):
            return
        # Analyze model performance on validation set
        # Shows how well the loaded model performs
        self.analyze_model_performance()
        # Predict on specific or random image as requested
        # Allows interactive exploration of model predictions
        self.predict_single_image(image_index)
        # Display completion message
        print("\n" + "=" * 50)
        print("INFERENCE COMPLETE")
        print("=" * 50)

def main():
    """Main entry point - handles command line arguments and workflow selection."""
    # Set up command line argument parser with detailed help
    parser = argparse.ArgumentParser(
        description="Cat vs Dog Classifier - Modular Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python main.py                                   # Train and evaluate models
            python main.py --use-model conv_model_big.keras  # Use pre-trained model
            python main.py --use-model model.keras --index 42 # Predict specific image
                """)
    # Add command line arguments
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
    # Parse command line arguments
    args = parser.parse_args()
    # Create the main classifier instance
    # This initializes all components and sets up the pipeline
    classifier = CatDogClassifier()
    # Choose workflow based on command line arguments
    if args.use_model:
        # Inference-only mode: use pre-trained model without training
        # This is faster and useful for testing/demonstration
        classifier.run_inference_only(args.use_model, args.index)
    else:
        # Full pipeline mode: train models from scratch and evaluate
        # This provides the complete machine learning workflow
        classifier.run_full_pipeline()

# The big red activation button.
if __name__ == "__main__":
    main()
