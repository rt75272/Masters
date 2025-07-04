# Cat vs Dog Classifier - Modular Implementation

A machine learning project that classifies images of cats and dogs using deep learning. This version features a modular, object-oriented architecture for better maintainability, extensibility, and code organization.

## Project Structure

```
module_06/
├── main.py              # Main entry point - orchestrates the entire workflow
├── cat_vs_dog.py        # Original monolithic implementation (for reference)
├── config.py            # Configuration settings and constants
├── gpu_utils.py         # GPU management and configuration
├── data_loader.py       # Data loading and preprocessing
├── models.py            # Neural network model definitions
├── trainer.py           # Model training and evaluation logic
├── analyzer.py          # Analysis and visualization utilities
├── requirements.txt     # Python dependencies
└── README.md           # This documentation
```

## Architecture Overview

The codebase is organized into several modular components:

### 1. Configuration (`config.py`)
- Centralized configuration management
- Image processing settings
- Training parameters
- Model architecture settings
- Data directory paths
- GPU settings

### 2. GPU Management (`gpu_utils.py`)
- **GPUManager Class**: Handles GPU detection, configuration, and memory management
- Automatic fallback to CPU if GPU is unavailable
- Memory growth configuration to prevent GPU memory allocation issues

### 3. Data Loading (`data_loader.py`)
- **DataLoader Class**: Manages all data loading and preprocessing operations
- Parallel image loading for better performance
- Image normalization and grayscale handling
- Dataset validation and error handling
- Shape analysis for debugging

### 4. Model Building (`models.py`)
- **ModelBuilder Class**: Constructs different neural network architectures
- Dense (fully connected) neural network
- Basic CNN with batch normalization and dropout
- Advanced CNN with extra convolutional layers
- Configurable architecture parameters

### 5. Training (`trainer.py`)
- **ModelTrainer Class**: Handles model training and evaluation
- Data augmentation pipeline
- Advanced callbacks (early stopping, learning rate reduction, checkpointing)
- Training history tracking
- Model saving functionality

### 6. Analysis (`analyzer.py`)
- **ModelAnalyzer Class**: Provides analysis and visualization tools
- Confusion matrix generation
- Prediction threshold analysis
- Image display utilities
- Performance metrics calculation

### 7. Main Orchestrator (`main.py`)
- **CatDogClassifier Class**: Coordinates all components
- Full training pipeline
- Inference-only mode
- Command-line interface
- Error handling and user feedback

## Features

### Model Architectures
1. **Dense Neural Network**: Simple fully connected network
2. **Basic CNN**: Convolutional network with batch normalization
3. **Advanced CNN**: Deeper network with extra convolutional layers

### Training Enhancements
- **Data Augmentation**: Random flips, rotations, zoom, and contrast adjustments
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Reduction**: Adaptive learning rate scheduling
- **Model Checkpointing**: Saves best performing model during training

### Performance Optimizations
- **Parallel Data Loading**: Uses ThreadPoolExecutor for faster image loading
- **GPU Memory Management**: Configures memory growth to prevent allocation issues
- **Batch Processing**: Efficient batch processing with TensorFlow datasets
- **Global Average Pooling**: Reduces overfitting compared to flattening

## Usage

### Training from Scratch
```bash
# Train all models and save the best one
python main.py
```

### Using Pre-trained Model
```bash
# Load a previously trained model for inference
python main.py --use-model conv_model_big.keras
```

### Analyzing Specific Images
```bash
# Predict a specific image by index
python main.py --use-model conv_model_big.keras --index 42
```

### Help
```bash
# Show all available options
python main.py --help
```

## Data Structure

The project expects the following directory structure:

```
data/
├── train/
│   ├── cats/      # Training cat images
│   └── dogs/      # Training dog images
└── test/
    ├── cats/      # Test/validation cat images
    └── dogs/      # Test/validation dog images
```

## Configuration Options

Key configuration parameters in `config.py`:

- `IMG_SIZE`: Image dimensions (default: 128x128)
- `SAMPLE_SIZE`: Training dataset size (default: 4096)
- `VALID_SIZE`: Validation dataset size (default: 1024)
- `EPOCHS`: Maximum training epochs (default: 50)
- `BATCH_SIZE`: Training batch size (default: 32)
- `LEARNING_RATE`: Learning rate for CNN models (default: 5e-5)

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Main dependencies:
- TensorFlow 2.x
- NumPy
- Pillow (PIL)
- scikit-learn
- seaborn
- matplotlib

## Model Performance

The system trains three different models:

1. **Dense Model**: Baseline fully connected network
2. **Basic CNN**: Convolutional network with moderate complexity
3. **Advanced CNN**: Deeper network with extra convolutional blocks

Performance metrics include:
- Training/validation accuracy
- Pearson correlation coefficient
- Confusion matrix
- Threshold analysis for different classification cutoffs

## Error Handling

The modular design includes comprehensive error handling:

- Directory validation before data loading
- GPU configuration error handling
- Model loading error handling
- Index validation for image prediction
- Graceful fallback to CPU if GPU unavailable

## Extensibility

The modular architecture makes it easy to:

- Add new model architectures in `models.py`
- Implement new data augmentation techniques in `trainer.py`
- Add custom analysis tools in `analyzer.py`
- Modify training strategies without affecting other components
- Add new configuration options in `config.py`

## Benefits of Modular Design

1. **Maintainability**: Each component has a single responsibility
2. **Testability**: Individual components can be tested in isolation
3. **Reusability**: Components can be reused in other projects
4. **Extensibility**: Easy to add new features without breaking existing code
5. **Readability**: Clear separation of concerns makes code easier to understand
6. **Debugging**: Issues can be isolated to specific components

## Migration from Monolithic Version

The original monolithic implementation (`cat_vs_dog.py`) has been refactored into the modular structure while maintaining all functionality. The new implementation provides:

- Better code organization
- Improved error handling
- Enhanced extensibility
- Easier maintenance
- Better documentation
- Cleaner interfaces between components

## Future Enhancements

Potential improvements for the modular architecture:

1. **Model Registry**: Add a registry system for managing multiple model types
2. **Experiment Tracking**: Integration with MLflow or similar tools
3. **Configuration Management**: YAML/JSON configuration files
4. **Data Pipeline**: More sophisticated data loading with caching
5. **Model Serving**: Add REST API for model serving
6. **Hyperparameter Tuning**: Integration with optimization frameworks
7. **Logging**: Structured logging with different levels
8. **Testing**: Unit tests for all components
