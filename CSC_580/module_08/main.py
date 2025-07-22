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
NUM_COLUMNS = 88

def display_neural_network_flowchart():
    """Display a comprehensive flowchart of the neural network processing pipeline."""
    print("\n" + "=" * NUM_COLUMNS)
    print("🧠 NEURAL NETWORK PROCESSING FLOWCHART")
    print("=" * NUM_COLUMNS)
    flowchart = """
    📊 DATA GENERATION PHASE
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  🎲 Random Sequence Generator                                                   │
    │  ├─ Generate random integers [0, cardinality-1]                                 │
    │  ├─ Input sequences: 6 timesteps                                                │
    │  ├─ Output sequences: 3 timesteps                                               │
    │  └─ One-hot encode: [batch, timesteps, features]                                │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    🔄 PREPROCESSING PHASE
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  📐 Data Reshaping & Preparation                                                │
    │  ├─ X1 (Encoder Input): [15000, 6, 51] - Input sequences                        │
    │  ├─ X2 (Decoder Input): [15000, 3, 51] - Target sequences (shifted)             │
    │  └─ Training/Validation split: 80/20                                            │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    🧠 ENCODER-DECODER ARCHITECTURE
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  🏗️  ENCODER BRANCH                                                             │
    │  ├─ Input: X1 [batch, 6, 51]                                                    │
    │  ├─ LSTM Layer: 256 units, return_state=True                                    │
    │  ├─ Dropout: 20% (regularization)                                               │
    │  ├─ Batch Normalization                                                         │
    │  └─ Output: Hidden states (h, c) → passed to decoder                            │
    │                                                                                 │
    │  🏗️  DECODER BRANCH                                                             │
    │  ├─ Input: X2 [batch, 3, 51] + Encoder states                                   │
    │  ├─ LSTM Layer: 256 units, return_sequences=True                                │
    │  ├─ Initial state: from encoder (h, c)                                          │
    │  ├─ Dropout: 20% (regularization)                                               │
    │  ├─ Batch Normalization                                                         │
    │  ├─ Dense Layer: 51 units (vocabulary size)                                     │
    │  ├─ Activation: Softmax (probability distribution)                              │
    │  └─ Output: [batch, 3, 51] - Predicted sequences                                │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ⚙️  TRAINING PHASE
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  🎯 Loss Function & Optimization                                                │
    │  ├─ Loss: Categorical Crossentropy                                              │
    │  ├─ Optimizer: Adam (learning_rate=0.001)                                       │
    │  ├─ Metrics: Accuracy                                                           │
    │  ├─ Epochs: 100                                                                 │
    │  ├─ Batch Size: 64                                                              │
    │  └─ Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint                │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    📈 EVALUATION PHASE
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  🔍 Model Assessment                                                            │
    │  ├─ Generate test sequences                                                     │
    │  ├─ Forward pass through trained model                                          │
    │  ├─ Decode predictions to integers                                              │
    │  ├─ Compare with ground truth                                                   │
    │  ├─ Calculate accuracy metrics                                                  │
    │  └─ Generate sample predictions for inspection                                  │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    💾 OUTPUT GENERATION
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  📄 Results & Predictions                                                       │
    │  ├─ Formatted accuracy report                                                   │
    │  ├─ Sample input/output comparisons                                             │
    │  ├─ Model performance metrics                                                   │
    │  └─ Saved predictions to file                                                   │
    └─────────────────────────────────────────────────────────────────────────────────┘

    🔬 TECHNICAL DETAILS:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  Architecture: Encoder-Decoder with Teacher Forcing                             │
    │  Task Type: Sequence-to-Sequence (Many-to-Many)                                 │
    │  Attention: Implicit through state transfer                                     │
    │  Regularization: Dropout + Batch Normalization                                  │
    │  Hardware: GPU-accelerated (CUDA) with CPU fallback                             │
    │  Memory: Dynamic GPU memory growth                                              │
    │  Reproducibility: Fixed random seeds (42)                                       │   
    └─────────────────────────────────────────────────────────────────────────────────┘
    """
    print(flowchart)
    print("=" * NUM_COLUMNS)

def main():
    """Main driver function to run the encoder-decoder sequence-to-sequence model."""
    # Configure GPU before any other operations.
    gpu_available = configure_gpu()
    # Set random seed for reproducibility.
    np.random.seed(42)
    tf.random.set_seed(42)
    # Display system information.
    print("=" * NUM_COLUMNS)
    print("🚀 ENCODER-DECODER SEQUENCE-TO-SEQUENCE MODEL")
    print("=" * NUM_COLUMNS)
    print("📋 System Configuration:")
    print(f" 🔧 TensorFlow version: {tf.__version__}")
    print(f" 🖥️  GPU Available: {len(tf.config.list_physical_devices('GPU'))} device(s)")
    print(f" 🔗 Built with CUDA: {tf.test.is_built_with_cuda()}")
    if tf.config.list_physical_devices('GPU'):
        gpu_name = str(tf.config.list_physical_devices('GPU')[0]).split("name='")[1].split("'")[0]
        print(f"   🎮 GPU Device: {gpu_name}")
    device_type = "GPU" if gpu_available else "CPU"
    print(f"   ⚡ Execution Device: {device_type}")
    print("=" * NUM_COLUMNS)
    n_features = 50 + 1
    n_steps_in = 6
    n_steps_out = 3
    n_units = 256
    n_samples = 15000 
    dropout_rate = 0.2 # Regularization.
    print("\n" + "=" * NUM_COLUMNS)
    print("📊 MODEL CONFIGURATION")
    print("=" * NUM_COLUMNS)
    print(f"   🔢 Vocabulary size: {n_features} classes")
    print(f"   📏 Input sequence length: {n_steps_in}")
    print(f"   📐 Output sequence length: {n_steps_out}")
    print(f"   🧠 LSTM units: {n_units}")
    print(f"   🎯 Training samples: {n_samples:,}")
    print(f"   🛡️  Dropout rate: {dropout_rate}")
    print("=" * NUM_COLUMNS)
    # Display comprehensive neural network flowchart.
    display_neural_network_flowchart()
    # Initialize data generator, model, and trainer.
    data_generator = DataGenerator(cardinality=n_features, random_seed=42)
    model = EncoderDecoderModel(n_features, n_features, n_units, 
                               use_gpu=gpu_available, dropout_rate=dropout_rate)
    trainer = ModelTrainer(model, data_generator)
    print("\n📈 GENERATING DATASET...")
    X1, X2, y = data_generator.get_dataset(n_steps_in, 
                                        n_steps_out, 
                                        n_samples)
    X1 = X1.reshape((n_samples, n_steps_in, n_features))
    X2 = X2.reshape((n_samples, n_steps_out, n_features))
    y = y.reshape((n_samples, n_steps_out, n_features))
    print("🔧 COMPILING MODEL...")
    model.compile_model(optimizer='adam', 
                       loss='categorical_crossentropy', 
                       metrics=['accuracy'],
                       learning_rate=0.001)
    print(f"\n🏋️  TRAINING MODEL ON {device_type}...")
    print("-" * NUM_COLUMNS)
    model.train(X1, X2, y, epochs=100, batch_size=64, verbose=2, use_callbacks=True)
    # Evaluate and save predictions.
    trainer.evaluate_accuracy(n_steps_in, n_steps_out, n_features, n_test_samples=100)
    trainer.save_predictions(n_steps_in, n_steps_out, n_features, 
                           filename='output_predictions.txt', n_samples=10)
    print("\n" + "=" * NUM_COLUMNS)
    print("🎉 TRAINING AND EVALUATION COMPLETE!")
    print("=" * NUM_COLUMNS)
    if not gpu_available:
        print("\n⚠️  Note: Training completed on CPU.")
        print("   For GPU acceleration, ensure CUDA drivers and libraries are installed.")
    else:
        print("=" * NUM_COLUMNS)
        print(f"\n✅ Successfully utilized {device_type} for training.")
    print("=" * NUM_COLUMNS)

# The big red activation button.
if __name__ == '__main__':
    main()
