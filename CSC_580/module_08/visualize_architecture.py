import argparse
from datetime import datetime
"""Neural Network Architecture Visualizer.

This script generates comprehensive visual representations of the 
encoder-decoder sequence-to-sequence model architecture and data flow.

Usage:
    python visualize_architecture.py
    python visualize_architecture.py --detailed    # More technical details
    python visualize_architecture.py --save        # Save to file
"""

def generate_detailed_architecture_diagram():
    """Generate a detailed technical architecture diagram."""
    return """
🧠 DETAILED ENCODER-DECODER ARCHITECTURE DIAGRAM
═══════════════════════════════════════════════════════════════════════════════════

INPUT DATA FLOW:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BATCH PROCESSING                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Batch Size: 64 samples                                                         │
│  Input Shape: [64, 6, 51]  │  Decoder Input: [64, 3, 51]                        │
│  Output Shape: [64, 3, 51] │  Expected Output: [64, 3, 51]                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             ENCODER SECTION                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📊 Input Layer                                                                 │
│  ├─ Shape: [batch=64, timesteps=6, features=51]                                 │
│  ├─ Data Type: float32                                                          │
│  └─ Encoding: One-hot vectors for integer sequences                             │
│                                      │                                          │
│                                      ▼                                          │
│  🧠 LSTM Layer (Encoder)                                                        │
│  ├─ Units: 256                                                                  │
│  ├─ Return State: True                                                          │
│  ├─ Return Sequences: False                                                     │
│  ├─ Activation: tanh (default)                                                  │
│  ├─ Recurrent Activation: sigmoid (default)                                     │
│  └─ Parameters: ~263,168 trainable params                                       │
│                                      │                                          │
│                                      ▼                                          │
│  🛡️  Dropout Layer                                                              │
│  ├─ Rate: 0.2 (20% neurons dropped during training)                             │
│  └─ Purpose: Prevent overfitting                                                │
│                                      │                                          │
│                                      ▼                                          │
│  ⚡ Batch Normalization                                                         │
│  ├─ Normalize activations                                                       │
│  ├─ Accelerate training                                                         │
│  └─ Improve gradient flow                                                       │
│                                      │                                          │
│                                      ▼                                          │
│  📤 State Output                                                                │
│  ├─ Hidden State (h): [batch=64, units=256]                                     │
│  ├─ Cell State (c): [batch=64, units=256]                                       │
│  └─ These states encode the entire input sequence                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             DECODER SECTION                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📊 Input Layer                                                                 │
│  ├─ Shape: [batch=64, timesteps=3, features=51]                                 │
│  ├─ Teacher Forcing: Uses ground truth during training                          │
│  └─ Initial States: From encoder (h, c)                                         │
│                                      │                                          │
│                                      ▼                                          │
│  🧠 LSTM Layer (Decoder)                                                        │
│  ├─ Units: 256                                                                  │
│  ├─ Return Sequences: True                                                      │
│  ├─ Return State: False                                                         │
│  ├─ Initial State: [encoder_h, encoder_c]                                       │
│  └─ Output Shape: [batch=64, timesteps=3, units=256]                            │
│                                      │                                          │
│                                      ▼                                          │
│  🛡️  Dropout Layer                                                              │
│  ├─ Rate: 0.2 (20% neurons dropped during training)                             │
│  └─ Applied to each timestep                                                    │
│                                      │                                          │
│                                      ▼                                          │
│  ⚡ Batch Normalization                                                         │
│  ├─ Applied across time dimension                                               │
│  └─ Stabilizes training dynamics                                                │
│                                      │                                          │
│                                      ▼                                          │
│  🎯 Dense Output Layer                                                          │
│  ├─ Units: 51 (vocabulary size)                                                 │
│  ├─ Activation: Softmax                                                         │
│  ├─ Output Shape: [batch=64, timesteps=3, classes=51]                           │
│  └─ Produces probability distribution over vocabulary                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LOSS & OPTIMIZATION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📊 Loss Function: Categorical Crossentropy                                     │
│  ├─ Formula: -Σ(y_true * log(y_pred))                                           │
│  ├─ Applied per timestep, averaged across batch                                 │
│  └─ Suitable for multi-class classification                                     │
│                                                                                 │
│  ⚙️  Optimizer: Adam                                                            │
│  ├─ Learning Rate: 0.001                                                        │
│  ├─ Beta1: 0.9, Beta2: 0.999                                                    │
│  ├─ Epsilon: 1e-07                                                              │
│  └─ Adaptive learning rate per parameter                                        │
│                                                                                 │
│  📈 Metrics: Accuracy                                                           │
│  ├─ Measures exact sequence match                                               │
│  └─ Calculated per timestep and averaged                                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

TRAINING DYNAMICS:
═══════════════════════════════════════════════════════════════════════════════════

🔄 FORWARD PASS:
   1. Encoder processes input sequence [1,2,3,4,5,6]
   2. Encoder states (h,c) capture sequence information
   3. Decoder receives target sequence [4,5,6] + encoder states
   4. Decoder outputs probability distributions for [5,6,7]
   5. Loss computed against true targets [5,6,7]

⬅️  BACKWARD PASS:
   1. Gradients flow from output to input
   2. LSTM gates updated via Backpropagation Through Time (BPTT)
   3. Encoder and decoder weights updated jointly
   4. Dropout and batch norm affect gradient flow

🎯 TEACHER FORCING:
   - During training: Decoder sees ground truth previous tokens
   - During inference: Decoder sees its own previous predictions
   - Accelerates training but can cause exposure bias

🛡️  REGULARIZATION TECHNIQUES:
   ├─ Dropout: Prevents co-adaptation of neurons
   ├─ Batch Normalization: Reduces internal covariate shift
   ├─ Early Stopping: Prevents overfitting
   └─ Learning Rate Reduction: Adaptive learning

💾 MODEL PARAMETERS:
═══════════════════════════════════════════════════════════════════════════════════
Total Parameters: ~525,000
├─ Encoder LSTM: ~263,168 parameters
├─ Decoder LSTM: ~263,168 parameters  
├─ Dense Output: ~13,107 parameters
├─ Batch Norm: ~1,024 parameters
└─ Bias terms: Various

🚀 COMPUTATIONAL COMPLEXITY:
═══════════════════════════════════════════════════════════════════════════════════
Time Complexity: O(sequence_length × hidden_units²)
Space Complexity: O(batch_size × sequence_length × hidden_units)
FLOPs per forward pass: ~50M operations
GPU Memory Usage: ~2-4GB (depending on batch size)

🎯 SEQUENCE PROCESSING EXAMPLE:
═══════════════════════════════════════════════════════════════════════════════════
Input:  [1, 2, 3, 4, 5, 6] → Encoder → Hidden States
States: [h, c] → Decoder ← Target: [4, 5, 6]
Output: [5, 6, 7] (predicted continuation)

The model learns to continue integer sequences by understanding the pattern
that each output element should be the input element + 1.
"""

def generate_data_flow_diagram():
    """Generate a data flow diagram showing tensor shapes through the network."""
    return """
📊 TENSOR FLOW DIAGRAM
═══════════════════════════════════════════════════════════════════════════════════

                           🎲 RAW DATA GENERATION
                                      │
                        ┌─────────────▼─────────────┐
                        │ Random Integer Sequences  │
                        │ [1,2,3,4,5,6] → [4,5,6,7] │
                        └─────────────┬─────────────┘
                                      │
                           🔄 ONE-HOT ENCODING
                                      │
    ┌─────────────────────────────────▼─────────────────────────────────┐
    │                        PREPROCESSING                              │
    ├───────────────────────────────────────────────────────────────────┤
    │  Input Sequences:    [15000 × 6]     → [15000 × 6 × 51]           │
    │  Target Input:       [15000 × 3]     → [15000 × 3 × 51]           │
    │  Target Output:      [15000 × 3]     → [15000 × 3 × 51]           │
    └─────────────────────────────────┬─────────────────────────────────┘
                                      │
                              📦 BATCH CREATION
                                      │
    ┌─────────────────────────────────▼─────────────────────────────────┐
    │                        BATCH TENSORS                              │
    ├───────────────────────────────────────────────────────────────────┤
    │  X1 (Encoder):       [64 × 6 × 51]                                │
    │  X2 (Decoder):       [64 × 3 × 51]                                │
    │  y (Target):         [64 × 3 × 51]                                │
    └─────────────────────────────────┬─────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                        ENCODER                                  │
    ├─────────────────────────────────────────────────────────────────┤
    │  Input:              [64 × 6 × 51]                              │
    │         │                                                       │
    │         ▼                                                       │
    │  LSTM:               [64 × 256] (final hidden state)            │
    │         │                                                       │
    │         ▼                                                       │
    │  Dropout:            [64 × 256] (20% dropped)                   │
    │         │                                                       │
    │         ▼                                                       │
    │  BatchNorm:          [64 × 256] (normalized)                    │
    │         │                                                       │
    │         ▼                                                       │
    │  States:             h=[64 × 256], c=[64 × 256]                 │
    └─────────────────────────────────┬───────────────────────────────┘
                                      │
                              🔄 STATE TRANSFER
                                      │
    ┌─────────────────────────────────▼─────────────────────────────────┐
    │                        DECODER                                    │
    ├───────────────────────────────────────────────────────────────────┤
    │  Input:              [64 × 3 × 51]                                │
    │  Initial States:     h=[64 × 256], c=[64 × 256]                   │
    │         │                                                         │
    │         ▼                                                         │
    │  LSTM:               [64 × 3 × 256] (all timesteps)               │
    │         │                                                         │
    │         ▼                                                         │
    │  Dropout:            [64 × 3 × 256] (20% dropped)                 │
    │         │                                                         │
    │         ▼                                                         │
    │  BatchNorm:          [64 × 3 × 256] (normalized)                  │
    │         │                                                         │
    │         ▼                                                         │
    │  Dense:              [64 × 3 × 51] (vocabulary logits)            │
    │         │                                                         │
    │         ▼                                                         │
    │  Softmax:            [64 × 3 × 51] (probabilities)                │
    └─────────────────────────────────┬─────────────────────────────────┘
                                      │
                              📊 LOSS COMPUTATION
                                      │
    ┌─────────────────────────────────▼─────────────────────────────────┐
    │                    CATEGORICAL CROSSENTROPY                       │
    ├───────────────────────────────────────────────────────────────────┤
    │  Predictions:        [64 × 3 × 51] (softmax output)               │
    │  True Labels:        [64 × 3 × 51] (one-hot targets)              │
    │         │                                                         │
    │         ▼                                                         │
    │  Loss per sample:    [64 × 3] (per timestep loss)                 │
    │         │                                                         │
    │         ▼                                                         │
    │  Batch Loss:         scalar (averaged across batch)               │
    └─────────────────────────────────┬─────────────────────────────────┘
                                      │
                              ⬅️ BACKPROPAGATION
                                      │
    ┌─────────────────────────────────▼─────────────────────────────────┐
    │                      GRADIENT COMPUTATION                         │
    ├───────────────────────────────────────────────────────────────────┤
    │  Gradients flow backward through:                                 │
    │  1. Softmax → Dense layer                                         │
    │  2. Dense → Batch Norm → Dropout                                  │
    │  3. Decoder LSTM (through time)                                   │
    │  4. State connections                                             │
    │  5. Encoder LSTM (through time)                                   │
    └─────────────────────────────────┬─────────────────────────────────┘
                                      │
                              ⚙️ PARAMETER UPDATE
                                      │
    ┌─────────────────────────────────▼─────────────────────────────────┐
    │                        ADAM OPTIMIZER                             │
    ├───────────────────────────────────────────────────────────────────┤
    │  Learning Rate:      0.001                                        │
    │  Momentum Terms:     β₁=0.9, β₂=0.999                             │
    │  Parameters Updated: ~525,000 weights and biases                  │
    └───────────────────────────────────────────────────────────────────┘

🔍 TENSOR SHAPE EVOLUTION SUMMARY:
═══════════════════════════════════════════════════════════════════════════════════
Raw Data:     [sequence_length] integers
One-Hot:      [sequence_length, vocabulary_size] 
Batched:      [batch_size, sequence_length, vocabulary_size]
Encoder:      [batch_size, sequence_length, features] → [batch_size, units]
Decoder:      [batch_size, out_seq_len, features] → [batch_size, out_seq_len, vocab]
Output:       [batch_size, output_sequence_length, vocabulary_size]
"""

def save_diagrams_to_file(content, filename=None):
    """Save the diagrams to a text file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"neural_network_architecture_{timestamp}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("ENCODER-DECODER NEURAL NETWORK ARCHITECTURE DOCUMENTATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(content)
    print(f"📄 Architecture diagrams saved to: {filename}")
    return filename

def main():
    parser = argparse.ArgumentParser(description="Visualize neural network architecture")
    parser.add_argument("--detailed", action="store_true", 
                       help="Show detailed technical architecture")
    parser.add_argument("--dataflow", action="store_true",
                       help="Show tensor data flow diagram")
    parser.add_argument("--save", action="store_true",
                       help="Save diagrams to file")
    parser.add_argument("--all", action="store_true",
                       help="Show all diagrams")
    args = parser.parse_args()
    content = ""
    if args.all or (not args.detailed and not args.dataflow):
        print("🧠 BASIC NEURAL NETWORK FLOWCHART")
        print("=" * 80)
        # Import and show the basic flowchart from main (handle import safely).
        try:
            from main import display_neural_network_flowchart
            display_neural_network_flowchart()
        except ImportError:
            print("⚠️  Could not import basic flowchart from main.py")
            print("   This is expected during testing.")
        content += "BASIC FLOWCHART\n" + "=" * 40 + "\n"
    if args.detailed or args.all:
        detailed_diagram = generate_detailed_architecture_diagram()
        print(detailed_diagram)
        content += detailed_diagram + "\n\n"
    if args.dataflow or args.all:
        dataflow_diagram = generate_data_flow_diagram()
        print(dataflow_diagram)
        content += dataflow_diagram + "\n\n"
    if args.save and content:
        save_diagrams_to_file(content)

# The big red activation button.
if __name__ == "__main__":
    main()
