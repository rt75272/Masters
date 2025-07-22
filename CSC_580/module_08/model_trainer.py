from numpy import array_equal
"""Model Trainer.

Training and evaluation utilities for encoder-decoder models.

Usage:
    from model_trainer import ModelTrainer
    trainer = ModelTrainer(model, data_generator)
    trainer.evaluate_accuracy(n_steps_in, 
                              n_steps_out, 
                              n_features, 
                              n_test_samples=100)
    trainer.save_predictions(n_steps_in, 
                              n_steps_out, 
                              n_features, 
                              filename='output_predictions.txt', 
                              n_samples=10)
""" 
NUM_COLUMNS = 88 

class ModelTrainer:
    """Handles training and evaluation of the encoder-decoder model."""
   
    def __init__(self, model, data_generator):
        """Constructor to initialize with model and data generator."""
        self.model = model
        self.data_generator = data_generator
    
    def evaluate_accuracy(self, 
                    n_steps_in, 
                    n_steps_out, 
                    n_features, 
                    n_test_samples=100):
        """Evaluates model accuracy on test sequences with detailed diagnostics."""
        print("\n" + "=" * NUM_COLUMNS)
        print("🧠 MODEL EVALUATION")
        print("=" * NUM_COLUMNS)
        print(f"📊 Testing on {n_test_samples} sequences...")
        print(f"🔢 Input length: {n_steps_in}, Output length: {n_steps_out}")
        print("=" * NUM_COLUMNS)
        total, correct = n_test_samples, 0
        sample_results = []
        for i in range(total):
            X1_test, X2_test, y_test = self.data_generator.get_dataset(
                n_steps_in, n_steps_out, 1)
            target = self.model.predict_sequence(
                X1_test.reshape(1, n_steps_in, n_features), n_steps_out, n_features)
            expected = self.data_generator.one_hot_decode(y_test[0])
            predicted = self.data_generator.one_hot_decode(target)
            is_correct = array_equal(expected, predicted)
            if is_correct:
                correct += 1
            # Store first 5 samples for detailed display.
            if i < 5:
                source = self.data_generator.one_hot_decode(X1_test[0])
                sample_results.append({
                    'source': source,
                    'expected': expected,
                    'predicted': predicted,
                    'correct': is_correct})
        # Display sample results.
        print("📝 Sample Predictions:")
        for i, result in enumerate(sample_results):
            status = "✅ CORRECT" if result['correct'] else "❌ WRONG"
            print(f"\n  {i+1}. Input:     {result['source']}")
            print(f"     Expected:  {result['expected']}")
            print(f"     Predicted: {result['predicted']} {status}")
            print("")
        # Final accuracy.
        accuracy = float(correct) / float(total) * 100.0
        print("=" * NUM_COLUMNS)
        if accuracy >= 95:
            print(f"🎉 EXCELLENT! Accuracy: {accuracy:.2f}% ({correct}/{total})")
        elif accuracy >= 80:
            print(f"👍 GOOD! Accuracy: {accuracy:.2f}% ({correct}/{total})")
        elif accuracy >= NUM_COLUMNS:
            print(f"⚠️  FAIR. Accuracy: {accuracy:.2f}% ({correct}/{total})")
        else:
            print(f"❌ POOR. Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print("=" * NUM_COLUMNS)
        return accuracy
    
    def save_predictions(self, 
                    n_steps_in, 
                    n_steps_out, 
                    n_features, 
                    filename='output_predictions.txt', 
                    n_samples=10):
        """Saves sample predictions to file with enhanced formatting."""
        print("\n" + "=" * NUM_COLUMNS)
        print("💾 SAVING SAMPLE PREDICTIONS")
        print("=" * NUM_COLUMNS)
        print(f"📁 Saving {n_samples} predictions to: {filename}")
        print("=" * NUM_COLUMNS)
        with open(filename, 'w') as f:
            # Write header to file.
            f.write("=" * 80 + "\n")
            f.write("ENCODER-DECODER MODEL PREDICTIONS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Task: Reverse first {n_steps_out} elements of {n_steps_in}-element sequence\n")
            f.write("=" * 80 + "\n\n")
            for i in range(n_samples):
                X1_test, X2_test, y_test = self.data_generator.get_dataset(
                    n_steps_in, n_steps_out, 1)
                source_seq = self.data_generator.one_hot_decode(X1_test[0])
                target_seq = self.data_generator.one_hot_decode(y_test[0])
                prediction = self.model.predict_sequence(
                    X1_test.reshape(1, n_steps_in, n_features), n_steps_out, n_features)
                predicted_seq = self.data_generator.one_hot_decode(prediction)
                # Check if prediction is correct.
                is_correct = target_seq == predicted_seq
                status = "✅ CORRECT" if is_correct else "❌ WRONG"
                # Console output.
                print(f"  {i+1:2d}. Input:     {source_seq}")
                print(f"      Expected:  {target_seq}")
                print(f"      Predicted: {predicted_seq} {status}")
                # File output.
                f.write(f"Sample {i+1}:\n")
                f.write(f"  Input:     {source_seq}\n")
                f.write(f"  Expected:  {target_seq}\n")
                f.write(f"  Predicted: {predicted_seq}\n")
                f.write(f"  Result:    {status}\n\n")
        print("=" * NUM_COLUMNS)
        print(f"✅ Predictions saved successfully to {filename}")
        print("=" * NUM_COLUMNS)
