from numpy import array_equal
"""Training and evaluation utilities for encoder-decoder models."""

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
        total, correct = n_test_samples, 0
        for i in range(total):
            X1_test, X2_test, y_test = self.data_generator.get_dataset(
                n_steps_in, n_steps_out, 1)
            target = self.model.predict_sequence(
                X1_test.reshape(1, n_steps_in, n_features), n_steps_out, n_features)
            expected = self.data_generator.one_hot_decode(y_test[0])
            predicted = self.data_generator.one_hot_decode(target)
            
            # Debug first few samples
            if i < 3:
                source = self.data_generator.one_hot_decode(X1_test[0])
                print(f"Debug {i}: Source={source}, Expected={expected}, Predicted={predicted}")
            
            if array_equal(expected, predicted):
                correct += 1
        accuracy = float(correct) / float(total) * 100.0
        print('Accuracy: %.2f%%' % accuracy)
        return accuracy
    
    def save_predictions(self, 
                    n_steps_in, 
                    n_steps_out, 
                    n_features, 
                    filename='output_predictions.txt', 
                    n_samples=10):
        """Saves sample predictions to file."""
        with open(filename, 'w') as f:
            for _ in range(n_samples):
                X1_test, X2_test, y_test = self.data_generator.get_dataset(
                    n_steps_in, n_steps_out, 1)
                source_seq = self.data_generator.one_hot_decode(X1_test[0])
                target_seq = self.data_generator.one_hot_decode(y_test[0])
                prediction = self.model.predict_sequence(
                    X1_test.reshape(1, n_steps_in, n_features), n_steps_out, n_features)
                predicted_seq = self.data_generator.one_hot_decode(prediction)
                line = f"X={source_seq} y={target_seq} yhat={predicted_seq}\n"
                print(line.strip())
                f.write(line)
