"""Comprehensive test suite for the encoder-decoder sequence-to-sequence model.

This module contains pytest test cases for all components of the encoder-decoder model:
- DataGenerator: data generation and encoding functionality
- EncoderDecoderModel: model architecture and training
- ModelTrainer: training and evaluation utilities
- GPU configuration and integration tests

Usage:
    $ pytest test_encoder_decoder.py -v
    $ pytest test_encoder_decoder.py::TestDataGenerator -v
    $ pytest test_encoder_decoder.py --cov=. --cov-report=html
"""

import pytest
import numpy as np
import tensorflow as tf
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import the modules to test
from data_generator import DataGenerator
from encoder_decoder_model import EncoderDecoderModel
from model_trainer import ModelTrainer
from gpu_config import configure_gpu


class TestDataGenerator:
    """Test cases for DataGenerator class."""
    
    @pytest.fixture
    def data_generator(self):
        """Create a DataGenerator instance for testing."""
        return DataGenerator(cardinality=51, random_seed=42)
    
    def test_init(self):
        """Test DataGenerator initialization."""
        dg = DataGenerator(cardinality=100, random_seed=123)
        assert dg.cardinality == 100
        
        # Test without random seed
        dg2 = DataGenerator(cardinality=50)
        assert dg2.cardinality == 50
    
    def test_generate_sequence(self, data_generator):
        """Test sequence generation."""
        sequence = data_generator.generate_sequence(length=6)
        
        # Check sequence length
        assert len(sequence) == 6
        
        # Check all values are within valid range (1 to cardinality-1)
        assert all(1 <= val <= 50 for val in sequence)
        
        # Check reproducibility with same seed
        dg2 = DataGenerator(cardinality=51, random_seed=42)
        sequence2 = dg2.generate_sequence(length=6)
        assert sequence == sequence2
    
    def test_generate_sequence_different_lengths(self, data_generator):
        """Test sequence generation with different lengths."""
        for length in [1, 3, 5, 10]:
            sequence = data_generator.generate_sequence(length)
            assert len(sequence) == length
            assert all(1 <= val <= 50 for val in sequence)
    
    def test_get_dataset_shape(self, data_generator):
        """Test dataset generation shapes."""
        n_in, n_out, n_samples = 6, 3, 5
        X1, X2, y = data_generator.get_dataset(n_in, n_out, n_samples)
        
        # Check shapes
        assert X1.shape == (n_samples, n_in, 51)
        assert X2.shape == (n_samples, n_out, 51)
        assert y.shape == (n_samples, n_out, 51)
    
    def test_get_dataset_logic(self, data_generator):
        """Test dataset generation logic (reverse first n_out elements)."""
        n_in, n_out, n_samples = 6, 3, 1
        X1, X2, y = data_generator.get_dataset(n_in, n_out, n_samples)
        
        # Decode the sequences
        source = data_generator.one_hot_decode(X1[0])
        target = data_generator.one_hot_decode(y[0])
        target_input = data_generator.one_hot_decode(X2[0])
        
        # Check that target is first 3 elements of source, reversed
        expected_target = list(reversed(source[:3]))
        assert target == expected_target
        
        # Check that target_input is [0] + target[:-1]
        expected_target_input = [0] + target[:-1]
        assert target_input == expected_target_input
    
    def test_one_hot_decode(self, data_generator):
        """Test one-hot decoding."""
        # Create a simple one-hot encoded sequence
        encoded = np.array([
            [0, 1, 0, 0, 0],  # index 1
            [0, 0, 0, 1, 0],  # index 3
            [1, 0, 0, 0, 0]   # index 0
        ])
        
        decoded = data_generator.one_hot_decode(encoded)
        assert decoded == [1, 3, 0]
        assert all(isinstance(x, int) for x in decoded)  # Should be plain ints
    
    def test_dataset_reproducibility(self):
        """Test that datasets are reproducible with same seed."""
        # Reset random state to ensure clean test
        import random
        random.seed(42)
        np.random.seed(42)
        
        dg1 = DataGenerator(cardinality=51, random_seed=42)
        X1_1, X2_1, y_1 = dg1.get_dataset(6, 3, 5)
        
        # Reset again for second generator
        random.seed(42) 
        np.random.seed(42)
        
        dg2 = DataGenerator(cardinality=51, random_seed=42)
        X1_2, X2_2, y_2 = dg2.get_dataset(6, 3, 5)
        
        np.testing.assert_array_equal(X1_1, X1_2)
        np.testing.assert_array_equal(X2_1, X2_2)
        np.testing.assert_array_equal(y_1, y_2)


class TestEncoderDecoderModel:
    """Test cases for EncoderDecoderModel class."""
    
    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        return EncoderDecoderModel(n_input=10, n_output=10, n_units=32, 
                                 use_gpu=False, dropout_rate=0.1)
    
    def test_init(self, model):
        """Test model initialization."""
        assert model.n_input == 10
        assert model.n_output == 10
        assert model.n_units == 32
        assert model.dropout_rate == 0.1
        assert model.use_gpu == False
        assert model.device == '/CPU:0'
        
        # Check that models are built
        assert model.train_model is not None
        assert model.encoder_model is not None
        assert model.decoder_model is not None
    
    def test_model_architecture(self, model):
        """Test model architecture."""
        # Check input shapes
        train_inputs = model.train_model.inputs
        assert len(train_inputs) == 2  # encoder_input, decoder_input
        assert train_inputs[0].shape[2] == 10  # n_input
        assert train_inputs[1].shape[2] == 10  # n_output
        
        # Check output shape
        train_output = model.train_model.outputs[0]
        assert train_output.shape[2] == 10  # n_output
    
    def test_compile_model(self, model):
        """Test model compilation."""
        model.compile_model(optimizer='adam', 
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        
        # Check that model is compiled
        assert model.train_model.optimizer is not None
        assert model.train_model.loss == 'categorical_crossentropy'
    
    def test_gpu_cpu_device_selection(self):
        """Test GPU/CPU device selection."""
        cpu_model = EncoderDecoderModel(10, 10, 32, use_gpu=False)
        assert cpu_model.device == '/CPU:0'
        
        gpu_model = EncoderDecoderModel(10, 10, 32, use_gpu=True)
        assert gpu_model.device == '/GPU:0'
    
    def test_predict_sequence_shape(self, model):
        """Test prediction sequence shape."""
        model.compile_model()
        
        # Create dummy input
        source = np.random.random((1, 6, 10))
        prediction = model.predict_sequence(source, n_steps=3, cardinality=10)
        
        assert prediction.shape == (3, 10)
    
    def test_training_interface(self, model):
        """Test training interface."""
        model.compile_model()
        
        # Create small dummy dataset
        n_samples = 10
        X1 = np.random.random((n_samples, 6, 10))
        X2 = np.random.random((n_samples, 3, 10))
        y = np.random.random((n_samples, 3, 10))
        
        # Test training (should not raise an error)
        history = model.train(X1, X2, y, epochs=1, batch_size=5, 
                            verbose=0, use_callbacks=False)
        
        assert hasattr(history, 'history')
        assert 'loss' in history.history


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    @pytest.fixture
    def setup_trainer(self):
        """Set up trainer with small model and data generator."""
        data_generator = DataGenerator(cardinality=10, random_seed=42)
        model = EncoderDecoderModel(10, 10, 32, use_gpu=False)
        model.compile_model()
        trainer = ModelTrainer(model, data_generator)
        return trainer, data_generator, model
    
    def test_init(self, setup_trainer):
        """Test ModelTrainer initialization."""
        trainer, data_generator, model = setup_trainer
        assert trainer.model == model
        assert trainer.data_generator == data_generator
    
    def test_evaluate_accuracy_format(self, setup_trainer, capsys):
        """Test evaluation accuracy output format."""
        trainer, _, _ = setup_trainer
        
        # Run evaluation with small sample
        accuracy = trainer.evaluate_accuracy(6, 3, 10, n_test_samples=3)
        
        # Check that accuracy is returned
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 100
        
        # Check console output
        captured = capsys.readouterr()
        assert "MODEL EVALUATION" in captured.out
        assert "Sample Predictions:" in captured.out
        assert "Accuracy:" in captured.out
    
    def test_save_predictions_file(self, setup_trainer):
        """Test prediction saving to file."""
        trainer, _, _ = setup_trainer
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_filename = f.name
        
        try:
            trainer.save_predictions(6, 3, 10, filename=temp_filename, n_samples=3)
            
            # Check file was created and has content
            assert os.path.exists(temp_filename)
            with open(temp_filename, 'r') as f:
                content = f.read()
                assert "ENCODER-DECODER MODEL PREDICTIONS" in content
                assert "Sample 1:" in content
                assert "Input:" in content
                assert "Expected:" in content
                assert "Predicted:" in content
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def test_save_predictions_console_output(self, setup_trainer, capsys):
        """Test prediction saving console output."""
        trainer, _, _ = setup_trainer
        
        trainer.save_predictions(6, 3, 10, filename='test_predictions.txt', n_samples=2)
        
        captured = capsys.readouterr()
        assert "SAVING SAMPLE PREDICTIONS" in captured.out
        assert "Input:" in captured.out
        assert "Expected:" in captured.out
        assert "Predicted:" in captured.out
        
        # Clean up
        if os.path.exists('test_predictions.txt'):
            os.unlink('test_predictions.txt')


class TestGPUConfig:
    """Test cases for GPU configuration."""
    
    @patch('tensorflow.config.experimental.list_physical_devices')
    def test_configure_gpu_available(self, mock_list_devices):
        """Test GPU configuration when GPU is available."""
        # Mock GPU device
        mock_gpu = MagicMock()
        mock_gpu.name = 'GPU:0'
        mock_list_devices.return_value = [mock_gpu]
        
        with patch('tensorflow.config.experimental.set_memory_growth') as mock_set_memory, \
             patch('tensorflow.config.experimental.set_visible_devices') as mock_set_visible:
            
            result = configure_gpu()
            
            assert result == True
            mock_set_memory.assert_called_once_with(mock_gpu, True)
            mock_set_visible.assert_called_once_with(mock_gpu, 'GPU')
    
    @patch('tensorflow.config.experimental.list_physical_devices')
    def test_configure_gpu_not_available(self, mock_list_devices):
        """Test GPU configuration when GPU is not available."""
        mock_list_devices.return_value = []
        
        result = configure_gpu()
        assert result == False
    
    @patch('tensorflow.config.experimental.list_physical_devices')
    @patch('tensorflow.config.experimental.set_memory_growth')
    def test_configure_gpu_runtime_error(self, mock_set_memory, mock_list_devices):
        """Test GPU configuration with runtime error."""
        mock_gpu = MagicMock()
        mock_list_devices.return_value = [mock_gpu]
        mock_set_memory.side_effect = RuntimeError("GPU error")
        
        result = configure_gpu()
        assert result == False


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_full_workflow_small(self):
        """Test complete workflow with small dataset."""
        # Initialize components
        data_generator = DataGenerator(cardinality=10, random_seed=42)
        model = EncoderDecoderModel(10, 10, 32, use_gpu=False)
        trainer = ModelTrainer(model, data_generator)
        
        # Generate small dataset
        X1, X2, y = data_generator.get_dataset(6, 3, 20)
        X1 = X1.reshape((20, 6, 10))
        X2 = X2.reshape((20, 3, 10))
        y = y.reshape((20, 3, 10))
        
        # Compile and train briefly
        model.compile_model()
        history = model.train(X1, X2, y, epochs=1, batch_size=5, 
                            verbose=0, use_callbacks=False)
        
        # Evaluate
        accuracy = trainer.evaluate_accuracy(6, 3, 10, n_test_samples=3)
        
        # Check that everything runs without errors
        assert isinstance(history.history['loss'], list)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 100
    
    def test_sequence_reversal_logic(self):
        """Test that the model learns the sequence reversal task."""
        # Create a very simple case to verify logic
        data_generator = DataGenerator(cardinality=5, random_seed=42)
        
        # Generate one sample and check the logic
        X1, X2, y = data_generator.get_dataset(6, 3, 1)
        
        source = data_generator.one_hot_decode(X1[0])
        target = data_generator.one_hot_decode(y[0])
        
        # Verify that target is first 3 elements of source, reversed
        expected = list(reversed(source[:3]))
        assert target == expected
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seeds."""
        # Set seeds
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Run workflow twice
        results1 = self._run_small_workflow()
        
        np.random.seed(42)
        tf.random.set_seed(42)
        results2 = self._run_small_workflow()
        
        # Compare results (should be identical)
        np.testing.assert_array_equal(results1['X1'], results2['X1'])
        np.testing.assert_array_equal(results1['y'], results2['y'])
    
    def _run_small_workflow(self):
        """Helper method for reproducibility testing."""
        data_generator = DataGenerator(cardinality=10, random_seed=42)
        X1, X2, y = data_generator.get_dataset(6, 3, 5)
        return {'X1': X1, 'X2': X2, 'y': y}


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_cardinality(self):
        """Test with invalid cardinality values."""
        with pytest.raises((ValueError, IndexError)):
            dg = DataGenerator(cardinality=1)  # Should be > 1
            dg.generate_sequence(5)
    
    def test_zero_length_sequence(self):
        """Test with zero-length sequence."""
        dg = DataGenerator(cardinality=10)
        sequence = dg.generate_sequence(0)
        assert len(sequence) == 0
    
    def test_large_sequence_dimensions(self):
        """Test with larger sequence dimensions."""
        dg = DataGenerator(cardinality=100, random_seed=42)
        X1, X2, y = dg.get_dataset(n_in=20, n_out=10, n_samples=5)
        
        assert X1.shape == (5, 20, 100)
        assert X2.shape == (5, 10, 100)
        assert y.shape == (5, 10, 100)
    
    def test_model_with_different_input_output_sizes(self):
        """Test model with different input and output vocabulary sizes."""
        model = EncoderDecoderModel(n_input=50, n_output=30, n_units=64)
        
        # Should initialize without errors
        assert model.n_input == 50
        assert model.n_output == 30
        assert model.train_model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
