import random
import numpy as np
from tensorflow.keras.utils import to_categorical # type: ignore
from numpy import array, argmax
from random import randint
"""Data Generator.

Data generation and encoding utilities for sequence-to-sequence tasks.

Usage:
    from data_generator import DataGenerator
    data_generator = DataGenerator(cardinality=51, random_seed=42)
    X1, X2, y = data_generator.get_dataset(n_steps_in=6, n_steps_out=3, n_samples=1000)
"""

class DataGenerator:
    """Handles data generation and encoding for sequence-to-sequence tasks."""
    
    def __init__(self, cardinality=51, random_seed=None):
        """Constructor to initialize DataGenerator with configuration parameters."""
        self.cardinality = cardinality
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
    
    def generate_sequence(self, length):
        """Generates a sequence of random integers (no zeros)."""
        max_value = self.cardinality - 1
        sequence = []
        for _ in range(length):
            random_int = randint(1, max_value)
            sequence.append(random_int)
        return sequence
    
    def get_dataset(self, n_in, n_out, n_samples):
        """Generates n_samples of input/output pairs, one-hot encoded for LSTM."""
        X1, X2, y = list(), list(), list()
        for _ in range(n_samples):
            # Generate random source sequence.
            source = self.generate_sequence(n_in)
            # Target: first n_out of source, reversed.
            target = list(reversed(source[:n_out]))
            # Padded input for decoder, shifted right with '0' start token.
            target_in = [0] + target[:-1]
            # One-hot encoding - remove extra list wrapper.
            src_encoded = to_categorical(source, num_classes=self.cardinality)
            tar_encoded = to_categorical(target, num_classes=self.cardinality)
            tar2_encoded = to_categorical(target_in, num_classes=self.cardinality)
            X1.append(src_encoded)
            X2.append(tar2_encoded)
            y.append(tar_encoded)
        return array(X1), array(X2), array(y)
    
    @staticmethod
    def one_hot_decode(encoded_seq):
        """Decodes a one-hot encoded sequence into a list of plain integers."""
        decoded_sequence = []
        for vector in encoded_seq:
            max_index = argmax(vector)
            # Convert numpy int64 to plain Python int for cleaner display.
            decoded_sequence.append(int(max_index))
        return decoded_sequence
