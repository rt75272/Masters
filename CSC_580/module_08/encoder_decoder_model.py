import numpy as np
import tensorflow as tf
from numpy import array
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense # type: ignore
"""Encoder-Decoder LSTM model for sequence-to-sequence tasks."""

class EncoderDecoderModel:
    """Encoder-Decoder LSTM model for sequence-to-sequence tasks."""
    
    def __init__(self, n_input, n_output, n_units, use_gpu=False):
        """Constructor to initialize the model with given parameters."""
        self.n_input = n_input
        self.n_output = n_output
        self.n_units = n_units
        self.use_gpu = use_gpu
        self.device = '/GPU:0' if use_gpu else '/CPU:0'
        self.train_model = None
        self.encoder_model = None
        self.decoder_model = None
        self._build_models()
    
    def _build_models(self):
        """Builds training, inference encoder, and inference decoder models."""
        with tf.device(self.device):
            encoder_inputs = Input(shape=(None, self.n_input))
            encoder = LSTM(self.n_units, return_state=True)
            encoder_outputs, state_h, state_c = encoder(encoder_inputs)
            encoder_states = [state_h, state_c]
            decoder_inputs = Input(shape=(None, self.n_output))
            decoder_lstm = LSTM(self.n_units, return_sequences=True, return_state=True)
            decoder_outputs, _, _ = decoder_lstm(decoder_inputs, 
                                                 initial_state=encoder_states)
            decoder_dense = Dense(self.n_output, activation='softmax')
            decoder_outputs = decoder_dense(decoder_outputs)
            self.train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
            self.encoder_model = Model(encoder_inputs, encoder_states)
            decoder_state_input_h = Input(shape=(self.n_units,))
            decoder_state_input_c = Input(shape=(self.n_units,))
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_outputs2, state_h2, state_c2 = decoder_lstm(
                decoder_inputs, initial_state=decoder_states_inputs)
            decoder_states2 = [state_h2, state_c2]
            decoder_outputs2 = decoder_dense(decoder_outputs2)
            self.decoder_model = Model(
                [decoder_inputs] + decoder_states_inputs,
                [decoder_outputs2] + decoder_states2)
    
    def compile_model(self, 
                      optimizer='adam', 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy']):
        """Compiles the training model."""
        self.train_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def train(self, X1, X2, y, epochs=30, batch_size=64, verbose=2):
        """Trains the model on the configured device (GPU or CPU)."""
        with tf.device(self.device):
            return self.train_model.fit([X1, X2], y, 
                                        epochs=epochs, 
                                        batch_size=batch_size, 
                                        verbose=verbose)
    
    def predict_sequence(self, source, n_steps, cardinality):
        """Generates target sequence given a source on the configured device."""
        with tf.device(self.device):
            # Encode source as state.
            state = self.encoder_model.predict(source, verbose=0)
            # Start with '0' start token.
            target_seq = np.zeros((1, 1, cardinality))
            output = []
            for _ in range(n_steps):
                yhat, h, c = self.decoder_model.predict([target_seq] + state, verbose=0)
                output.append(yhat[0, 0, :])
                state = [h, c]
                target_seq = yhat # predicted token becomes next input.
            return array(output)
