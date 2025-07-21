import numpy as np
import tensorflow as tf
from numpy import array
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.layers import Input # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.layers import Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
"""Encoder-Decoder model.

Encoder-Decoder LSTM model for sequence-to-sequence tasks.

Usage:
    from encoder_decoder_model import EncoderDecoderModel
    model = EncoderDecoderModel(n_input, 
                         n_output, 
                         n_units, 
                         use_gpu=True, 
                         dropout_rate=0.2)
    model.compile_model(optimizer='adam', 
                         loss='categorical_crossentropy', 
                         metrics=['accuracy'])
    model.train(X1, X2, y, epochs=50, batch_size=64)
"""

class EncoderDecoderModel:
    """Enhanced Encoder-Decoder LSTM model with improved architecture."""
    
    def __init__(self, n_input, n_output, n_units, use_gpu=False, dropout_rate=0.2):
        """Constructor to initialize the model with given parameters."""
        self.n_input = n_input
        self.n_output = n_output
        self.n_units = n_units
        self.use_gpu = use_gpu
        self.dropout_rate = dropout_rate
        self.device = '/GPU:0' if use_gpu else '/CPU:0'
        self.train_model = None
        self.encoder_model = None
        self.decoder_model = None
        self._build_models()
    
    def _build_models(self):
        """Builds enhanced training, inference encoder, and inference decoder models."""
        with tf.device(self.device):
            # Encoder.
            encoder_inputs = Input(shape=(None, self.n_input))
            encoder = LSTM(self.n_units, return_state=True, 
                          dropout=self.dropout_rate, 
                          recurrent_dropout=self.dropout_rate)
            encoder_outputs, state_h, state_c = encoder(encoder_inputs)
            encoder_states = [state_h, state_c]
            # Decoder.
            decoder_inputs = Input(shape=(None, self.n_output))
            decoder_lstm = LSTM(self.n_units, return_sequences=True, return_state=True,
                               dropout=self.dropout_rate, 
                               recurrent_dropout=self.dropout_rate)
            decoder_outputs, _, _ = decoder_lstm(decoder_inputs, 
                                                 initial_state=encoder_states)
            # Add batch normalization and dropout for regularization.
            decoder_outputs = BatchNormalization()(decoder_outputs)
            decoder_outputs = Dropout(self.dropout_rate)(decoder_outputs)
            # Dense layer with softmax.
            decoder_dense = Dense(self.n_output, activation='softmax')
            decoder_outputs = decoder_dense(decoder_outputs)
            # Training model.
            self.train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
            # Inference encoder model.
            self.encoder_model = Model(encoder_inputs, encoder_states)
            # Inference decoder model.
            decoder_state_input_h = Input(shape=(self.n_units,))
            decoder_state_input_c = Input(shape=(self.n_units,))
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_outputs2, state_h2, state_c2 = decoder_lstm(
                decoder_inputs, initial_state=decoder_states_inputs)
            decoder_states2 = [state_h2, state_c2]
            decoder_outputs2 = BatchNormalization()(decoder_outputs2)
            decoder_outputs2 = decoder_dense(decoder_outputs2)
            self.decoder_model = Model(
                [decoder_inputs] + decoder_states_inputs,
                [decoder_outputs2] + decoder_states2)
    
    def compile_model(self, 
                      optimizer='adam', 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'],
                      learning_rate=0.001):
        """Compiles the training model with an optimizer."""
        if optimizer == 'adam':
            optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        self.train_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def get_callbacks(self):
        """Returns callbacks for improved training."""
        early_stopping = EarlyStopping(
            monitor='loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
        return [early_stopping, reduce_lr]
    
    def train(self, X1, X2, y, epochs=30, batch_size=64, verbose=2, use_callbacks=True):
        """Trains the model on the configured device."""
        with tf.device(self.device):
            callbacks = self.get_callbacks() if use_callbacks else None
            return self.train_model.fit([X1, X2], y, 
                                        epochs=epochs, 
                                        batch_size=batch_size, 
                                        verbose=verbose,
                                        callbacks=callbacks,
                                        validation_split=0.1)
    
    def predict_sequence(self, source, n_steps, cardinality):
        """Generates target sequence given a source on the configured device."""
        with tf.device(self.device):
            state = self.encoder_model.predict(source, verbose=0)
            target_seq = np.zeros((1, 1, cardinality))
            output = []
            for _ in range(n_steps):
                yhat, h, c = self.decoder_model.predict([target_seq] + state, verbose=0)
                output.append(yhat[0, 0, :])
                state = [h, c]
                target_seq = yhat # Predicted token becomes next input.
            return array(output)
