import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import GRU, Dropout, Dense, Flatten, Layer
from keras.optimizers import Adam
from keras.metrics import MeanAbsolutePercentageError, RootMeanSquaredError
from .base_model import BaseModel

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.w = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, x):
        et = tf.squeeze(tf.tanh(tf.matmul(x, self.w) + self.b), axis=-1)
        at = tf.nn.softmax(et, axis=-1)
        at = tf.expand_dims(at, axis=-1)
        output = x * at
        return tf.reduce_sum(output, axis=1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    def get_config(self):
        return super(AttentionLayer, self).get_config()

class GRUModel(BaseModel):
    def __init__(self, use_attention=False, seq_length=10, n_features=17, lookahead=1):
        super().__init__()
        if use_attention:
            self.model = self._build_gru_attention(seq_length, n_features, lookahead)
        else:
            self.model = self._build_gru(seq_length, n_features, lookahead)

    def _build_gru(self, seq_length, n_features, lookahead):
        model = Sequential()
        model.add(GRU(32, return_sequences=True, input_shape=(seq_length, n_features)))
        model.add(Dropout(0.3))
        model.add(GRU(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(128))
        model.add(Dropout(0.2))
        model.add(Dense(lookahead, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mse', metrics=[MeanAbsolutePercentageError(), RootMeanSquaredError()])
        return model

    def _build_gru_attention(self, seq_length, n_features, lookahead):
        model = Sequential()
        model.add(GRU(32, return_sequences=True, input_shape=(seq_length, n_features)))
        model.add(Dropout(0.3))
        model.add(GRU(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(AttentionLayer())
        model.add(Flatten())
        model.add(Dense(lookahead, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mse', metrics=[MeanAbsolutePercentageError(), RootMeanSquaredError()])
        return model

    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path, custom_objects={'AttentionLayer': AttentionLayer})

def build_gru(seq_length, n_features, lookahead):
    """Legacy wrapper for backward compatibility."""
    return GRUModel(use_attention=False, seq_length=seq_length, n_features=n_features, lookahead=lookahead).model

def build_gru_attention(seq_length, n_features, lookahead):
    """Legacy wrapper for backward compatibility."""
    return GRUModel(use_attention=True, seq_length=seq_length, n_features=n_features, lookahead=lookahead).model
