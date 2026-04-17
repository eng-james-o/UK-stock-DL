from keras.models import Sequential, load_model
from keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, LSTM, MaxPooling1D, MaxPooling2D, Reshape
from keras.optimizers import Adam
from keras.metrics import MeanAbsolutePercentageError, RootMeanSquaredError
from .base_model import BaseModel

class CNNLSTMModel(BaseModel):
    def __init__(self, model_type='1d', seq_length=10, n_features=17, lookahead=1):
        super().__init__()
        if model_type == '1d':
            self.model = self._build_cnn_lstm_1d(seq_length, n_features, lookahead)
        elif model_type == '2d':
            self.model = self._build_cnn_lstm_2d(seq_length, n_features, lookahead)
        else:
            raise ValueError("model_type must be '1d' or '2d'")

    def _build_cnn_lstm_1d(self, seq_length, n_features, lookahead):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(lookahead))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[MeanAbsolutePercentageError(), RootMeanSquaredError()])
        return model

    def _build_cnn_lstm_2d(self, seq_length, n_features, lookahead):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(seq_length, n_features, 1)))
        model.add(MaxPooling2D(pool_size=2))
        # Note: Reshape target dimensions might need adjustment based on input shape
        # In the original code it was (14, 64) which might be specific to a certain input size
        # We'll keep it as is for now but it's a potential point of failure for different seq_lengths
        model.add(Reshape((-1, 64)))
        model.add(LSTM(64, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(lookahead))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[MeanAbsolutePercentageError(), RootMeanSquaredError()])
        return model

    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)

def build_cnn_lstm_1d(seq_length, n_features, lookahead):
    """Legacy wrapper for backward compatibility."""
    return CNNLSTMModel(model_type='1d', seq_length=seq_length, n_features=n_features, lookahead=lookahead).model

def build_cnn_lstm_2d(seq_length, n_features, lookahead):
    """Legacy wrapper for backward compatibility."""
    return CNNLSTMModel(model_type='2d', seq_length=seq_length, n_features=n_features, lookahead=lookahead).model
