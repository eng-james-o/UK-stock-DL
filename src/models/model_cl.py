from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, LSTM, MaxPooling1D, MaxPooling2D, Reshape
from keras.optimizers import Adam
from keras.metrics import MeanAbsolutePercentageError, RootMeanSquaredError

def build_cnn_lstm_1d(seq_length, n_features, lookahead):
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

def build_cnn_lstm_2d(seq_length, n_features, lookahead):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(seq_length, n_features, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Reshape((14, 64)))
    model.add(LSTM(64, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(lookahead))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[MeanAbsolutePercentageError(), RootMeanSquaredError()])
    return model
