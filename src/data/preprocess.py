import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ..features.technical_indicators import *

def preprocess_data(data, window_size=14, sentiment_score=0.0):
    """
    Preprocess stock data and add technical indicators and sentiment.

    Args:
        data (pd.DataFrame): Raw stock data.
        window_size (int): Window size for technical indicators.
        sentiment_score (float): Global sentiment score to be added (optional).
    """
    # Typical price
    data['TP'] = (data['Close'] + data['Low'] + data['High'])/3
    # Moving averages
    data['MA'] = data['Close'].rolling(window=window_size).mean()
    data['EMA'] = data['Close'].ewm(span=window_size).mean()
    data['STD'] = data['TP'].rolling(window_size).std(ddof=0)
    # Technical indicators
    data['RSI'] = rsi(data.Close, periods=window_size, ema=True)
    data['ROC'] = roc(data.Close, window_size)
    data['ATR'] = atr(data.High, data.Low, data.Close)
    data['CCI'] = cci(data.MA, data.TP, window_size)
    k_period, d_period = 14, 3
    n_high = data['High'].rolling(k_period).max()
    n_low = data['Low'].rolling(k_period).min()
    data['%K'] = (data['Close'] - n_low) * 100 / (n_high - n_low)
    data['%D'] = data['%K'].rolling(d_period).mean()
    data['plus_di'], data['minus_di'], data['ADMI'] = get_adx(data['High'], data['Low'], data['Close'], data['ATR'], lookback=14)
    data['AO'] = AO(data.Close)
    data['WIL_R'] = wil_r(data.High, data.Low, data.Close, 14)
    data['OBV'] = obv(data.Close, data.Volume)

    # Add sentiment feature
    data['Sentiment'] = sentiment_score

    data = data.dropna()
    return data

def normalize_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns)
    return scaled_data, scaler

def unscale_data(scaled_value, scaler, col_idx=0):
    """Individually unscale a specific column (default is Close price at index 0)."""
    unscaled_value = scaled_value * (scaler.data_max_[col_idx] - scaler.data_min_[col_idx]) + (scaler.data_min_[col_idx])
    return unscaled_value

def lag_data(data:pd.DataFrame, seq_length=10, lookahead=1):
    X_data, y_data = [],[]
    # Use integer-based indexing for robustness with sliding windows
    vals = data.values
    close_idx = data.columns.get_loc('Close')

    for i in range(len(data) - lookahead - seq_length + 1):
        seq = vals[i : i+seq_length]
        y = vals[i+seq_length : i+seq_length+lookahead, close_idx]
        X_data.append(seq)
        y_data.append(y)

    X_data = np.array(X_data)
    y_data = np.array(y_data)
    return X_data, y_data
