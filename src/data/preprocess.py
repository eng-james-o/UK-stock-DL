import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ..features.technical_indicators import *

def preprocess_data(data, window_size=14):
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
    data = data.dropna()
    return data

def normalize_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns)
    return scaled_data, scaler

def lag_data(data:pd.DataFrame, seq_length=10, lookahead=1):
    X_data, y_data = [],[]
    for i in range(len(data) - lookahead - seq_length + 1):
        seq = data.loc[i : i+seq_length-1]
        y = data.Close.loc[i+seq_length: i+seq_length+lookahead-1]
        X_data.append(seq); y_data.append(y)
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    return X_data, y_data
