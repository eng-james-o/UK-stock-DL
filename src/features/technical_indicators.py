import pandas as pd
import numpy as np

def rsi(close, periods=10, ema=True):
    close_delta = close.diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    if ema:
        ma_up = up.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
    else:
        ma_up = up.rolling(window=periods, adjust=False).mean()
        ma_down = down.rolling(window=periods, adjust=False).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

def roc(close, n):
    N = close.diff(n)
    D = close.shift(n)
    roc = pd.Series(N/D, name='Rate of Change')
    return roc

def atr(high, low, close, window_size=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window_size).mean()

def cci(ma, tp, ndays):
    MAD = tp.rolling(ndays).apply(lambda x: (pd.Series(x) - pd.Series(x).mean()).abs().mean())
    cci = (tp - ma) / (0.015 * MAD)
    return cci

def get_adx(high, low, close, atr, lookback):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    plus_di = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth = adx.ewm(alpha = 1/lookback).mean()
    return plus_di, minus_di, adx_smooth

def AO(close):
    sma5 = close.rolling(5).mean()
    sma34 = close.rolling(34).mean()
    ao = sma5 - sma34
    return ao

def wil_r(high, low, close, lookback):
    highh = high.rolling(lookback).max()
    lowl = low.rolling(lookback).min()
    wr = -100 * ((highh - close) / (highh - lowl))
    return wr

def obv(close, volume):
    obv_series = [0]
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv_series.append(obv_series[-1] + volume[i])
        elif close[i] < close[i-1]:
            obv_series.append(obv_series[-1] - volume[i])
        else:
            obv_series.append(obv_series[-1])
    return obv_series
