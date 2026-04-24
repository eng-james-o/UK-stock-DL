import yfinance as yf
import pandas as pd

def download_data(ticker_symbol, start_date, end_date):
    """Download stock data from Yahoo Finance."""
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return data
