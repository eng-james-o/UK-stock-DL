import yfinance as yf
import pandas as pd

def download_data(ticker_symbol, start_date, end_date):
    """Download stock data from Yahoo Finance."""
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return data

def fetch_news(ticker_symbol):
    """
    Fetch recent news headlines for a given ticker from Yahoo Finance.

    Args:
        ticker_symbol (str): The stock ticker symbol.

    Returns:
        list: A list of dictionaries containing news headlines and timestamps.
    """
    ticker = yf.Ticker(ticker_symbol)
    return ticker.news
