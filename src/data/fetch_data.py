import yfinance as yf
import pandas as pd
from ..features.sentiment import get_sentiment

def download_data(ticker_symbol, start_date, end_date):
    """Download stock data from Yahoo Finance."""
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return data

def fetch_news_sentiment(ticker_symbol):
    """
    Fetch recent news for a ticker and calculate average sentiment.
    Note: yfinance provides recent news. For historical backtesting,
    a more robust news API would be required.
    """
    ticker = yf.Ticker(ticker_symbol)
    news = ticker.news
    if not news:
        return 0.0

    sentiments = []
    for item in news:
        title = item.get('content', {}).get('title', '')
        if title:
            sentiments.append(get_sentiment(title))

    if not sentiments:
        return 0.0

    return sum(sentiments) / len(sentiments)
