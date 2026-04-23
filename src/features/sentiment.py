from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

def analyze_sentiment(news_list):
    """
    Analyze sentiment of a list of news items.

    Args:
        news_list (list): List of news items as returned by yfinance.

    Returns:
        pd.DataFrame: A DataFrame with 'Date' and 'Sentiment' score.
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment_data = []

    for item in news_list:
        content = item.get('content', {})
        title = content.get('title', '')
        pub_date = content.get('pubDate', '')

        if title and pub_date:
            # Get compound sentiment score
            score = analyzer.polarity_scores(title)['compound']
            # Convert date to YYYY-MM-DD
            date = pd.to_datetime(pub_date).strftime('%Y-%m-%d')
            sentiment_data.append({'Date': date, 'Sentiment': score})

    df = pd.DataFrame(sentiment_data)
    if not df.empty:
        # Group by date and take the average sentiment score if there are multiple news items for the same day
        df = df.groupby('Date')['Sentiment'].mean().reset_index()
        df['Date'] = pd.to_datetime(df['Date'])

    return df
