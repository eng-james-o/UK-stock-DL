from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_sentiment(text):
    """
    Calculate sentiment score for a given text using VADER.

    Args:
        text (str): The input text (e.g., news headline).

    Returns:
        float: Compound sentiment score ranging from -1 (negative) to 1 (positive).
    """
    analyzer = SentimentIntensityAnalyzer()
    if not text or not isinstance(text, str):
        return 0.0
    vs = analyzer.polarity_scores(text)
    return vs['compound']
