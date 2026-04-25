from src.features.sentiment import get_sentiment

def test_get_sentiment_positive():
    text = "The company reported record-breaking profits and optimistic growth."
    score = get_sentiment(text)
    assert score > 0

def test_get_sentiment_negative():
    text = "The stock plummeted following a massive fraud investigation."
    score = get_sentiment(text)
    assert score < 0

def test_get_sentiment_empty():
    assert get_sentiment("") == 0.0
