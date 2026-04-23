import unittest
import pandas as pd
from src.features.sentiment import analyze_sentiment

class TestSentimentAnalysis(unittest.TestCase):
    def test_analyze_sentiment_basic(self):
        news = [
            {
                'content': {
                    'title': 'The market is looking wonderful today with great gains!',
                    'pubDate': '2023-10-27T12:00:00Z'
                }
            },
            {
                'content': {
                    'title': 'Economic disaster strikes as stocks plummet to new lows.',
                    'pubDate': '2023-10-28T12:00:00Z'
                }
            }
        ]
        df = analyze_sentiment(news)

        # Should have two dates
        self.assertEqual(len(df), 2)

        # 2023-10-27 should be positive
        pos_score = df[df['Date'] == '2023-10-27']['Sentiment'].values[0]
        self.assertGreater(pos_score, 0)

        # 2023-10-28 should be negative
        neg_score = df[df['Date'] == '2023-10-28']['Sentiment'].values[0]
        self.assertLess(neg_score, 0)

    def test_analyze_sentiment_aggregation(self):
        news = [
            {'content': {'title': 'Good', 'pubDate': '2023-10-27T10:00:00Z'}},
            {'content': {'title': 'Great', 'pubDate': '2023-10-27T11:00:00Z'}}
        ]
        df = analyze_sentiment(news)
        # Should aggregate into 1 row for the date
        self.assertEqual(len(df), 1)

if __name__ == '__main__':
    unittest.main()
