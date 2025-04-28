import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class NewsSentimentAnalyzer:
    def __init__(self):
        self.api_key = '75d3452f0ea1403eb117ccb27fce3bc7'
        self.news_url = 'https://newsapi.org/v2/everything'
        self.analyzer = SentimentIntensityAnalyzer()

    def fetch_headlines(self):
        params = {
            'q': 'bitcoin',
            'apiKey': self.api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 20
        }
        try:
            response = requests.get(self.news_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('status') == 'ok' and 'articles' in data:
                headlines = [article['title'] for article in data['articles']]
                return headlines
            else:
                print("❗ Unexpected NewsAPI structure.")
                return []
        except Exception as e:
            print(f"❗ Error fetching news: {e}")
            return []

    def analyze_sentiment(self):
        headlines = self.fetch_headlines()
        if not headlines:
            return 0.0, []

        sentiment_scores = [self.analyzer.polarity_scores(title)['compound'] for title in headlines]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

        return avg_sentiment, headlines