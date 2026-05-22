import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load cleaned data from Task 1
df = pd.read_csv("data/cleaned_reviews.csv")
print(f"Loaded {len(df)} reviews.")

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    """Return label and compound score."""
    score = analyzer.polarity_scores(str(text))
    compound = score['compound']
    if compound >= 0.05:
        return 'POSITIVE', compound
    elif compound <= -0.05:
        return 'NEGATIVE', compound
    else:
        return 'NEUTRAL', compound

# Apply sentiment analysis
df[['sentiment_label', 'sentiment_score']] = df['review'].apply(
    lambda x: pd.Series(get_sentiment(x))
)

# Save to a new CSV (still not the final one; we’ll add themes later)
df.to_csv("data/sentiment_enriched.csv", index=False)
print("Sentiment scores added and saved to data/sentiment_enriched.csv")