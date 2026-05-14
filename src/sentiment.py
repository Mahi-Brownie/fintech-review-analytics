"""Sentiment analysis using VADER"""
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_sentiment(input_file="data/processed/cleaned_reviews.csv", output_file="data/processed/sentiment_reviews.csv"):
    df = pd.read_csv(input_file)
    analyzer = SentimentIntensityAnalyzer()
    
    logger.info(f"Analyzing sentiment for {len(df)} reviews...")
    
    sentiments = []
    for text in df["review"]:
        scores = analyzer.polarity_scores(str(text))
        compound = scores["compound"]
        if compound >= 0.05:
            label = "POSITIVE"
        elif compound <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        sentiments.append({"sentiment_label": label, "sentiment_score": compound})
    
    sentiment_df = pd.DataFrame(sentiments)
    df = pd.concat([df, sentiment_df], axis=1)
    
    df.to_csv(output_file, index=False)
    logger.info(f"Saved sentiment results to {output_file}")
    logger.info(f"Sentiment distribution:\n{df['sentiment_label'].value_counts()}")
    return df

if __name__ == "__main__":
    analyze_sentiment()
