"""Thematic analysis using TF-IDF keyword extraction"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Theme definitions with keywords
THEME_KEYWORDS = {
    "Transaction Performance": ["transfer", "transaction", "payment", "send", "money", "fast", "slow", "instant", "balance", "deduct"],
    "Authentication Issues": ["login", "otp", "password", "fingerprint", "biometric", "verify", "code", "sms", "authenticate"],
    "UI/UX & Design": ["ui", "interface", "design", "navigation", "button", "screen", "display", "layout", "easy", "friendly"],
    "App Stability": ["crash", "freeze", "bug", "error", "stuck", "loading", "slow", "hang", "stop", "broken"],
    "Customer Support": ["support", "help", "service", "response", "complaint", "call", "email", "contact", "assist", "resolve"],
    "Feature Requests": ["fingerprint", "budget", "saving", "invest", "loan", "card", "qr", "scan", "notify", "alert"]
}

def extract_themes(input_file="data/processed/sentiment_reviews.csv", output_file="data/processed/themed_reviews.csv"):
    df = pd.read_csv(input_file)
    
    logger.info("Extracting themes from reviews...")
    
    # Simple keyword matching for themes
    def assign_theme(text):
        text_lower = str(text).lower()
        best_theme = "General"
        best_score = 0
        
        for theme, keywords in THEME_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_theme = theme
        
        return best_theme if best_score > 0 else "General Feedback"
    
    df["identified_theme"] = df["review"].apply(assign_theme)
    
    df.to_csv(output_file, index=False)
    logger.info(f"Saved themed results to {output_file}")
    logger.info(f"Theme distribution:\n{df['identified_theme'].value_counts()}")
    return df

if __name__ == "__main__":
    extract_themes()
