"""Data preprocessing for bank reviews"""
import pandas as pd
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess(input_file="data/raw/all_reviews_raw.csv", output_file="data/processed/cleaned_reviews.csv"):
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} reviews")
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["review", "bank"])
    logger.info(f"Removed {before - len(df)} duplicates")
    
    # Drop missing critical fields
    df = df.dropna(subset=["review", "rating"])
    
    # Normalize dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    
    # Clean text
    df["review"] = df["review"].str.strip()
    df = df[df["review"].str.len() > 5]
    
    # Add review ID
    df = df.reset_index(drop=True)
    df["review_id"] = [f"R{i+1:05d}" for i in range(len(df))]
    
    # Select columns
    df = df[["review_id", "review", "rating", "date", "bank", "source"]]
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df)} cleaned reviews to {output_file}")
    logger.info(f"Reviews per bank:\n{df['bank'].value_counts()}")
    return df

if __name__ == "__main__":
    preprocess()
