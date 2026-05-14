"""Visualizations for fintech review analysis"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_visualizations(input_file="data/processed/themed_reviews.csv", output_dir="data/plots"):
    df = pd.read_csv(input_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use("ggplot")
    
    # Plot 1: Sentiment distribution by bank
    fig, ax = plt.subplots(figsize=(10, 6))
    sentiment_counts = pd.crosstab(df["bank"], df["sentiment_label"], normalize="index") * 100
    sentiment_counts.plot(kind="bar", stacked=True, ax=ax, color=["#ff6b6b", "#ffd93d", "#6bcb77"])
    ax.set_title("Sentiment Distribution by Bank", fontsize=14)
    ax.set_xlabel("Bank")
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_by_bank.png", dpi=150)
    plt.close()
    logger.info("Plot 1 saved: sentiment_by_bank.png")
    
    # Plot 2: Rating distribution by bank
    fig, ax = plt.subplots(figsize=(10, 6))
    for bank in df["bank"].unique():
        bank_data = df[df["bank"] == bank]["rating"]
        ax.hist(bank_data, alpha=0.5, label=bank, bins=5)
    ax.set_title("Rating Distribution by Bank", fontsize=14)
    ax.set_xlabel("Rating (Stars)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rating_distribution.png", dpi=150)
    plt.close()
    logger.info("Plot 2 saved: rating_distribution.png")
    
    # Plot 3: Average rating by bank
    fig, ax = plt.subplots(figsize=(8, 5))
    avg_ratings = df.groupby("bank")["rating"].mean().sort_values()
    colors = ["#ff6b6b", "#ffd93d", "#6bcb77"]
    avg_ratings.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Average Rating by Bank", fontsize=14)
    ax.set_xlabel("Average Rating")
    for i, v in enumerate(avg_ratings):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/avg_rating.png", dpi=150)
    plt.close()
    logger.info("Plot 3 saved: avg_rating.png")
    
    # Plot 4: Theme frequency by bank
    fig, ax = plt.subplots(figsize=(12, 6))
    theme_counts = pd.crosstab(df["bank"], df["identified_theme"])
    theme_counts.plot(kind="barh", stacked=True, ax=ax, colormap="viridis")
    ax.set_title("Theme Distribution by Bank", fontsize=14)
    ax.set_xlabel("Number of Reviews")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/theme_distribution.png", dpi=150)
    plt.close()
    logger.info("Plot 4 saved: theme_distribution.png")
    
    # Plot 5: Sentiment score by rating
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column="sentiment_score", by="rating", ax=ax)
    ax.set_title("Sentiment Score vs Rating", fontsize=14)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Sentiment Score")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_vs_rating.png", dpi=150)
    plt.close()
    logger.info("Plot 5 saved: sentiment_vs_rating.png")
    
    # Generate summary stats
    summary = {
        "total_reviews": len(df),
        "avg_rating": df["rating"].mean(),
        "sentiment_distribution": df["sentiment_label"].value_counts().to_dict(),
        "reviews_per_bank": df["bank"].value_counts().to_dict(),
        "top_themes": df["identified_theme"].value_counts().head(5).to_dict()
    }
    
    logger.info(f"Summary stats: {summary}")
    return summary

if __name__ == "__main__":
    create_visualizations()
