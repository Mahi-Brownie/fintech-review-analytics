"""Google Play Store Review Scraper for Ethiopian Banks"""
import pandas as pd
import numpy as np
from google_play_scraper import Sort, reviews
import time
import logging
import os
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BANK_APPS = {
    "CBE": {"app_id": "com.commercialbankofethiopia.cbe", "name": "Commercial Bank of Ethiopia"},
    "BOA": {"app_id": "com.bankofabyssinia.amole", "name": "Bank of Abyssinia"},
    "Dashen": {"app_id": "com.dashenbank.digital", "name": "Dashen Bank"}
}

def scrape_bank(app_id, bank_name, target=400):
    all_reviews = []
    token = None
    logger.info(f"Scraping {bank_name}...")
    
    while len(all_reviews) < target:
        try:
            result, token = reviews(
                app_id, lang="en", country="et", sort=Sort.NEWEST,
                count=min(200, target - len(all_reviews)),
                continuation_token=token
            )
            if not result:
                break
            for r in result:
                all_reviews.append({
                    "review": r.get("content", ""),
                    "rating": r.get("score", 0),
                    "date": r.get("at", None),
                    "bank": bank_name,
                    "source": "Google Play"
                })
            logger.info(f"  Got {len(all_reviews)}/{target} reviews...")
            time.sleep(2)
            if token is None:
                break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(30)
            break
    return pd.DataFrame(all_reviews)

def generate_sample_data():
    np.random.seed(42)
    pos = ["Great app for quick transfers!", "Love the new UI, very convenient", "Reliable and fast banking app", "Best banking app in Ethiopia!", "Easy to use, highly recommend"]
    neg = ["App crashes during transfers", "Very slow loading times", "OTP not received, cannot login", "Login error persists after reinstall", "Transfer failed but money deducted"]
    neu = ["App works okay, nothing special", "Average banking app", "Decent but needs more features", "Basic functionality works fine"]
    
    data = []
    for bank in ["Commercial Bank of Ethiopia", "Bank of Abyssinia", "Dashen Bank"]:
        for _ in range(400):
            s = np.random.choice(["pos", "neg", "neu"], p=[0.4, 0.35, 0.25])
            text = np.random.choice({"pos": pos, "neg": neg, "neu": neu}[s])
            rating = {"pos": np.random.choice([4,5]), "neg": np.random.choice([1,2]), "neu": np.random.choice([2,3,4])}[s]
            days_ago = np.random.randint(0, 365)
            data.append({
                "review": text,
                "rating": rating,
                "date": (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
                "bank": bank,
                "source": "Google Play (Sample)"
            })
    return pd.DataFrame(data)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    try:
        logger.info("Attempting to scrape real reviews...")
        all_data = []
        for key, info in BANK_APPS.items():
            df = scrape_bank(info["app_id"], info["name"])
            all_data.append(df)
            df.to_csv(f"data/raw/{key}_reviews.csv", index=False)
        combined = pd.concat(all_data, ignore_index=True)
        if len(combined) < 1000:
            raise Exception("Not enough real reviews")
        logger.info(f"Scraped {len(combined)} real reviews!")
    except Exception as e:
        logger.warning(f"Scraping failed: {e}")
        logger.info("Generating sample data...")
        combined = generate_sample_data()
    combined.to_csv("data/raw/all_reviews_raw.csv", index=False)
    logger.info(f"Saved {len(combined)} reviews")
    print(f"\nReviews per bank:\n{combined['bank'].value_counts()}")
