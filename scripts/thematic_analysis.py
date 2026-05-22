import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Load data with sentiment columns
df = pd.read_csv("data/sentiment_enriched.csv")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    """Lemmatize, remove stopwords and non-alpha tokens."""
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

print("Cleaning review text...")
df['clean_text'] = df['review'].apply(preprocess)

# Define theme keywords (you can adjust these after seeing the top keywords)
theme_keywords = {
    "Account & Login": ["login", "password", "otp", "register", "verify", "forgot", "reset", "log"],
    "Transaction & Transfer": ["transfer", "send", "receive", "money", "payment", "transaction", "fund", "balance"],
    "App Performance": ["crash", "slow", "load", "freeze", "bug", "error", "hang", "network", "timeout"],
    "User Interface": ["ui", "design", "easy", "interface", "navigation", "button", "menu", "update", "look"],
    "Customer Support": ["support", "service", "help", "call", "branch", "agent", "response", "complaint"]
}

def assign_theme(text):
    """Assign a theme based on keyword presence."""
    text_lower = str(text).lower()
    for theme, keywords in theme_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                return theme
    return "Other"

print("Assigning themes...")
df['identified_theme'] = df['review'].apply(assign_theme)

# Also extract top keywords per bank (for your report / later analysis)
print("\nTop keywords per bank:")
for bank in df['bank'].unique():
    bank_df = df[df['bank'] == bank]
    vectorizer = TfidfVectorizer(max_features=15, ngram_range=(1,2), stop_words='english')
    vectorizer.fit(bank_df['clean_text'])
    keywords = vectorizer.get_feature_names_out()
    print(f"{bank}: {', '.join(keywords)}")

# Save final processed dataset
df.to_csv("data/processed_reviews.csv", index=False)
print("\nProcessed data saved to data/processed_reviews.csv")