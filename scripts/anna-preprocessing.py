"""
Data preprocessing script for claims classification project.
Anna - Data Preprocessing

This script:
1. Loads raw HTML data from claims-raw.RData
2. Scrapes both paragraph AND header content (Task 1 requirement)
3. Cleans and tokenizes the text
4. Produces claims-clean.RData for the team to use
"""

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pyreadr

# Load raw data
print("Loading raw data...")
raw_data = pyreadr.read_r('../data/claims-raw.RData')
claims_raw = raw_data['claims_raw']  # Adjust key name if different

print(f"Loaded {len(claims_raw)} records")

def parse_html(html_string):
    """
    Parse HTML and extract text from paragraphs AND headers (h1-h6).
    This is the improved scraping strategy for Task 1.
    """
    if not html_string or not isinstance(html_string, str):
        return ""
    
    try:
        soup = BeautifulSoup(html_string, 'html.parser')
        
        # Extract paragraphs
        paragraphs = soup.find_all('p')
        p_text = ' '.join([p.get_text() for p in paragraphs])
        
        # Extract headers (h1, h2, h3, h4, h5, h6) - Task 1 improvement
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        h_text = ' '.join([h.get_text() for h in headers])
        
        # Combine header and paragraph text
        full_text = h_text + ' ' + p_text
        
        return full_text
    except:
        return ""

def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove single quotes
    text = text.replace("'", "")
    # Remove newlines, punctuation, numbers, symbols
    text = re.sub(r'[\n\[\]{}()\-_+=*&^%$#@!~`|\\/<>?;:"\']', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    # Add space between lowercase and uppercase letters
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Convert to lowercase
    text = text.lower()
    # Remove 'nbsp' and similar
    text = re.sub(r'nbsp|amp|quot', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

# Parse and clean all HTML
print("Parsing HTML and extracting text (including headers)...")
claims_raw['text_clean'] = claims_raw['text_tmp'].apply(lambda x: clean_text(parse_html(x)))

# Filter out entries that don't have valid HTML or text
claims_clean = claims_raw[claims_raw['text_clean'].str.len() > 0].copy()

print(f"Cleaned {len(claims_clean)} records with valid text")

# Tokenize and compute TF-IDF
print("Tokenizing and computing TF-IDF features...")

# Simple tokenization with stopword removal
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=1000,  # Keep top 1000 features
    stop_words='english',
    min_df=2,  # Word must appear in at least 2 documents
    max_df=0.95,  # Word must appear in less than 95% of documents
    ngram_range=(1, 1)  # Unigrams only for base preprocessing
)

# Fit and transform
tfidf_matrix = vectorizer.fit_transform(claims_clean['text_clean'])

# Convert to DataFrame
feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names,
    index=claims_clean.index
)

# Combine with metadata
claims_clean_final = pd.concat([
    claims_clean[['.id', 'bclass']].reset_index(drop=True),
    tfidf_df.reset_index(drop=True)
], axis=1)

print(f"Final dataset shape: {claims_clean_final.shape}")
print(f"Features: {claims_clean_final.shape[1] - 2} TF-IDF features")

# Save as pickle (Python format) for easy use
print("Saving cleaned data...")
with open('../data/claims-clean.pkl', 'wb') as f:
    pickle.dump(claims_clean_final, f)

print("Saved as claims-clean.pkl")

# Also try to save as RData format if rpy2 is available
try:
    import pyreadr
    pyreadr.write_rdata('../data/claims-clean.RData', claims_clean_final, df_name='claims_clean')
    print("Also saved as claims-clean.RData for R users")
except:
    print("Note: Could not save as .RData format. Install pyreadr if R compatibility is needed.")

print("\nPreprocessing complete!")
print(f"Output: ../data/claims-clean.pkl (and .RData if available)")
print(f"Dimensions: {claims_clean_final.shape[0]} observations x {claims_clean_final.shape[1]} columns")
print(f"Columns: .id, bclass, + {claims_clean_final.shape[1]-2} TF-IDF features")
