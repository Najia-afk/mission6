import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List, Optional

class TextPreprocessor:
    def __init__(self, language: str = 'english'):
        """Initialize with language settings"""
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        
    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def get_preprocessing_stats(self, text: str) -> dict:
        """Get detailed statistics about preprocessing effects"""
        if pd.isna(text):
            return {
                'status': 'empty_input',
                'stats': None
            }
            
        original = str(text)
        cleaned = self.preprocess(text)
        
        # Original text stats
        orig_tokens = word_tokenize(original.lower())
        orig_stop_words = [w for w in orig_tokens if w in self.stop_words]
        
        return {
            'original_length': len(original),
            'processed_length': len(cleaned),
            'original_words': len(orig_tokens),
            'processed_words': len(cleaned.split()),
            'removed_stopwords': len(orig_stop_words),
            'stopwords_percentage': round(len(orig_stop_words) / len(orig_tokens) * 100, 2),
            'reduction_percentage': round((len(original) - len(cleaned)) / len(original) * 100, 2),
            'unique_words_original': len(set(orig_tokens)),
            'unique_words_processed': len(set(cleaned.split())),
            'sample_removed_words': list(set(orig_stop_words))[:5]
        }

    def get_batch_stats(self, texts: List[str]) -> pd.DataFrame:
        """Get statistics for a batch of texts"""
        stats = [self.get_preprocessing_stats(text) for text in texts]
        return pd.DataFrame(stats)