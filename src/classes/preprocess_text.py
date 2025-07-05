import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from typing import List, Optional
from src.utils.nltk_setup_data import setup_nltk_data

import warnings

# Filter all RuntimeWarnings globally
warnings.filterwarnings('ignore', category=RuntimeWarning)
        

class TextPreprocessor:
    def __init__(self, language: str = 'english'):
        """Initialize with language settings"""
        # Setup NLTK data with SSL fix
        setup_nltk_data()
        
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words(language))
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize a sentence into words"""
        if pd.isna(text) or not text:
            return []
        # Convert to lowercase and tokenize
        text = text.lower()
        tokens = word_tokenize(text)
        return tokens
    
    def stem(self, text: str) -> str:
        """Apply stemming to a sentence"""
        if pd.isna(text) or not text:
            return ""
        tokens = self.tokenize(text)
        # Remove punctuation and apply stemming
        clean_tokens = []
        for token in tokens:
            if re.match(r'^[a-zA-Z]+$', token):  # Only alphabetic tokens
                stemmed = self.stemmer.stem(token)
                clean_tokens.append(stemmed)
        return ' '.join(clean_tokens)
    
    def lemmatize(self, text: str) -> str:
        """Apply lemmatization to a sentence"""
        if pd.isna(text) or not text:
            return ""
        tokens = self.tokenize(text)
        # Remove punctuation and apply lemmatization
        clean_tokens = []
        for token in tokens:
            if re.match(r'^[a-zA-Z]+$', token):  # Only alphabetic tokens
                lemmatized = self.lemmatizer.lemmatize(token)
                clean_tokens.append(lemmatized)
        return ' '.join(clean_tokens)
        
        
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
                'original_text': '',
                'processed_text': '',
                'original_length': 0,
                'processed_length': 0,
                'original_words': 0,
                'processed_words': 0,
                'removed_stopwords': 0,
                'stopwords_percentage': 0.0,
                'reduction_percentage': 0.0,
                'unique_words_original': 0,
                'unique_words_processed': 0,
                'sample_removed_words': []
            }
            
        original = str(text).strip()
        
        # Step-by-step preprocessing to track changes
        step1_lower = original.lower()
        step2_clean = re.sub(r'[^a-zA-Z\s]', '', step1_lower)
        step3_tokens = word_tokenize(step2_clean)
        
        # Find stopwords and content words separately
        stopwords_found = [token for token in step3_tokens if token in self.stop_words]
        content_tokens = [token for token in step3_tokens if token not in self.stop_words]
        
        # Apply lemmatization to content words
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in content_tokens]
        processed_text = ' '.join(lemmatized_tokens)
        
        # Calculate meaningful statistics
        orig_word_count = len(step3_tokens) if step3_tokens else 0
        processed_word_count = len(lemmatized_tokens)
        stopwords_removed = len(stopwords_found)
        
        return {
            'original_text': original,
            'processed_text': processed_text,
            'original_length': len(original),
            'processed_length': len(processed_text),
            'original_words': orig_word_count,
            'processed_words': processed_word_count,
            'removed_stopwords': stopwords_removed,
            'stopwords_percentage': round(stopwords_removed / orig_word_count * 100, 2) if orig_word_count > 0 else 0.0,
            'reduction_percentage': round((orig_word_count - processed_word_count) / orig_word_count * 100, 2) if orig_word_count > 0 else 0.0,
            'unique_words_original': len(set(step3_tokens)),
            'unique_words_processed': len(set(lemmatized_tokens)),
            'sample_removed_words': list(set(stopwords_found))[:5]
        }

    def get_batch_stats(self, texts: List[str]) -> pd.DataFrame:
        """Get statistics for a batch of texts"""
        stats = [self.get_preprocessing_stats(text) for text in texts]
        return pd.DataFrame(stats)
    
    def extract_top_category(self, category_tree: str) -> str:
        """
        Extract the first (top-level) category from a category tree string.
        
        Args:
            category_tree: String containing the full category hierarchy
                e.g. ["Baby Care >> Baby Bath & Skin >> Baby Bath Towels"]
                
        Returns:
            First category as string (e.g., "Baby Care")
        """
        if not isinstance(category_tree, str):
            return ""
            
        # Remove brackets if present
        if category_tree.startswith('[') and category_tree.endswith(']'):
            category_tree = category_tree[1:-1].strip('"\'')
            
        # Split by ">>" and get first category
        categories = category_tree.split(">>")
        if categories:
            return categories[0].strip()
        
        return ""

    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts efficiently"""
        return [self.preprocess(text) for text in texts]