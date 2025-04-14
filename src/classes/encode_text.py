import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TextEncoder:
    def __init__(self, min_df: float = 0.01, max_df: float = 0.95):
        """Initialize encoders with parameters"""
        self.bow_vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
        self.tfidf_vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
        self.bow_matrix = None
        self.tfidf_matrix = None
        self.feature_names = None
        
    def fit_transform(self, texts: pd.Series) -> Dict[str, Any]:
        """Fit and transform texts using both BoW and TF-IDF"""
        # Bag of Words
        self.bow_matrix = self.bow_vectorizer.fit_transform(texts)
        self.feature_names = self.bow_vectorizer.get_feature_names_out()
        
        # TF-IDF
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        return {
            'bow_matrix': self.bow_matrix,
            'tfidf_matrix': self.tfidf_matrix,
            'features': self.feature_names
        }
    
    def get_top_features(self, n_features: int = 10) -> pd.DataFrame:
        """Get top features based on TF-IDF scores"""
        if self.tfidf_matrix is None:
            raise ValueError("Must call fit_transform first")
            
        mean_tfidf = np.array(self.tfidf_matrix.mean(axis=0)).flatten()
        feature_scores = pd.DataFrame({
            'feature': self.feature_names,
            'score': mean_tfidf
        })
        return feature_scores.nlargest(n_features, 'score')
    
    def plot_top_features(self, threshold: float = 0.98) -> go.Figure:
        """Plot top features that explain threshold% of variance"""
        if self.tfidf_matrix is None:
            raise ValueError("Must call fit_transform first")
        
        # Calculate feature importance
        mean_tfidf = np.array(self.tfidf_matrix.mean(axis=0)).flatten()
        feature_scores = pd.DataFrame({
            'feature': self.feature_names,
            'score': mean_tfidf
        }).sort_values('score', ascending=False)
        
        # Calculate cumulative percentages
        total = feature_scores['score'].sum()
        feature_scores['percentage'] = feature_scores['score'] / total
        feature_scores['cumulative'] = feature_scores['percentage'].cumsum()
        
        # Find cutoff point
        cutoff_idx = (feature_scores['cumulative'] <= threshold).sum()
        
        # Create subplot
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=feature_scores['feature'][:cutoff_idx],
                y=feature_scores['percentage'][:cutoff_idx],
                name="Feature Importance"
            )
        )
        
        # Add cumulative line
        fig.add_trace(
            go.Scatter(
                x=feature_scores['feature'][:cutoff_idx],
                y=feature_scores['cumulative'][:cutoff_idx],
                name="Cumulative %",
                line=dict(color='red')
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title=f"Top Features Explaining {threshold*100}% of Variance",
            xaxis_title="Features",
            yaxis_title="Importance %",
            yaxis2_title="Cumulative %",
            showlegend=True
        )
        
        return fig