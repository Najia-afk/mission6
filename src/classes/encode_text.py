import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud


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
    
    def _plot_features(self, matrix: np.ndarray, title: str, threshold: float = 0.98) -> go.Figure:
        """Base method for plotting feature importance"""
        # Calculate feature importance
        mean_scores = np.array(matrix.mean(axis=0)).flatten()
        feature_scores = pd.DataFrame({
            'feature': self.feature_names,
            'score': mean_scores
        }).sort_values('score', ascending=False)
        
        # Calculate percentages
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
            title=f"{title} (Top {threshold*100}%)",
            xaxis=dict(
                title="Features",
                tickangle=45
            ),
            yaxis_title="Importance %",
            yaxis2_title="Cumulative %",
            showlegend=True
        )
        
        return fig
    
    def plot_bow_features(self, threshold: float = 0.98) -> go.Figure:
        """Plot BoW features distribution"""
        if self.bow_matrix is None:
            raise ValueError("Must call fit_transform first")
        return self._plot_features(self.bow_matrix, "BoW Features", threshold)
    
    def plot_tfidf_features(self, threshold: float = 0.98) -> go.Figure:
        """Plot TF-IDF features distribution"""
        if self.tfidf_matrix is None:
            raise ValueError("Must call fit_transform first")
        return self._plot_features(self.tfidf_matrix, "TF-IDF Features", threshold)
    
    def plot_feature_comparison(self, threshold: float = 0.98) -> go.Figure:
        """Compare BoW and TF-IDF feature distributions side by side"""
        if self.bow_matrix is None or self.tfidf_matrix is None:
            raise ValueError("Must call fit_transform first")
            
        # Calculate feature importance
        bow_sum = np.array(self.bow_matrix.sum(axis=0)).flatten()
        tfidf_mean = np.array(self.tfidf_matrix.mean(axis=0)).flatten()
        
        # Create and process DataFrame
        feature_scores = pd.DataFrame({
            'feature': self.feature_names,
            'bow_score': bow_sum,
            'tfidf_score': tfidf_mean
        })
        
        # Normalize scores
        feature_scores['bow_pct'] = feature_scores['bow_score'] / feature_scores['bow_score'].sum()
        feature_scores['tfidf_pct'] = feature_scores['tfidf_score'] / feature_scores['tfidf_score'].sum()
        
        # Calculate combined importance and sort by it
        feature_scores['combined_importance'] = feature_scores['bow_pct'] + feature_scores['tfidf_pct']
        feature_scores = feature_scores.sort_values('combined_importance', ascending=False)
        
        # Take top features based on combined importance
        n_features = min(20, len(feature_scores))  # Limit to readable number
        plot_data = feature_scores.copy()
        plot_data = plot_data.head(n_features)
        
        # Calculate difference for color coding
        plot_data['diff'] = plot_data['tfidf_pct'] - plot_data['bow_pct']
        
        # Create figure
        fig = go.Figure()
        
        # Add BoW bars
        fig.add_trace(
            go.Bar(
                name='BoW',
                x=plot_data['feature'],
                y=plot_data['bow_pct'],
                offset=-0.2,
                width=0.4
            )
        )
        
        # Add TF-IDF bars
        fig.add_trace(
            go.Bar(
                name='TF-IDF',
                x=plot_data['feature'],
                y=plot_data['tfidf_pct'],
                offset=0.2,
                width=0.4
            )
        )
    
        
        # Update layout
        fig.update_layout(
            title="BoW vs TF-IDF Top Features by Combined Importance",
            xaxis=dict(
                title="Features",
                tickangle=45
            ),
            yaxis=dict(
                title="Normalized Importance"
            ),
            yaxis2=dict(
                title="TF-IDF - BoW Difference",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            barmode='overlay',
            bargap=0,
            height=600,
            showlegend=True
        )
        
        return fig
        
    def plot_scatter_comparison(self) -> go.Figure:
        """Create scatter plot comparing BoW and TF-IDF weights"""
        if self.bow_matrix is None or self.tfidf_matrix is None:
            raise ValueError("Must call fit_transform first")
            
        # Calculate feature importance
        bow_sum = np.array(self.bow_matrix.sum(axis=0)).flatten()
        tfidf_mean = np.array(self.tfidf_matrix.mean(axis=0)).flatten()
        
        # Normalize scores
        bow_pct = bow_sum / bow_sum.sum()
        tfidf_pct = tfidf_mean / tfidf_mean.sum()
        
        # Calculate difference
        diff = tfidf_pct - bow_pct
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=bow_pct,
                y=tfidf_pct,
                mode='markers',
                marker=dict(
                    size=8,
                    color=diff,
                    colorscale='RdBu',
                    colorbar=dict(title="TF-IDF - BoW"),
                    showscale=True
                ),
                text=self.feature_names,
                hovertemplate='<b>%{text}</b><br>BoW: %{x:.5f}<br>TF-IDF: %{y:.5f}<br>Diff: %{marker.color:.5f}<extra></extra>'
            )
        )
        
        # Add reference line (x=y)
        max_val = max(bow_pct.max(), tfidf_pct.max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Equal Weight'
            )
        )
        
        # Update layout
        fig.update_layout(
            title="TF-IDF vs BoW Feature Importance",
            xaxis_title="BoW Weight (normalized)",
            yaxis_title="TF-IDF Weight (normalized)",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_word_cloud(self, use_tfidf=True, max_words=100, colormap='viridis') -> go.Figure:
        """
        Create a word cloud visualization using the WordCloud library.
        
        Args:
            use_tfidf: Whether to use TF-IDF weights (True) or BoW counts (False)
            max_words: Maximum number of words to include
            colormap: Matplotlib colormap for word colors
        
        Returns:
            Plotly Figure with word cloud visualization
        """
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
        import numpy as np
        
        if self.bow_matrix is None or self.tfidf_matrix is None:
            raise ValueError("Must call fit_transform first")
        
        # Get word importance based on selected method
        if use_tfidf:
            importance = np.array(self.tfidf_matrix.mean(axis=0)).flatten()
            title = "TF-IDF Word Cloud"
        else:
            importance = np.array(self.bow_matrix.sum(axis=0)).flatten()
            title = "Bag of Words Cloud"
        
        # Create dictionary of word frequencies
        word_freq = {word: freq for word, freq in zip(self.feature_names, importance) if freq > 0}
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=500,
            background_color='white',
            max_words=max_words,
            colormap=colormap,
            prefer_horizontal=0.9,
            relative_scaling=1,  # Importance impacts size but not too extremely
            min_font_size=1,
            max_font_size=100
        ).generate_from_frequencies(word_freq)
        
        # Convert word cloud to image
        img_bytes = io.BytesIO()
        wordcloud.to_image().save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Create plotly figure from image
        fig = go.Figure()
        
        # Add image
        fig.add_layout_image(
            dict(
                source=Image.open(img_bytes),
                x=0,
                y=1,
                xref="paper",
                yref="paper",
                sizex=1,
                sizey=1,
                sizing="stretch",
                opacity=1,
                layer="below"
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[0, 1],
                visible=False
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[0, 1],
                visible=False
            ),
            width=800,
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        
        return fig