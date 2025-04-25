import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from typing import Dict, Any, Tuple, Optional

class DimensionalityReducer:
    def __init__(self):
        """Initialize dimensionality reduction methods"""
        self.pca = None
        self.tsne = None
        self.pca_results = None
        self.tsne_results = None
        
    def fit_transform_pca(self, X: np.ndarray, n_components: int = 2) -> pd.DataFrame:
        """Apply PCA and return results as DataFrame"""
        self.pca = PCA(n_components=n_components)
        self.pca_results = self.pca.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
        
        # Create DataFrame with results
        columns = [f'PC{i+1}' for i in range(n_components)]
        results_df = pd.DataFrame(self.pca_results, columns=columns)
        
        return results_df
    
    def fit_transform_tsne(self, X: np.ndarray, n_components: int = 2, 
                           perplexity: int = 30) -> pd.DataFrame:
        """Apply t-SNE and return results as DataFrame"""
        self.tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        self.tsne_results = self.tsne.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
        
        # Create DataFrame with results
        columns = [f't-SNE{i+1}' for i in range(n_components)]
        results_df = pd.DataFrame(self.tsne_results, columns=columns)
        
        return results_df
    
    def plot_pca(self, labels: Optional[pd.Series] = None) -> go.Figure:
        """Plot PCA results with optional category labels"""
        if self.pca_results is None:
            raise ValueError("Must run fit_transform_pca first")
            
        # Create plot data
        plot_data = pd.DataFrame(self.pca_results, columns=['PC1', 'PC2'])
        if labels is not None:
            plot_data['Category'] = labels.values
            
        # Create plotly figure
        if labels is not None:
            fig = px.scatter(plot_data, x='PC1', y='PC2', color='Category',
                            title="PCA Visualization")
        else:
            fig = px.scatter(plot_data, x='PC1', y='PC2',
                            title="PCA Visualization")
            
        # Add variance explained
        explained_var_ratio = self.pca.explained_variance_ratio_
        fig.update_layout(
            xaxis_title=f"PC1 ({explained_var_ratio[0]:.2%} variance)",
            yaxis_title=f"PC2 ({explained_var_ratio[1]:.2%} variance)",
        )
        
        return fig
    
    def plot_tsne(self, labels: Optional[pd.Series] = None) -> go.Figure:
        """Plot t-SNE results with optional category labels"""
        if self.tsne_results is None:
            raise ValueError("Must run fit_transform_tsne first")
            
        # Create plot data
        plot_data = pd.DataFrame(self.tsne_results, columns=['t-SNE1', 't-SNE2'])
        if labels is not None:
            plot_data['Category'] = labels.values
            
        # Create plotly figure
        if labels is not None:
            fig = px.scatter(plot_data, x='t-SNE1', y='t-SNE2', color='Category',
                            title="t-SNE Visualization")
        else:
            fig = px.scatter(plot_data, x='t-SNE1', y='t-SNE2',
                            title="t-SNE Visualization")
            
        return fig
    
    def evaluate_separation(self, X: np.ndarray, labels: pd.Series) -> Dict[str, float]:
        """Evaluate cluster separation quality metrics"""
        if len(np.unique(labels)) < 2:
            return {"error": "Need at least 2 unique labels for evaluation"}
            
        # Convert to dense array if sparse
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        
        try:
            # Calculate silhouette score
            silhouette = silhouette_score(X_dense, labels)
            
            # Return metrics
            return {
                "silhouette_score": silhouette,
                "num_clusters": len(np.unique(labels))
            }
        except Exception as e:
            return {"error": str(e)}