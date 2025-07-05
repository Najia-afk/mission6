from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

class DimensionalityReducer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pca = None
        self.tsne = None

    def fit_transform_pca(self, data, n_components=2):
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        return self.pca.fit_transform(data.toarray() if hasattr(data, "toarray") else data)

    def fit_transform_tsne(self, data, n_components=2, perplexity=30):
        self.tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=self.random_state)
        return self.tsne.fit_transform(data.toarray() if hasattr(data, "toarray") else data)

    def plot_pca(self, pca_features, labels=None):
        df = pd.DataFrame(pca_features, columns=['PCA1', 'PCA2'])
        df['label'] = labels
        fig = px.scatter(df, x='PCA1', y='PCA2', color='label', title='PCA Projection')
        return fig

    def plot_tsne(self, tsne_features, labels=None):
        df = pd.DataFrame(tsne_features, columns=['t-SNE1', 't-SNE2'])
        df['label'] = labels
        fig = px.scatter(df, x='t-SNE1', y='t-SNE2', color='label', title='t-SNE Projection')
        return fig
