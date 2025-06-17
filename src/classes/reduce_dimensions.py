import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn import metrics  # Add this import for adjusted_rand_score
from typing import Dict, Any, Tuple, Optional, List, Union

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
        
    def get_cluster_color(self, cluster_idx: int, colorscale: str = 'viridis', transparent: bool = False) -> str:
        """Get a consistent color for a cluster index"""
        import plotly.colors
        
        # Define a set of colors for different clusters
        # Use Plotly's qualitative colors if available, otherwise fallback to predefined list
        try:
            if hasattr(plotly.colors.qualitative, 'Plotly'):
                colors = plotly.colors.qualitative.Plotly
            else:
                # Fallback to a hardcoded color list
                colors = [
                    '#1f77b4',  # muted blue
                    '#ff7f0e',  # safety orange
                    '#2ca02c',  # cooked asparagus green
                    '#d62728',  # brick red
                    '#9467bd',  # muted purple
                    '#8c564b',  # chestnut brown
                    '#e377c2',  # raspberry yogurt pink
                    '#7f7f7f',  # middle gray
                    '#bcbd22',  # curry yellow-green
                    '#17becf'   # blue-teal
                ]
        except:
            # Fallback to a hardcoded color list
            colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
        
        # Generate a color for this cluster index
        idx = cluster_idx % len(colors)
        color = colors[idx]
        
        # Add transparency if requested
        if transparent and color.startswith('rgb'):
            color = color.replace('rgb', 'rgba').replace(')', ', 0.5)')
        elif transparent and color.startswith('#'):
            # Convert hex to rgba
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            color = f'rgba({r}, {g}, {b}, 0.5)'
        
        return color
    
    def get_cluster_name(self, cluster_idx: int) -> str:
        """Get a consistent name for a cluster index"""
        return f"Cluster {cluster_idx}"
    
    def cluster_data(self, X: np.ndarray, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Cluster data using KMeans
        
        Parameters:
        -----------
        X : array-like
            Data to cluster
        n_clusters : int
            Number of clusters
            
        Returns:
        --------
        results : dict
            Dictionary with clustering results
        """
        # Convert to dense array if sparse
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_dense)
        
        return {
            'model': kmeans,
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_
        }
    
    def plot_silhouette(self, X: np.ndarray, labels: Union[np.ndarray, pd.Series], 
                         figsize: Tuple[int, int] = (1200,800)) -> go.Figure:
        """
        Create a silhouette plot to visualize cluster quality using Plotly.
        
        Parameters:
        -----------
        X : array-like
            Input data
        labels : array-like
            Cluster labels for each sample
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        fig : plotly Figure
            Silhouette plot visualization
        """
        # Convert inputs to numpy arrays
        if hasattr(X, 'toarray'):
            X = X.toarray()
        if isinstance(labels, pd.Series):
            labels_array = labels.values
        else:
            labels_array = np.array(labels)
            
        # Handle string labels by mapping to integers
        if labels_array.dtype == np.dtype('O'):
            unique_labels = np.unique(labels_array)
            label_to_int = {label: i for i, label in enumerate(unique_labels)}
            int_labels = np.array([label_to_int[label] for label in labels_array])
        else:
            int_labels = labels_array
            
        # Get number of clusters
        n_clusters = len(np.unique(int_labels))
        if n_clusters < 2:
            raise ValueError("Need at least 2 clusters for silhouette analysis")
            
        # Get cluster sizes
        cluster_counts = np.bincount(int_labels)
        cluster_proportions = cluster_counts / cluster_counts.sum()
        
        # Sample data if needed for faster silhouette calculation
        max_samples = 20000
        if len(X) > max_samples:
            print(f"Sampling {max_samples} records from {len(X)} for faster silhouette calculation...")
            
            # Stratified sampling to maintain cluster proportions
            indices = []
            for i in range(n_clusters):
                cluster_indices = np.where(int_labels == i)[0]
                n_samples = int(max_samples * cluster_proportions[i])
                if n_samples > 0:
                    cluster_sample = np.random.choice(
                        cluster_indices, 
                        size=min(n_samples, len(cluster_indices)), 
                        replace=False
                    )
                    indices.extend(cluster_sample)
            
            # Ensure we have enough samples
            if len(indices) < max_samples:
                remaining = max_samples - len(indices)
                all_indices = set(range(len(X)))
                remaining_indices = list(all_indices - set(indices))
                if remaining_indices:
                    extra_indices = np.random.choice(
                        remaining_indices,
                        size=min(remaining, len(remaining_indices)),
                        replace=False
                    )
                    indices.extend(extra_indices)
                    
            X_sample = X[indices]
            sample_int_labels = int_labels[indices]
            sample_orig_labels = labels_array[indices] if labels_array.dtype == np.dtype('O') else None
        else:
            X_sample = X
            sample_int_labels = int_labels
            sample_orig_labels = labels_array if labels_array.dtype == np.dtype('O') else None
        
        # Calculate silhouette scores
        silhouette_vals = silhouette_samples(X_sample, sample_int_labels)
        
        # Calculate average silhouette score
        avg_score = np.mean(silhouette_vals)
        
        # Create a DataFrame for visualization
        silhouette_df = pd.DataFrame({
            'sample_idx': range(len(silhouette_vals)),
            'cluster': sample_int_labels,
            'silhouette_val': silhouette_vals
        })
        
        # If we have original string labels, add them to the dataframe
        if sample_orig_labels is not None:
            silhouette_df['cluster_name'] = sample_orig_labels
        else:
            silhouette_df['cluster_name'] = silhouette_df['cluster'].apply(lambda x: f"Cluster {x}")
        
        # Sort within each cluster for better visualization
        silhouette_df = silhouette_df.sort_values(['cluster', 'silhouette_val'])
        
        # Create figure
        fig = go.Figure()
        
        # Add silhouette traces for each cluster
        total_height = figsize[1] * 0.8  # 80% of figure height for the plots
        y_lower = 10
        
        for i in range(n_clusters):
            # Get silhouette values for current cluster
            cluster_df = silhouette_df[silhouette_df['cluster'] == i]
            if len(cluster_df) == 0:
                continue
                
            cluster_silhouette_vals = cluster_df['silhouette_val'].sort_values()
            
            # Get cluster name (use first name in the cluster)
            if sample_orig_labels is not None:
                cluster_name = cluster_df['cluster_name'].iloc[0]
            else:
                cluster_name = f"Cluster {i}"
            
            # Calculate height based on proportion
            cluster_height = total_height * cluster_proportions[i]
            
            # Calculate y positions
            y_upper = y_lower + cluster_height
            y_positions = np.linspace(y_lower, y_upper - 1, len(cluster_silhouette_vals))
            
            # Use consistent colors
            fill_color = self.get_cluster_color(i, transparent=True)
            line_color = self.get_cluster_color(i)
            
            # Add the silhouette plot for this cluster
            fig.add_trace(
                go.Scatter(
                    x=cluster_silhouette_vals,
                    y=y_positions,
                    mode='lines',
                    line=dict(width=0.5, color=line_color),
                    fill='tozerox',
                    fillcolor=fill_color,
                    name=f"{cluster_name} ({cluster_counts[i]} samples, {cluster_proportions[i]:.1%})"
                )
            )
            
            # Update y_lower for next cluster
            y_lower = y_upper + 5
        
        # Add a vertical line for the average silhouette score
        fig.add_vline(
            x=avg_score, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Avg Silhouette: {avg_score:.3f}",
            annotation_position="top right"
        )
        
        # Update layout
        fig.update_layout(
            title=f'Silhouette Analysis for Clustering (n={n_clusters} clusters)',
            xaxis_title='Silhouette Coefficient',
            yaxis_title='Cluster Distribution',
            height=figsize[1],
            showlegend=True,
            xaxis=dict(range=[-0.1, 1.05]),
            yaxis=dict(showticklabels=False)
        )
        
        return fig
    
    def plot_intercluster_distance(self, X: np.ndarray, labels: Union[np.ndarray, pd.Series] = None, 
                                   n_clusters: int = 5, figsize: Tuple[int, int] = (1200, 900)) -> go.Figure:
        """
        Create a circle-based visualization showing relationships between cluster centers.
        
        Parameters:
        -----------
        X : array-like
            Input data
        labels : array-like, optional
            Cluster labels (if None, KMeans will be run)
        n_clusters : int
            Number of clusters (used only if labels is None)
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        fig : plotly Figure
            Intercluster distance visualization
        """
        # Convert to dense array if sparse
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        
        # If labels not provided, run clustering
        if labels is None:
            kmeans_results = self.cluster_data(X_dense, n_clusters=n_clusters)
            centers = kmeans_results['centers']
            int_labels = kmeans_results['labels']
            label_names = None
        else:
            # If labels provided, compute centers
            if isinstance(labels, pd.Series):
                labels_array = labels.values
            else:
                labels_array = np.array(labels)
                
            # Handle string labels by mapping to integers
            if labels_array.dtype == np.dtype('O'):
                unique_labels = np.unique(labels_array)
                label_to_int = {label: i for i, label in enumerate(unique_labels)}
                int_labels = np.array([label_to_int[label] for label in labels_array])
                label_names = unique_labels
            else:
                int_labels = labels_array
                label_names = None
                
            n_clusters = len(np.unique(int_labels))
            centers = np.zeros((n_clusters, X_dense.shape[1]))
            
            for i in range(n_clusters):
                mask = (int_labels == i)
                if np.sum(mask) > 0:
                    centers[i] = np.mean(X_dense[mask], axis=0)
        
        # Get cluster sizes
        cluster_sizes = np.bincount(int_labels)
        size_scale = 100  # Max circle size
        sizes = (cluster_sizes / cluster_sizes.max()) * size_scale + 20  # Add minimum size
        
        # Compute pairwise distances between centers
        distances = pdist(centers)
        distance_matrix = squareform(distances)
        
        # Use MDS to position clusters in 2D space based on their distances
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        positions = mds.fit_transform(distance_matrix)
        
        # Create figure
        fig = go.Figure()
        
        # Add circles for each cluster
        for i in range(n_clusters):
            if cluster_sizes[i] > 0:
                # Get cluster name
                if label_names is not None:
                    cluster_name = label_names[i]
                else:
                    cluster_name = f"Cluster {i}"
                    
                fig.add_trace(go.Scatter(
                    x=[positions[i, 0]],
                    y=[positions[i, 1]],
                    mode='markers',
                    marker=dict(
                        size=sizes[i],
                        color=self.get_cluster_color(i),
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    name=f"{cluster_name} (n={cluster_sizes[i]})",
                    text=[f"{cluster_name}: {cluster_sizes[i]} samples"],
                    hoverinfo='text'
                ))
        
        # Add lines between clusters with distance labels
        max_dist = np.max(distances) if len(distances) > 0 else 1.0
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                # Skip if either cluster is empty
                if cluster_sizes[i] == 0 or cluster_sizes[j] == 0:
                    continue
                    
                dist = distance_matrix[i, j]
                normalized_dist = dist/max_dist
                
                # Calculate line points
                line_color = self.get_cluster_color(i)
                width = 1.5 + 3 * (1 - normalized_dist)  # Thicker for closer clusters
                
                # Add line with distance
                fig.add_trace(go.Scatter(
                    x=[positions[i, 0], positions[j, 0]],
                    y=[positions[i, 1], positions[j, 1]],
                    mode='lines',
                    line=dict(
                        color=line_color,
                        width=width,
                        dash='solid' if normalized_dist < 0.5 else 'dot'
                    ),
                    hovertext=[f'Distance: {dist:.2f}'],
                    showlegend=False
                ))
                
                # Add text at midpoint
                mid_x = (positions[i, 0] + positions[j, 0]) / 2
                mid_y = (positions[i, 1] + positions[j, 1]) / 2
                
                fig.add_trace(go.Scatter(
                    x=[mid_x],
                    y=[mid_y],
                    mode='text',
                    text=[f"{dist:.2f}"],
                    textposition='middle center',
                    textfont=dict(
                        color=line_color,
                        size=10
                    ),
                    hoverinfo='none',
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title="Intercluster Distance Visualization",
            height=figsize[1],
            showlegend=True,
            hovermode='closest',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        
        return fig
    
    def evaluate_clustering(self, X: np.ndarray, true_labels: Union[np.ndarray, pd.Series], 
                       n_clusters: int = None, use_tsne: bool = True) -> Dict[str, Any]:
        """
        Perform clustering and evaluate against true labels using Adjusted Rand Index.
        
        Parameters:
        -----------
        X : array-like
            Input data for clustering
        true_labels : array-like
            True category labels
        n_clusters : int, optional
            Number of clusters (defaults to number of unique true labels)
        use_tsne : bool
            Whether to cluster on t-SNE results (True) or original data (False)
            
        Returns:
        --------
        result : dict
            Dictionary containing cluster evaluation results and dataframe
        """
        # Convert to array/Series as needed
        if isinstance(true_labels, pd.Series):
            true_labels_array = true_labels.values
        else:
            true_labels_array = np.array(true_labels)
        
        # Determine number of clusters if not specified
        if n_clusters is None:
            n_clusters = len(np.unique(true_labels_array))
        
        # Decide what data to cluster
        if use_tsne:
            # Apply t-SNE if not already done
            if self.tsne_results is None:
                print("Running t-SNE dimensionality reduction...")
                self.fit_transform_tsne(X)
            
            # Use t-SNE results for clustering
            cluster_data = self.tsne_results
        else:
            # Use original data (dense format if sparse)
            cluster_data = X.toarray() if hasattr(X, 'toarray') else X
        
        # Perform K-means clustering
        print(f"Clustering into {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cluster_data)
        
        # Create dataframe with results
        df_result = pd.DataFrame()
        
        # Add t-SNE coordinates if available
        if self.tsne_results is not None:
            df_result['tsne1'] = self.tsne_results[:, 0]
            df_result['tsne2'] = self.tsne_results[:, 1]
        
        # Add cluster assignments
        df_result['cluster'] = cluster_labels
        
        # Add true labels
        if isinstance(true_labels, pd.Series):
            df_result['true_category'] = true_labels.values
        else:
            df_result['true_category'] = true_labels
        
        # Calculate Adjusted Rand Index
        ari_score = metrics.adjusted_rand_score(true_labels_array, cluster_labels)
        
        # Calculate cluster composition as percentages
        cluster_category_distribution = pd.crosstab(
            df_result["cluster"], 
            df_result["true_category"], 
            normalize='index'
        ) * 100  # Convert to percentage
        
        return {
            'dataframe': df_result,
            'ari_score': ari_score,
            'cluster_distribution': cluster_category_distribution,
            'kmeans_model': kmeans
        }
    
    def plot_cluster_category_heatmap(self, cluster_distribution: pd.DataFrame, 
                                     figsize: Tuple[int, int] = (800, 600)) -> go.Figure:
        """
        Create a heatmap visualization of cluster composition by true categories.
        
        Parameters:
        -----------
        cluster_distribution : DataFrame
            Crosstab of clusters vs categories (output from evaluate_clustering)
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        fig : plotly Figure
            Heatmap visualization
        """
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=cluster_distribution.values,
            x=cluster_distribution.columns,
            y=cluster_distribution.index,
            colorscale='YlGnBu',
            text=cluster_distribution.round(1).values,
            texttemplate='%{text:.1f}',
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title='Cluster Composition by True Category (%)',
            xaxis_title='True Category',
            yaxis_title='Cluster',
            height=figsize[1],
            width=figsize[0]
        )
        
        return fig