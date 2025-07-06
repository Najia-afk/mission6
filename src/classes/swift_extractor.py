"""
SWIFT (CLIP-based) Feature Extraction Class for E-commerce Product Classification
Handles vision-language feature extraction using CLIP pre-trained model
"""

import time
import numpy as np
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from tqdm.notebook import tqdm
import warnings
from typing import Dict, Any, List, Tuple, Union, Optional
import pandas as pd

warnings.filterwarnings('ignore')

class SWIFTFeatureExtractor:
    """
    Feature extraction using CLIP (SWIFT approach) pre-trained model
    """
    
    def __init__(self, model_name='ViT-B/32', device=None):
        """
        Initialize the SWIFT feature extractor using CLIP
        
        Args:
            model_name (str): CLIP model variant to use
            device: Device to run on (auto-detected if None)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processing_times = []
        self.feature_shape = None
        self.extracted_features = None
        
        # Initialize the CLIP model
        print(f"Initializing CLIP model '{model_name}' on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        print(f"Model initialized: Using CLIP {model_name} for feature extraction")
    
    def extract_features(self, images: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """
        Extract features from images using CLIP
        
        Args:
            images: List of preprocessed images (numpy arrays)
            batch_size: Batch size for processing
            
        Returns:
            np.ndarray: Extracted features
        """
        if not images:
            print("No images provided for feature extraction")
            return np.array([])
        
        start_time = time.time()
        features_list = []
        
        # Process images in batches with progress bar
        total_batches = (len(images) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting CLIP features", total=total_batches, unit="batch"):
            batch = images[i:i+batch_size]
            
            # Convert numpy arrays to PIL Images and preprocess
            batch_tensors = []
            for img_array in batch:
                # Convert numpy array to PIL Image
                if img_array.dtype != np.uint8:
                    img_array = (img_array * 255).astype(np.uint8)
                
                if len(img_array.shape) == 3:
                    pil_image = Image.fromarray(img_array)
                else:
                    # Handle grayscale
                    pil_image = Image.fromarray(img_array, mode='L').convert('RGB')
                
                # Apply CLIP preprocessing
                preprocessed = self.preprocess(pil_image)
                batch_tensors.append(preprocessed)
            
            # Stack into batch tensor
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extract features
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_tensor)
                # Normalize features (standard for CLIP)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                batch_features = batch_features.cpu().numpy()
            
            features_list.append(batch_features)
            
            # Record processing time for each image
            current_time = time.time()
            batch_time = (current_time - start_time) / len(batch)
            self.processing_times.extend([batch_time] * len(batch))
            start_time = current_time
        
        # Combine all batches
        if features_list:
            self.extracted_features = np.vstack(features_list)
            self.feature_shape = self.extracted_features.shape
            print(f"✅ Feature extraction complete: {self.feature_shape}")
            return self.extracted_features
        else:
            return np.array([])
    
    def apply_dimensionality_reduction(self, features: np.ndarray, variance_threshold: float = 0.95) -> Tuple[np.ndarray, Any]:
        """
        Apply PCA dimensionality reduction
        
        Args:
            features: Feature matrix
            variance_threshold: Variance to preserve
            
        Returns:
            Tuple of (reduced_features, pca_object)
        """
        print(f"🔄 Applying PCA dimensionality reduction (preserving {variance_threshold:.1%} variance)...")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply PCA
        pca = PCA(n_components=variance_threshold)
        features_reduced = pca.fit_transform(features_scaled)
        
        print(f"   Original dimensions: {features.shape[1]:,}")
        print(f"   Reduced dimensions: {features_reduced.shape[1]:,}")
        print(f"   Variance explained: {pca.explained_variance_ratio_.sum():.3f}")
        print(f"   Compression ratio: {features.shape[1] / features_reduced.shape[1]:.1f}x")
        
        return features_reduced, pca
    
    def perform_clustering(self, features: np.ndarray, n_clusters: Optional[int] = None, cluster_range: Tuple[int, int] = (2, 8)) -> Dict[str, Any]:
        """
        Perform clustering analysis on features
        
        Args:
            features: Feature matrix
            n_clusters: Number of clusters (auto-determined if None)
            cluster_range: Range of cluster numbers to test
            
        Returns:
            Dictionary with clustering results
        """
        print(f"🎯 Performing clustering analysis...")
        
        if n_clusters is None:
            print(f"Finding optimal number of clusters in range {cluster_range}...")
            silhouette_scores = []
            inertias = []
            
            # Try different numbers of clusters with progress bar
            for k in tqdm(range(cluster_range[0], cluster_range[1] + 1), desc="Testing cluster counts", unit="k"):
                if k >= features.shape[0]:
                    print(f"Skipping k={k}: more clusters than samples")
                    continue
                    
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                inertias.append(kmeans.inertia_)
                
                # Calculate silhouette score if we have enough samples and clusters
                if features.shape[0] > k and k > 1:
                    silhouette_avg = silhouette_score(features, labels)
                    silhouette_scores.append(silhouette_avg)
                else:
                    silhouette_scores.append(0)
            
            # Find optimal number of clusters
            if silhouette_scores:
                n_clusters = cluster_range[0] + np.argmax(silhouette_scores)
                print(f"Optimal number of clusters: {n_clusters} (silhouette score: {max(silhouette_scores):.3f})")
            else:
                n_clusters = cluster_range[0]
                print(f"Using default number of clusters: {n_clusters}")
        
        # Perform final clustering
        print(f"Performing KMeans clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Calculate clustering metrics
        silhouette_avg = silhouette_score(features, labels) if n_clusters > 1 and features.shape[0] > n_clusters else 0
        
        print(f"Clustering completed: {n_clusters} clusters, silhouette score: {silhouette_avg:.3f}")
        
        return {
            'n_clusters': n_clusters,
            'labels': labels,
            'silhouette_score': silhouette_avg,
            'inertia': kmeans.inertia_,
            'cluster_centers': kmeans.cluster_centers_
        }
    
    def compare_with_categories(self, df, tsne_features, clustering_results, reducer=None):
        """
        Comprehensive analysis comparing SWIFT clustering with real product categories
        
        Args:
            df: DataFrame containing product data
            tsne_features: t-SNE reduced features
            clustering_results: Results from clustering
            reducer: Optional dimensionality reducer for additional analysis
            
        Returns:
            dict: Analysis results with visualizations and metrics
        """
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        print("🔍 SWIFT Analysis: Comparing clustering with real product categories...")
        
        # Extract cluster labels
        swift_cluster_labels = clustering_results['labels']
        
        # Extract real categories for the processed images
        swift_categories = []
        for i in range(len(swift_cluster_labels)):
            if i < len(df):
                # Use the sanitized product_category column
                category = df.iloc[i]['product_category']
                swift_categories.append(category)
            else:
                swift_categories.append('Unknown')
        
        swift_categories = np.array(swift_categories)
        
        # Calculate ARI
        swift_ari = adjusted_rand_score(swift_categories, swift_cluster_labels)
        
        print(f"📊 SWIFT processed {len(swift_cluster_labels)} images")
        print(f"📋 Extracted {len(swift_categories)} categories")
        print(f"📂 Unique categories: {len(np.unique(swift_categories))}")
        print(f"🎯 Adjusted Rand Index(ARI): {swift_ari:.4f}")
        print(f"🔗 Cluster quality (Silhouette): {clustering_results['silhouette_score']:.3f}")
        print(f"📊 Number of clusters: {len(np.unique(swift_cluster_labels))}")
        print(f"💡 Interpretation: {'Good alignment' if swift_ari > 0.5 else 'Moderate alignment' if swift_ari > 0.2 else 'Poor alignment'}")
        
        # Category distribution
        print(f"\n🏷️ Category distribution:")
        unique_cats, counts = np.unique(swift_categories, return_counts=True)
        for cat, count in zip(unique_cats, counts):
            print(f"   {cat}: {count} images")
        
        # Create DataFrame for visualizations
        swift_tsne_df = pd.DataFrame({
            't-SNE1': tsne_features[:, 0],
            't-SNE2': tsne_features[:, 1],
            'Category': swift_categories,
            'Cluster': swift_cluster_labels
        })
        
        # 1. t-SNE visualization colored by real categories
        swift_tsne_fig = px.scatter(
            swift_tsne_df, 
            x='t-SNE1', 
            y='t-SNE2', 
            color='Category',
            title='🚀 SWIFT (CLIP) Features: t-SNE Visualization by Product Categories',
            hover_data={'Cluster': True},
            labels={
                't-SNE1': 't-SNE Component 1',
                't-SNE2': 't-SNE Component 2'
            },
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        swift_tsne_fig.update_layout(
            width=1000,
            height=700,
            template='plotly_white',
            title_x=0.5,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        swift_tsne_fig.add_annotation(
            text=f"📊 {len(swift_categories)} images • {len(np.unique(swift_categories))} categories • {len(np.unique(swift_cluster_labels))} clusters<br>"
                 f"🎯 ARI Score: {swift_ari:.4f} • Silhouette Score: {clustering_results['silhouette_score']:.3f}",
            xref="paper", yref="paper",
            x=0.5, y=-0.1, xanchor='center', yanchor='top',
            showarrow=False, 
            font=dict(size=12, color="gray"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
        
        print("🔍 SWIFT t-SNE Visualization:")
        swift_tsne_fig.show()
        
        # 2. Side-by-side comparison: Categories vs Clusters
        print("\n📊 Creating side-by-side comparison: Real Categories vs SWIFT Clusters...")
        comparison_fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                '🏷️ Colored by Real Product Categories',
                '🚀 Colored by SWIFT Clusters'
            ],
            horizontal_spacing=0.1
        )
        
        # Left plot: Real categories
        for i, category in enumerate(np.unique(swift_categories)):
            mask = swift_categories == category
            comparison_fig.add_trace(
                go.Scatter(
                    x=tsne_features[mask, 0],
                    y=tsne_features[mask, 1],
                    mode='markers',
                    name=category,
                    marker=dict(
                        size=8,
                        color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)],
                        line=dict(width=1, color='white')
                    ),
                    showlegend=True,
                    legendgroup='categories'
                ),
                row=1, col=1
            )
        
        # Right plot: Clusters
        unique_clusters = np.unique(swift_cluster_labels)
        cluster_colors = px.colors.qualitative.Dark2
        
        for i, cluster in enumerate(unique_clusters):
            mask = swift_cluster_labels == cluster
            comparison_fig.add_trace(
                go.Scatter(
                    x=tsne_features[mask, 0],
                    y=tsne_features[mask, 1],
                    mode='markers',
                    name=f'Cluster {cluster}',
                    marker=dict(
                        size=8,
                        color=cluster_colors[i % len(cluster_colors)],
                        line=dict(width=1, color='white')
                    ),
                    showlegend=True,
                    legendgroup='clusters'
                ),
                row=1, col=2
            )
        
        comparison_fig.update_layout(
            title='🔍 SWIFT (CLIP) Features: t-SNE Analysis Comparison',
            title_x=0.5,
            width=1400,
            height=600,
            template='plotly_white',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        comparison_fig.update_xaxes(title_text="t-SNE Component 1")
        comparison_fig.update_yaxes(title_text="t-SNE Component 2")
        
        comparison_fig.add_annotation(
            text=f"📈 SWIFT Performance: ARI = {swift_ari:.4f} • Silhouette = {clustering_results['silhouette_score']:.3f}<br>"
                 f"💡 {'Good alignment' if swift_ari > 0.5 else 'Moderate alignment' if swift_ari > 0.2 else 'Poor alignment'} between clusters and true categories",
            xref="paper", yref="paper",
            x=0.5, y=-0.12, xanchor='center', yanchor='top',
            showarrow=False, 
            font=dict(size=12, color="gray"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1
        )
        
        print("🔍 SWIFT Side-by-Side Comparison:")
        comparison_fig.show()
        
        # Return comprehensive results
        return {
            'ari_score': swift_ari,
            'tsne_fig': swift_tsne_fig,
            'comparison_fig': comparison_fig,
            'clustering_results': clustering_results,
            'categories': swift_categories,
            'cluster_labels': swift_cluster_labels,
            'tsne_data': swift_tsne_df,
            'category_distribution': dict(zip(unique_cats, counts)),
            'n_categories': len(np.unique(swift_categories)),
            'n_clusters': len(unique_clusters),
            'silhouette_score': clustering_results['silhouette_score']
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the feature extraction
        
        Returns:
            dict: Summary statistics
        """
        if self.extracted_features is None:
            return {"error": "No features extracted yet"}
        
        summary = {
            'model_name': self.model_name,
            'device': self.device,
            'feature_shape': self.feature_shape,
            'total_images': len(self.processing_times),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'total_processing_time': sum(self.processing_times),
            'feature_dimensionality': self.extracted_features.shape[1] if self.extracted_features is not None else 0
        }
        
        return summary
