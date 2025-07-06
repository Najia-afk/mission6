"""
VGG16 Feature Extraction Class for E-commerce Product Classification
Handles deep feature extraction using VGG16 pre-trained model
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm.notebook import tqdm  # Import tqdm for progress bars
import warnings
from typing import Dict, Any, List, Tuple, Union, Optional

warnings.filterwarnings('ignore')

class VGG16FeatureExtractor:
    """
    Feature extraction using VGG16 pre-trained model
    """
    
    def __init__(self, input_shape=(224, 224, 3), layer_name='block5_pool'):
        """
        Initialize the VGG16 feature extractor
        
        Args:
            input_shape (tuple): Input shape for the model (height, width, channels)
            layer_name (str): Name of the layer to extract features from
        """
        self.input_shape = input_shape
        self.layer_name = layer_name
        self.processing_times = []
        self.feature_shape = None
        self.extracted_features = None
        
        # Initialize the VGG16 model
        print("Initializing VGG16 model...")
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
        print(f"Model initialized: Using layer '{layer_name}' for feature extraction")
    
    def extract_features(self, images: List[np.ndarray], batch_size: int = 8) -> np.ndarray:
        """
        Extract features from images using VGG16
        
        Args:
            images: List of preprocessed images
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
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting VGG16 features", total=total_batches, unit="batch"):
            batch = images[i:i+batch_size]
            
            # Convert to numpy array if not already
            batch_array = np.array(batch)
            
            # Ensure batch has correct shape and type
            if batch_array.dtype != np.float32:
                batch_array = batch_array.astype(np.float32)
            
            # If images are in [0,1] range, convert to [0,255]
            if batch_array.max() <= 1.0:
                batch_array = batch_array * 255.0
            
            # Preprocess for VGG16
            batch_processed = preprocess_input(batch_array)
            
            # Extract features
            batch_features = self.model.predict(batch_processed, verbose=0)
            
            # Flatten features
            batch_features_flat = batch_features.reshape(batch_features.shape[0], -1)
            features_list.append(batch_features_flat)
            
            # Record processing time for each image
            current_time = time.time()
            batch_time = (current_time - start_time) / len(batch)
            self.processing_times.extend([batch_time] * len(batch))
            start_time = current_time
        
        # Combine all batches
        features = np.vstack(features_list)
        self.feature_shape = features.shape
        self.extracted_features = features  # Store the extracted features
        
        print(f"Features extracted: Shape={features.shape}")
        return features
    
    def apply_dimensionality_reduction(
        self, 
        features: np.ndarray, 
        n_components: int = 50, 
        method: str = 'pca'
    ) -> Tuple[np.ndarray, Any, StandardScaler]:
        """
        Apply dimensionality reduction to features
        
        Args:
            features: Feature matrix
            n_components: Number of components to keep
            method: Dimensionality reduction method ('pca' or 'tsne')
            
        Returns:
            Tuple containing:
                - Reduced features
                - PCA/TSNE object
                - Scaler object
        """
        if features.shape[0] == 0:
            print("No features to reduce")
            return np.array([]), None, None
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            # Ensure n_components is not larger than the number of samples or features
            n_components = min(n_components, min(features.shape))
            
            # Apply PCA with progress tracking
            print(f"Applying PCA to reduce dimensions from {features.shape[1]} to {n_components}...")
            reducer = PCA(n_components=n_components)
            reduced_features = reducer.fit_transform(features_scaled)
            
            # Print variance explained
            cumulative_variance = np.sum(reducer.explained_variance_ratio_)
            print(f"PCA completed: {cumulative_variance:.2%} of variance preserved")
            
        elif method.lower() == 'tsne':
            # t-SNE is computationally expensive, so we use a progress bar
            print(f"Applying t-SNE to reduce dimensions to {n_components}...")
            reducer = TSNE(n_components=n_components, random_state=42)
            
            # Show a warning for large datasets
            if features.shape[0] > 1000:
                print(f"Warning: t-SNE on {features.shape[0]} samples may take a long time.")
            
            # Apply t-SNE with a custom progress callback
            # Since t-SNE doesn't support direct progress tracking, we'll just use tqdm as a time indicator
            with tqdm(total=100, desc="t-SNE progress", unit="%") as pbar:
                reduced_features = reducer.fit_transform(features_scaled)
                pbar.update(100)  # Mark as complete
            
            print("t-SNE completed")
            
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        return reduced_features, reducer, scaler
    
    def perform_clustering(
        self, 
        features: np.ndarray, 
        n_clusters: Optional[int] = None, 
        cluster_range: Tuple[int, int] = (2, 10)
    ) -> Dict[str, Any]:
        """
        Perform clustering on features
        
        Args:
            features: Feature matrix
            n_clusters: Number of clusters (if None, will be determined automatically)
            cluster_range: Range of clusters to try if n_clusters is None
            
        Returns:
            Dict containing clustering results
        """
        if features.shape[0] < 2:
            print("Not enough samples for clustering")
            return {
                'n_clusters': 0,
                'labels': np.array([]),
                'silhouette_score': 0.0,
                'inertia': 0.0
            }
        
        # Determine optimal number of clusters if not specified
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
    
    def create_analysis_dashboard(
        self, 
        original_features: np.ndarray, 
        reduced_features: np.ndarray, 
        clustering_results: Dict[str, Any],
        processing_times: List[float],
        pca_info: Optional[Any] = None
    ) -> go.Figure:
        """
        Create a comprehensive analysis dashboard
        
        Args:
            original_features: Original feature matrix
            reduced_features: Dimensionality-reduced features
            clustering_results: Results from perform_clustering
            processing_times: List of processing times per image
            pca_info: PCA object for variance information
            
        Returns:
            plotly.graph_objects.Figure: Dashboard figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Feature Extraction Performance',
                'Clustering Results',
                'VGG16 Processing Summary',
                'Processing Time Distribution'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "scatter"}],
                [{"type": "table"}, {"type": "histogram"}]
            ]
        )
        
        # 1. Performance Indicator
        avg_time = np.mean(processing_times)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=1.0 / avg_time,  # Images per second
                title={'text': "Processing Speed (img/sec)"},
                number={'suffix': " img/sec"},
                gauge={
                    'axis': {'range': [0, max(5, 2.0 / avg_time)]},
                    'bar': {'color': "green" if avg_time < 0.5 else "orange" if avg_time < 1.0 else "red"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgray"},
                        {'range': [1, 2], 'color': "gray"},
                        {'range': [2, 5], 'color': "lightgreen"}
                    ],
                }
            ),
            row=1, col=1
        )
        
        # 2. Clustering Visualization (if we have 2D reduced features)
        if reduced_features.shape[1] >= 2 and 'labels' in clustering_results:
            # Use the first two dimensions for visualization
            x = reduced_features[:, 0]
            y = reduced_features[:, 1]
            colors = clustering_results['labels']
            
            fig.add_trace(
                go.Scatter(
                    x=x, y=y, 
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=colors,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Cluster')
                    ),
                    name='Clusters'
                ),
                row=1, col=2
            )
            
            # Add cluster centers if available
            if 'cluster_centers' in clustering_results and reduced_features.shape[1] >= 2:
                centers = clustering_results['cluster_centers']
                if centers.shape[1] >= 2:
                    fig.add_trace(
                        go.Scatter(
                            x=centers[:, 0], 
                            y=centers[:, 1],
                            mode='markers',
                            marker=dict(
                                size=14,
                                color='red',
                                symbol='x'
                            ),
                            name='Cluster Centers'
                        ),
                        row=1, col=2
                    )
        
        # 3. VGG16 Processing Summary Table
        # Prepare summary data
        variance_preserved = pca_info.explained_variance_ratio_.sum() if pca_info else 0.0
        compression_ratio = original_features.shape[1] / reduced_features.shape[1] if reduced_features.shape[1] > 0 else 0
        
        summary_data = [
            ['Original Feature Dimensions', f"{original_features.shape[1]:,}"],
            ['PCA Reduced Dimensions', f"{reduced_features.shape[1]:,}"],
            ['Samples Processed', f"{original_features.shape[0]:,}"],
            ['Compression Ratio', f"{compression_ratio:.1f}x"],
            ['Variance Preserved', f"{variance_preserved:.1%}"],
            ['Optimal Clusters', f"{clustering_results['n_clusters']}"],
            ['Silhouette Score', f"{clustering_results['silhouette_score']:.3f}"],
            ['Avg Processing Time', f"{np.mean(processing_times):.3f}s/image"],
            ['Processing Speed', f"{1/np.mean(processing_times):.1f} img/sec"],
            ['Model Layer Used', f"{self.layer_name}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    fill_color='lightblue',
                    align='center',
                    font=dict(size=12, color='black')
                ),
                cells=dict(
                    values=[[row[0] for row in summary_data],
                           [row[1] for row in summary_data]],
                    fill_color='white',
                    align='center',
                    font=dict(size=11)
                )
            ),
            row=2, col=1
        )
        
        # 4. Processing Time Histogram
        fig.add_trace(
            go.Histogram(
                x=processing_times,
                nbinsx=20,
                marker_color='purple',
                name='Processing Times'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='VGG16 Feature Extraction Dashboard',
            template='plotly_white',
            showlegend=False,
            width=1000,
            height=800
        )
        
        return fig
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the feature extraction process
        
        Returns:
            Dict containing summary information
        """
        return {
            'feature_shape': self.feature_shape,
            'samples_processed': len(self.processing_times),
            'processing_times': {
                'mean': np.mean(self.processing_times),
                'median': np.median(self.processing_times),
                'min': min(self.processing_times) if self.processing_times else 0,
                'max': max(self.processing_times) if self.processing_times else 0,
                'total': sum(self.processing_times)
            },
            'layer_used': self.layer_name,
            'input_shape': self.input_shape
        }
        # 6. Summary table
        summary_data = [
            ['Original Dimensions', f"{features.shape[1]:,}"],
            ['Reduced Dimensions', f"{reduced_features.shape[1]:,}"],
            ['Compression Ratio', f"{compression_ratio:.1f}x"],
            ['Number of Clusters', clustering_results['n_clusters']],
            ['Silhouette Score', f"{clustering_results['silhouette_score']:.3f}"],
            ['Samples Processed', features.shape[0]]
        ]
        
        if feature_times:
            summary_data.append(['Avg Processing Time', f"{np.mean(feature_times):.3f}s"])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color='lightblue',
                           align='center'),
                cells=dict(values=[[row[0] for row in summary_data],
                                  [row[1] for row in summary_data]],
                          fill_color='white',
                          align='center')
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            title='VGG16 Deep Learning Feature Analysis Dashboard',
            template='plotly_white',
            showlegend=False,
            width=1200,
            height=900
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Feature Type", row=1, col=1)
        fig.update_yaxes(title_text="Dimensions", row=1, col=1)
        fig.update_xaxes(title_text="Number of Clusters", row=1, col=2)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        fig.update_xaxes(title_text="t-SNE 1", row=2, col=1)
        fig.update_yaxes(title_text="t-SNE 2", row=2, col=1)
        fig.update_xaxes(title_text="Cluster", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        fig.update_xaxes(title_text="Metric", row=3, col=1)
        fig.update_yaxes(title_text="Time (seconds)", row=3, col=1)
        
        return fig
    
    def get_feature_summary(self):
        """
        Get a summary of feature extraction results
        
        Returns:
            dict: Summary statistics
        """
        if self.extracted_features is None:
            return {'status': 'no_features', 'message': 'No features extracted yet'}
        
        summary = {
            'feature_shape': self.extracted_features.shape,
            'total_features': self.extracted_features.shape[1],
            'samples_processed': self.extracted_features.shape[0],
            'model_layer': self.layer_name,
            'processing_times': {
                'mean': np.mean(self.processing_times) if self.processing_times else 0,
                'std': np.std(self.processing_times) if self.processing_times else 0,
                'min': np.min(self.processing_times) if self.processing_times else 0,
                'max': np.max(self.processing_times) if self.processing_times else 0
            },
            'feature_statistics': {
                'mean': np.mean(self.extracted_features),
                'std': np.std(self.extracted_features),
                'min': np.min(self.extracted_features),
                'max': np.max(self.extracted_features)
            }
        }
        
        return summary
