"""
VGG16 Feature Extractor for Deep Learning-based Image Classification
Handles deep feature extraction using pre-trained VGG16 model
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import cv2
import warnings
warnings.filterwarnings('ignore')


class VGG16FeatureExtractor:
    """
    Deep feature extraction using pre-trained VGG16 model
    """
    
    def __init__(self, input_shape=(224, 224, 3), layer_name='block5_pool'):
        """
        Initialize the VGG16 feature extractor
        
        Args:
            input_shape (tuple): Input image shape
            layer_name (str): Layer to extract features from
        """
        self.input_shape = input_shape
        self.layer_name = layer_name
        self.model = None
        self.feature_extractor = None
        self.extracted_features = None
        self.processing_times = []
        
    def load_model(self):
        """
        Load and configure the VGG16 model for feature extraction
        """
        print("Loading VGG16 model...")
        
        # Load pre-trained VGG16 model
        self.model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Create feature extractor from specified layer
        if self.layer_name in [layer.name for layer in self.model.layers]:
            layer_output = self.model.get_layer(self.layer_name).output
            self.feature_extractor = Model(inputs=self.model.input, outputs=layer_output)
        else:
            # Use the last convolutional layer if specified layer not found
            self.feature_extractor = Model(inputs=self.model.input, outputs=self.model.layers[-2].output)
            print(f"Layer '{self.layer_name}' not found, using '{self.model.layers[-2].name}'")
        
        print(f"Feature extractor created using layer: {self.feature_extractor.layers[-1].name}")
        print(f"Output shape: {self.feature_extractor.output_shape}")
        
    def preprocess_images(self, processed_images):
        """
        Preprocess images for VGG16 model
        
        Args:
            processed_images (list): List of preprocessed images
            
        Returns:
            np.ndarray: Preprocessed image batch
        """
        batch = []
        
        for img in processed_images:
            # Ensure image is in the right format
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Image is already RGB
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
            else:
                # Convert grayscale to RGB
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif len(img.shape) == 3 and img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
            
            # Resize to target size if needed
            if img.shape[:2] != self.input_shape[:2]:
                img = cv2.resize(img, self.input_shape[:2])
            
            batch.append(img)
        
        # Convert to numpy array and preprocess for VGG16
        batch = np.array(batch)
        batch = preprocess_input(batch.astype(np.float32))
        
        return batch
    
    def extract_features(self, processed_images, batch_size=32):
        """
        Extract deep features from processed images
        
        Args:
            processed_images (list): List of preprocessed images
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: Extracted features
        """
        if self.feature_extractor is None:
            self.load_model()
        
        print(f"Extracting VGG16 features from {len(processed_images)} images...")
        
        # Preprocess images for VGG16
        image_batch = self.preprocess_images(processed_images)
        
        # Extract features in batches
        all_features = []
        
        for i in range(0, len(image_batch), batch_size):
            batch = image_batch[i:i+batch_size]
            
            # Extract features
            import time
            start_time = time.time()
            features = self.feature_extractor.predict(batch, verbose=0)
            processing_time = time.time() - start_time
            
            # Flatten features for each image
            flattened_features = features.reshape(features.shape[0], -1)
            all_features.append(flattened_features)
            
            # Record processing time
            self.processing_times.extend([processing_time / len(batch)] * len(batch))
            
            print(f"Processed batch {i//batch_size + 1}/{(len(image_batch) + batch_size - 1)//batch_size}")
        
        # Combine all features
        self.extracted_features = np.vstack(all_features)
        
        print(f"Feature extraction complete!")
        print(f"Feature shape: {self.extracted_features.shape}")
        print(f"Average processing time: {np.mean(self.processing_times):.3f}s per image")
        
        return self.extracted_features
    
    def apply_dimensionality_reduction(self, features=None, n_components=50, method='pca'):
        """
        Apply dimensionality reduction to features
        
        Args:
            features (np.ndarray): Features to reduce (uses extracted_features if None)
            n_components (int): Number of components to keep
            method (str): Reduction method ('pca' or 'tsne')
            
        Returns:
            tuple: (reduced_features, reducer_object)
        """
        if features is None:
            features = self.extracted_features
        
        if features is None:
            raise ValueError("No features available. Extract features first.")
        
        print(f"Applying {method.upper()} dimensionality reduction...")
        print(f"Original shape: {features.shape}")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        if method.lower() == 'pca':
            # Ensure n_components doesn't exceed available dimensions
            n_components = min(n_components, min(features.shape) - 1)
            
            reducer = PCA(n_components=n_components)
            reduced_features = reducer.fit_transform(features_scaled)
            
            print(f"PCA completed: {features.shape} -> {reduced_features.shape}")
            print(f"Variance explained: {reducer.explained_variance_ratio_.sum():.3f}")
            
        elif method.lower() == 'tsne':
            # For t-SNE, first apply PCA if needed
            if features.shape[1] > 50:
                pca = PCA(n_components=50)
                features_scaled = pca.fit_transform(features_scaled)
                print(f"Applied PCA preprocessing: {features.shape} -> {features_scaled.shape}")
            
            # Adjust perplexity based on sample size
            perplexity = min(30, len(features_scaled) // 4)
            
            reducer = TSNE(
                n_components=min(n_components, 3),  # t-SNE typically 2D or 3D
                perplexity=perplexity,
                random_state=42,
                n_iter=1000
            )
            reduced_features = reducer.fit_transform(features_scaled)
            
            print(f"t-SNE completed: {features_scaled.shape} -> {reduced_features.shape}")
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return reduced_features, reducer, scaler
    
    def perform_clustering(self, reduced_features, n_clusters=None, cluster_range=(2, 10)):
        """
        Perform clustering on reduced features
        
        Args:
            reduced_features (np.ndarray): Reduced feature space
            n_clusters (int): Number of clusters (if None, finds optimal)
            cluster_range (tuple): Range to search for optimal clusters
            
        Returns:
            dict: Clustering results
        """
        print("Performing clustering analysis...")
        
        if n_clusters is None:
            # Find optimal number of clusters
            silhouette_scores = []
            inertias = []
            
            for k in range(cluster_range[0], min(cluster_range[1] + 1, len(reduced_features))):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(reduced_features)
                
                if len(np.unique(cluster_labels)) > 1:
                    silhouette_avg = silhouette_score(reduced_features, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                    inertias.append(kmeans.inertia_)
                else:
                    silhouette_scores.append(-1)
                    inertias.append(float('inf'))
            
            # Find optimal k
            optimal_k = cluster_range[0] + np.argmax(silhouette_scores)
            print(f"Optimal number of clusters: {optimal_k}")
        else:
            optimal_k = n_clusters
        
        # Perform final clustering
        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        final_labels = final_kmeans.fit_predict(reduced_features)
        final_silhouette = silhouette_score(reduced_features, final_labels)
        
        results = {
            'n_clusters': optimal_k,
            'cluster_labels': final_labels,
            'silhouette_score': final_silhouette,
            'kmeans_model': final_kmeans,
            'cluster_centers': final_kmeans.cluster_centers_
        }
        
        if n_clusters is None:
            results['silhouette_scores'] = silhouette_scores
            results['inertias'] = inertias
            results['cluster_range'] = list(range(cluster_range[0], cluster_range[1] + 1))
        
        print(f"Clustering completed with {optimal_k} clusters")
        print(f"Silhouette score: {final_silhouette:.3f}")
        
        return results
    
    def create_analysis_dashboard(self, features, reduced_features, clustering_results, feature_times=None):
        """
        Create comprehensive analysis dashboard
        
        Args:
            features (np.ndarray): Original features
            reduced_features (np.ndarray): Dimensionally reduced features
            clustering_results (dict): Results from clustering
            feature_times (list): Processing times per image
            
        Returns:
            plotly.graph_objects.Figure: Dashboard figure
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Feature Dimensionality Comparison',
                'Clustering Quality (Silhouette Score)',
                't-SNE Visualization with Clusters',
                'Cluster Size Distribution',
                'Processing Performance',
                'Feature Analysis Summary'
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "table"}]
            ]
        )
        
        # 1. Dimensionality comparison
        dimensions = ['Original Features', 'Reduced Features']
        dim_values = [features.shape[1], reduced_features.shape[1]]
        compression_ratio = features.shape[1] / reduced_features.shape[1]
        
        fig.add_trace(
            go.Bar(
                x=dimensions,
                y=dim_values,
                text=[f'{v:,}' for v in dim_values],
                textposition='auto',
                marker_color=['steelblue', 'darkgreen'],
                name='Dimensions'
            ),
            row=1, col=1
        )
        
        # 2. Clustering quality
        if 'silhouette_scores' in clustering_results:
            fig.add_trace(
                go.Scatter(
                    x=clustering_results['cluster_range'],
                    y=clustering_results['silhouette_scores'],
                    mode='lines+markers',
                    name='Silhouette Score',
                    line=dict(color='red', width=2),
                    marker=dict(size=8)
                ),
                row=1, col=2
            )
        
        # 3. t-SNE visualization with clusters
        if reduced_features.shape[1] >= 2:
            cluster_labels = clustering_results['cluster_labels']
            
            for cluster_id in np.unique(cluster_labels):
                mask = cluster_labels == cluster_id
                fig.add_trace(
                    go.Scatter(
                        x=reduced_features[mask, 0],
                        y=reduced_features[mask, 1],
                        mode='markers',
                        name=f'Cluster {cluster_id}',
                        marker=dict(size=8, opacity=0.7),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 4. Cluster size distribution
        cluster_labels = clustering_results['cluster_labels']
        unique, counts = np.unique(cluster_labels, return_counts=True)
        
        fig.add_trace(
            go.Bar(
                x=[f'Cluster {i}' for i in unique],
                y=counts,
                text=counts,
                textposition='auto',
                marker_color='lightcoral',
                name='Cluster Sizes'
            ),
            row=2, col=2
        )
        
        # 5. Processing performance
        if feature_times is None:
            feature_times = self.processing_times
        
        if feature_times:
            performance_metrics = ['Avg Time (s)', 'Min Time (s)', 'Max Time (s)']
            performance_values = [
                np.mean(feature_times),
                np.min(feature_times),
                np.max(feature_times)
            ]
            
            fig.add_trace(
                go.Bar(
                    x=performance_metrics,
                    y=performance_values,
                    text=[f'{v:.3f}' for v in performance_values],
                    textposition='auto',
                    marker_color='lightblue',
                    name='Performance'
                ),
                row=3, col=1
            )
        
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
