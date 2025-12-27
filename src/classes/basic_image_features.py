"""
Basic Image Feature Extractor Class

This module provides comprehensive basic image feature extraction capabilities including:
- SIFT keypoint detection and description
- Local Binary Pattern (LBP) texture analysis
- Gray-Level Co-occurrence Matrix (GLCM) texture properties
- Gabor filter responses
- Patch-based statistical features
- Feature combination and analysis tools

Author: Mission 6 Team
Version: 1.0
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.feature_extraction import image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.notebook import tqdm  # Import tqdm for progress bars
import warnings
warnings.filterwarnings('ignore')


class BasicImageFeatureExtractor:
    """
    Comprehensive basic image feature extraction class.
    
    This class provides methods for extracting various types of basic image features
    including SIFT, LBP, GLCM, Gabor filters, and patch statistics.
    """
    
    def __init__(self, sift_features=128, lbp_radius=1, lbp_points=8, 
                 patch_size=(16, 16), max_patches=25):
        """
        Initialize the feature extractor.
        
        Args:
            sift_features (int): Number of SIFT features to extract
            lbp_radius (int): Radius for LBP calculation
            lbp_points (int): Number of LBP sample points
            patch_size (tuple): Size of patches for statistical features
            max_patches (int): Maximum number of patches to extract
        """
        self.sift_features = sift_features
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points * lbp_radius
        self.patch_size = patch_size
        self.max_patches = max_patches
        
        # Initialize SIFT detector
        try:
            self.sift = cv2.SIFT_create()
            self.sift_available = True
        except Exception:
            self.sift_available = False
            
        # Initialize feature storage
        self.feature_results = {
            'sift_features': [],
            'lbp_features': [],
            'glcm_features': [],
            'gabor_features': [],
            'patch_features': [],
            'image_names': []
        }
        
        # Analysis results
        self.combined_features = None
        self.feature_names = []
        self.scaler = StandardScaler()
        
    def extract_sift_features(self, image):
        """
        Extract SIFT keypoint features from an image.
        
        Args:
            image (np.ndarray): Grayscale image
            
        Returns:
            np.ndarray: SIFT feature vector
        """
        if not self.sift_available:
            return np.zeros(self.sift_features)
            
        try:
            keypoints, descriptors = self.sift.detectAndCompute(image, None)
            
            if descriptors is not None:
                # Aggregate SIFT descriptors (mean of all descriptors)
                sift_feature = np.mean(descriptors, axis=0)
                return sift_feature
            else:
                return np.zeros(self.sift_features)
                
        except Exception:
            return np.zeros(self.sift_features)
    
    def extract_lbp_features(self, image):
        """
        Extract Local Binary Pattern texture features.
        
        Args:
            image (np.ndarray): Grayscale image
            
        Returns:
            np.ndarray: LBP histogram features
        """
        try:
            lbp = local_binary_pattern(image, self.lbp_points, self.lbp_radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=self.lbp_points + 2, 
                                     range=(0, self.lbp_points + 2))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize
            return lbp_hist
        except Exception:
            return np.zeros(self.lbp_points + 2)
    
    def extract_glcm_features(self, image):
        """
        Extract Gray-Level Co-occurrence Matrix texture features.
        
        Args:
            image (np.ndarray): Grayscale image
            
        Returns:
            np.ndarray: GLCM texture properties
        """
        try:
            # Reduce levels for computational efficiency
            img_reduced = (image // 32).astype(np.uint8)  # 8 gray levels
            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            glcm = graycomatrix(img_reduced, distances, angles, levels=8, 
                              symmetric=True, normed=True)
            
            # Extract texture features
            contrast = graycoprops(glcm, 'contrast').flatten()
            dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
            homogeneity = graycoprops(glcm, 'homogeneity').flatten()
            energy = graycoprops(glcm, 'energy').flatten()
            
            glcm_features = np.concatenate([contrast, dissimilarity, homogeneity, energy])
            return glcm_features
        except Exception:
            return np.zeros(16)  # 4 properties × 4 angles
    
    def extract_gabor_features(self, image):
        """
        Extract Gabor filter response features.
        
        Args:
            image (np.ndarray): Grayscale image
            
        Returns:
            np.ndarray: Gabor filter responses
        """
        try:
            gabor_responses = []
            frequencies = [0.1, 0.3, 0.5]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            for freq in frequencies:
                for angle in angles:
                    filtered_real, _ = gabor(image, frequency=freq, theta=angle)
                    gabor_responses.extend([
                        np.mean(filtered_real),
                        np.std(filtered_real),
                        np.mean(np.abs(filtered_real))
                    ])
            
            return np.array(gabor_responses)
        except Exception:
            return np.zeros(36)  # 3 freqs × 4 angles × 3 stats
    
    def extract_patch_features(self, image):
        """
        Extract patch-based statistical features.
        
        Args:
            image (np.ndarray): Grayscale image
            
        Returns:
            np.ndarray: Patch statistical features
        """
        try:
            patches = image.extract_patches_2d(image, self.patch_size, 
                                             max_patches=self.max_patches, 
                                             random_state=42)
            patch_stats = []
            for patch in patches:
                patch_stats.extend([
                    np.mean(patch),
                    np.std(patch),
                    np.min(patch),
                    np.max(patch)
                ])
            return np.array(patch_stats)
        except Exception:
            # Fallback: calculate simple statistics on random patches
            h, w = image.shape
            patch_stats = []
            np.random.seed(42)
            
            for _ in range(self.max_patches):
                # Random patch location
                y = np.random.randint(0, max(1, h - self.patch_size[0]))
                x = np.random.randint(0, max(1, w - self.patch_size[1]))
                
                # Extract patch
                patch = image[y:y+self.patch_size[0], x:x+self.patch_size[1]]
                
                if patch.size > 0:
                    patch_stats.extend([
                        np.mean(patch),
                        np.std(patch),
                        np.min(patch),
                        np.max(patch)
                    ])
                else:
                    patch_stats.extend([0, 0, 0, 0])
            
            return np.array(patch_stats)
    
    def extract_features_from_image(self, image, image_name=None):
        """
        Extract all feature types from a single image.
        
        Args:
            image (np.ndarray): Input image (grayscale or RGB)
            image_name (str): Optional name for the image
            
        Returns:
            dict: Dictionary containing all extracted features
        """
        # Ensure image is in correct format (uint8)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                # Assume normalized image [0,1]
                image = (image * 255).astype(np.uint8)
            else:
                # Assume image in [0,255] range but wrong dtype
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
            
        # Extract features
        sift_feat = self.extract_sift_features(gray_image)
        lbp_feat = self.extract_lbp_features(gray_image)
        glcm_feat = self.extract_glcm_features(gray_image)
        gabor_feat = self.extract_gabor_features(gray_image)
        patch_feat = self.extract_patch_features(gray_image)
        
        return {
            'sift': sift_feat,
            'lbp': lbp_feat,
            'glcm': glcm_feat,
            'gabor': gabor_feat,
            'patch': patch_feat,
            'image_name': image_name or 'unknown'
        }
    
    def extract_features_batch(self, images, image_names=None):
        """
        Extract features from a batch of images.
        
        Args:
            images (list): List of images
            image_names (list): Optional list of image names
            
        Returns:
            dict: Dictionary containing all extracted features
        """
        # Reset feature storage
        self.feature_results = {key: [] for key in self.feature_results.keys()}
        
        if image_names is None:
            image_names = [f'image_{i+1}' for i in range(len(images))]
            
        print(f"🔄 Extracting basic image features from {len(images)} images...")
        
        # Use tqdm for progress bar
        for idx, img in tqdm(enumerate(images), desc="Extracting basic features", total=len(images), unit="img"):
            # Handle different image formats
            if isinstance(img, dict):
                if 'processed' in img:
                    processed_img = img['processed']
                elif 'normalized' in img:
                    processed_img = img['normalized']
                else:
                    processed_img = img
                img_name = img.get('filename', image_names[idx])
            else:
                processed_img = img
                img_name = image_names[idx]
            
            # Extract features
            features = self.extract_features_from_image(processed_img, img_name)
            
            # Store results
            self.feature_results['sift_features'].append(features['sift'])
            self.feature_results['lbp_features'].append(features['lbp'])
            self.feature_results['glcm_features'].append(features['glcm'])
            self.feature_results['gabor_features'].append(features['gabor'])
            self.feature_results['patch_features'].append(features['patch'])
            self.feature_results['image_names'].append(features['image_name'])
        
        # Convert to numpy arrays
        for key in ['sift_features', 'lbp_features', 'glcm_features', 'gabor_features', 'patch_features']:
            if self.feature_results[key]:
                self.feature_results[key] = np.array(self.feature_results[key])
            else:
                self.feature_results[key] = np.array([]).reshape(0, 0)
        
        print(f"✅ Feature extraction complete!")
        return self.feature_results
    
    def combine_features(self):
        """
        Combine all extracted features into a single feature matrix.
        
        Returns:
            tuple: (combined_features, feature_names)
        """
        combined_features = []
        feature_names = []
        
        if len(self.feature_results['sift_features']) > 0:
            combined_features.append(self.feature_results['sift_features'])
            feature_names.extend([f'SIFT_{i}' for i in range(self.feature_results['sift_features'].shape[1])])
        
        if len(self.feature_results['lbp_features']) > 0:
            combined_features.append(self.feature_results['lbp_features'])
            feature_names.extend([f'LBP_{i}' for i in range(self.feature_results['lbp_features'].shape[1])])
        
        if len(self.feature_results['glcm_features']) > 0:
            combined_features.append(self.feature_results['glcm_features'])
            feature_names.extend([f'GLCM_{i}' for i in range(self.feature_results['glcm_features'].shape[1])])
        
        if len(self.feature_results['gabor_features']) > 0:
            combined_features.append(self.feature_results['gabor_features'])
            feature_names.extend([f'Gabor_{i}' for i in range(self.feature_results['gabor_features'].shape[1])])
        
        if len(self.feature_results['patch_features']) > 0:
            combined_features.append(self.feature_results['patch_features'])
            feature_names.extend([f'Patch_{i}' for i in range(self.feature_results['patch_features'].shape[1])])
        
        if combined_features:
            self.combined_features = np.concatenate(combined_features, axis=1)
            self.feature_names = feature_names
        else:
            self.combined_features = np.array([]).reshape(0, 0)
            self.feature_names = []
            
        return self.combined_features, self.feature_names
    
    def perform_analysis(self, labels=None, n_clusters=None):
        """
        Perform comprehensive analysis on extracted features.
        
        Args:
            labels (array-like): Optional true labels for evaluation
            n_clusters (int): Number of clusters for k-means
            
        Returns:
            dict: Analysis results including PCA, clustering, and metrics
        """
        if self.combined_features is None or len(self.combined_features) == 0:
            raise ValueError("No features available. Run feature extraction first.")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(self.combined_features)
        
        results = {
            'feature_matrix': self.combined_features,
            'scaled_features': X_scaled,
            'feature_names': self.feature_names
        }
        
        # PCA Analysis
        n_components = min(3, X_scaled.shape[0]-1, X_scaled.shape[1])
        if n_components > 0:
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            results['pca'] = {
                'transformed': X_pca,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
                'components': pca.components_
            }
        
        # t-SNE Analysis
        if X_scaled.shape[0] >= 4:
            tsne = TSNE(n_components=2, random_state=42, 
                       perplexity=min(3, X_scaled.shape[0]-1))
            X_tsne = tsne.fit_transform(X_scaled)
            results['tsne'] = X_tsne
        else:
            results['tsne'] = X_pca[:, :2] if n_components >= 2 else np.zeros((X_scaled.shape[0], 2))
        
        # Clustering Analysis
        if n_clusters is None:
            n_clusters = min(3, X_scaled.shape[0])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        results['clustering'] = {
            'labels': cluster_labels,
            'centers': kmeans.cluster_centers_,
            'n_clusters': n_clusters
        }
        
        # Calculate metrics
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            results['clustering']['silhouette_score'] = silhouette_avg
        else:
            results['clustering']['silhouette_score'] = 0
        
        # Evaluate against true labels if provided
        if labels is not None:
            if isinstance(labels, (list, np.ndarray)) and len(labels) == len(cluster_labels):
                if len(set(labels)) > 1 and len(set(cluster_labels)) > 1:
                    ari_score = adjusted_rand_score(labels, cluster_labels)
                    results['clustering']['ari_score'] = ari_score
                else:
                    results['clustering']['ari_score'] = 0
        
        return results
    
    def create_feature_visualization(self, analysis_results=None):
        """
        Create comprehensive visualization of feature extraction results.
        
        Args:
            analysis_results (dict): Results from perform_analysis()
            
        Returns:
            plotly.graph_objects.Figure: Comprehensive visualization
        """
        if analysis_results is None and self.combined_features is not None:
            analysis_results = self.perform_analysis()
        
        if analysis_results is None:
            raise ValueError("No analysis results available")
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'SIFT Feature Distribution',
                'LBP Texture Patterns', 
                'GLCM Texture Properties',
                'Gabor Filter Responses',
                'Patch Statistics',
                'Feature Dimensionality Comparison'
            )
        )
        
        # 1. SIFT Feature Distribution
        if len(self.feature_results['sift_features']) > 0:
            sift_means = np.mean(self.feature_results['sift_features'], axis=1)
            sift_stds = np.std(self.feature_results['sift_features'], axis=1)
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(sift_means) + 1)),
                    y=sift_means,
                    error_y=dict(type='data', array=sift_stds),
                    mode='markers+lines',
                    name='SIFT Features',
                    marker=dict(size=10, color='blue')
                ),
                row=1, col=1
            )
        
        # 2. LBP Texture Patterns
        if len(self.feature_results['lbp_features']) > 0:
            lbp_data = self.feature_results['lbp_features']
            for i, pattern in enumerate(lbp_data):
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(pattern))),
                        y=pattern,
                        name=f'Image {i+1}' if i == 0 else None,
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # 3. GLCM Texture Properties
        if len(self.feature_results['glcm_features']) > 0:
            glcm_data = self.feature_results['glcm_features']
            avg_glcm = np.mean(glcm_data, axis=0)
            std_glcm = np.std(glcm_data, axis=0)
            
            fig.add_trace(
                go.Bar(
                    x=list(range(len(avg_glcm))),
                    y=avg_glcm,
                    error_y=dict(type='data', array=std_glcm),
                    name='GLCM Features',
                    marker_color='green'
                ),
                row=1, col=3
            )
        
        # 4. Gabor Filter Responses
        if len(self.feature_results['gabor_features']) > 0:
            gabor_data = self.feature_results['gabor_features']
            fig.add_trace(
                go.Heatmap(
                    z=gabor_data,
                    colorscale='Viridis',
                    name='Gabor Responses'
                ),
                row=2, col=1
            )
        
        # 5. Patch Statistics
        if len(self.feature_results['patch_features']) > 0:
            patch_data = self.feature_results['patch_features']
            patch_means = patch_data[:, ::4]  # Every 4th element (means)
            
            for i, means in enumerate(patch_means):
                fig.add_trace(
                    go.Box(
                        y=means,
                        name=f'Img {i+1}',
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # 6. Feature Dimensionality Comparison
        feature_dims = {
            'SIFT': self.feature_results['sift_features'].shape[1] if len(self.feature_results['sift_features']) > 0 else 0,
            'LBP': self.feature_results['lbp_features'].shape[1] if len(self.feature_results['lbp_features']) > 0 else 0,
            'GLCM': self.feature_results['glcm_features'].shape[1] if len(self.feature_results['glcm_features']) > 0 else 0,
            'Gabor': self.feature_results['gabor_features'].shape[1] if len(self.feature_results['gabor_features']) > 0 else 0,
            'Patches': self.feature_results['patch_features'].shape[1] if len(self.feature_results['patch_features']) > 0 else 0
        }
        
        fig.add_trace(
            go.Bar(
                x=list(feature_dims.keys()),
                y=list(feature_dims.values()),
                name='Feature Dimensions',
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'],
                text=[f'{dim}D' for dim in feature_dims.values()],
                textposition='auto'
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            height=800,
            title_text="🔧 Basic Image Feature Extraction Analysis",
            title_x=0.5,
            showlegend=True
        )
        
        return fig
    
    def create_analysis_visualization(self, analysis_results, labels=None):
        """
        Create analysis visualization showing PCA, t-SNE, and clustering results.
        
        Args:
            analysis_results (dict): Results from perform_analysis()
            labels (array-like): Optional true labels for visualization
            
        Returns:
            plotly.graph_objects.Figure: Analysis visualization
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'PCA: Feature Space Projection',
                't-SNE: Nonlinear Embedding',
                'Clustering Results',
                'Feature Importance (PCA)',
                'Cluster Characteristics',
                'Analysis Summary'
            )
        )
        
        # Get data
        X_pca = analysis_results.get('pca', {}).get('transformed', np.array([]))
        X_tsne = analysis_results.get('tsne', np.array([]))
        cluster_labels = analysis_results.get('clustering', {}).get('labels', np.array([]))
        
        # Color mappings
        if labels is not None:
            unique_labels = list(set(labels))
            label_colors = {label: i for i, label in enumerate(unique_labels)}
            color_values = [label_colors[label] for label in labels]
        else:
            color_values = list(range(len(X_pca))) if len(X_pca) > 0 else []
        
        # 1. PCA Visualization
        if len(X_pca) > 0 and X_pca.shape[1] >= 2:
            fig.add_trace(
                go.Scatter(
                    x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color=color_values,
                        colorscale='viridis',
                        line=dict(width=2, color='black')
                    ),
                    text=[f'Img{i+1}' for i in range(len(X_pca))],
                    textposition="top center",
                    name='Images'
                ),
                row=1, col=1
            )
        
        # 2. t-SNE Visualization
        if len(X_tsne) > 0:
            fig.add_trace(
                go.Scatter(
                    x=X_tsne[:, 0],
                    y=X_tsne[:, 1],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color=cluster_labels if len(cluster_labels) > 0 else color_values,
                        colorscale='plasma',
                        line=dict(width=2, color='white')
                    ),
                    text=[f'Img{i+1}' for i in range(len(X_tsne))],
                    textposition="top center",
                    name='Clustered Images'
                ),
                row=1, col=2
            )
        
        # 3. Clustering Results
        if len(cluster_labels) > 0 and len(X_pca) > 0:
            fig.add_trace(
                go.Scatter(
                    x=X_pca[:, 0] if X_pca.shape[1] >= 2 else X_tsne[:, 0],
                    y=X_pca[:, 1] if X_pca.shape[1] >= 2 else X_tsne[:, 1],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=cluster_labels,
                        colorscale='rainbow',
                        line=dict(width=1, color='black')
                    ),
                    name='Clusters'
                ),
                row=1, col=3
            )
        
        # 4. Feature Importance
        if 'pca' in analysis_results and 'components' in analysis_results['pca']:
            components = analysis_results['pca']['components']
            if len(components) > 0:
                pc1_loadings = np.abs(components[0, :])
                top_indices = np.argsort(pc1_loadings)[-10:]
                
                fig.add_trace(
                    go.Bar(
                        x=pc1_loadings[top_indices],
                        y=[f'F{i}' for i in top_indices],
                        orientation='h',
                        name='PC1 Loadings',
                        marker_color='lightblue'
                    ),
                    row=2, col=1
                )
        
        # 5. Cluster Characteristics
        if len(cluster_labels) > 0:
            unique_clusters = np.unique(cluster_labels)
            cluster_sizes = [np.sum(cluster_labels == i) for i in unique_clusters]
            
            fig.add_trace(
                go.Bar(
                    x=[f'Cluster {i}' for i in unique_clusters],
                    y=cluster_sizes,
                    name='Cluster Sizes',
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(unique_clusters)],
                    text=cluster_sizes,
                    textposition='auto'
                ),
                row=2, col=2
            )
        
        # 6. Analysis Summary
        if analysis_results:
            silhouette_score = analysis_results.get('clustering', {}).get('silhouette_score', 0)
            ari_score = analysis_results.get('clustering', {}).get('ari_score', 0)
            pca_variance = analysis_results.get('pca', {}).get('explained_variance_ratio', [0])
            
            summary_text = f"""
<b>📊 Basic Image Feature Analysis Summary</b>

<b>Feature Matrix:</b>
• Images: {len(self.feature_results['image_names'])}
• Total features: {self.combined_features.shape[1] if self.combined_features is not None else 0}
• Feature types: {len([k for k, v in self.feature_results.items() if k != 'image_names' and len(v) > 0])}

<b>Dimensionality Reduction:</b>
• PCA variance: {pca_variance[0]:.3f}
• Components: {len(pca_variance)}

<b>Clustering:</b>
• Silhouette score: {silhouette_score:.3f}
• ARI score: {ari_score:.3f}
• Clusters: {len(np.unique(cluster_labels)) if len(cluster_labels) > 0 else 0}

<b>Assessment:</b>
• Quality: {'Good' if silhouette_score > 0.3 else 'Moderate' if silhouette_score > 0.1 else 'Limited'}
"""
            
            fig.add_annotation(
                text=summary_text,
                xref="x domain", yref="y domain",
                x=0.05, y=0.95, xanchor='left', yanchor='top',
                showarrow=False,
                font=dict(size=10, family="monospace"),
                bgcolor="rgba(240,245,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                row=2, col=3
            )
        
        fig.update_layout(
            height=800,
            title_text="📊 Basic Image Feature Analysis Results",
            title_x=0.5,
            showlegend=True
        )
        
        return fig
    
    def get_feature_summary(self):
        """
        Get a summary of extracted features and analysis results.
        
        Returns:
            dict: Summary of feature extraction and analysis
        """
        if self.combined_features is None:
            return {"error": "No features extracted yet"}
        
        feature_dims = {
            'SIFT': self.feature_results['sift_features'].shape[1] if len(self.feature_results['sift_features']) > 0 else 0,
            'LBP': self.feature_results['lbp_features'].shape[1] if len(self.feature_results['lbp_features']) > 0 else 0,
            'GLCM': self.feature_results['glcm_features'].shape[1] if len(self.feature_results['glcm_features']) > 0 else 0,
            'Gabor': self.feature_results['gabor_features'].shape[1] if len(self.feature_results['gabor_features']) > 0 else 0,
            'Patches': self.feature_results['patch_features'].shape[1] if len(self.feature_results['patch_features']) > 0 else 0
        }
        
        return {
            'images_processed': len(self.feature_results['image_names']),
            'feature_matrix_shape': self.combined_features.shape,
            'total_features': self.combined_features.shape[1],
            'feature_types': len([v for v in feature_dims.values() if v > 0]),
            'feature_dimensions': feature_dims,
            'feature_names': self.feature_names[:10]  # First 10 feature names
        }
