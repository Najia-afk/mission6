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
    
    def __init__(self, model_name='ViT-B/32', device=None, input_shape=(224, 224, 3)):
        """
        Initialize the SWIFT feature extractor using CLIP
        
        Args:
            model_name (str): CLIP model variant to use
            device: Device to run on (auto-detected if None)
            input_shape: Input shape for the model (for compatibility with VGG16FeatureExtractor)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processing_times = []
        self.feature_shape = None
        self.extracted_features = None
        self.input_shape = input_shape
        self.layer_name = model_name  # For compatibility with VGG16FeatureExtractor
        
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
        
    def find_optimal_pca_components(
        self, 
        features: np.ndarray, 
        max_components: int = 50, 
        step_size: int = 5
    ) -> Tuple[int, go.Figure]:
        """
        Find optimal number of PCA components using elbow method
        with both variance explained and silhouette score analysis
        
        Args:
            features: Feature matrix
            max_components: Maximum number of components to test
            step_size: Step size for testing components
            
        Returns:
            Tuple containing:
                - Optimal number of components
                - Plotly figure with elbow plots
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from tqdm.notebook import tqdm  # Import tqdm for progress tracking
        
        print("🔍 Finding optimal number of PCA components...")
        
        # Create a range of potential component numbers to test
        max_components = min(max_components, features.shape[1])
        component_range = np.arange(step_size, max_components + 1, step_size)
        
        # Arrays to store results
        variance_ratios = []
        silhouette_scores = []
        n_components_list = []
        
        # Scale features once (reused for all component tests)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        print(f"Testing {len(component_range)} different component counts...")
        # Add tqdm progress bar for component testing
        for n_comp in tqdm(component_range, desc="Testing PCA components", unit="components"):
            # Apply PCA with current number of components
            pca = PCA(n_components=n_comp)
            features_pca_test = pca.fit_transform(features_scaled)
            
            # Store cumulative explained variance
            cum_variance = np.sum(pca.explained_variance_ratio_)
            variance_ratios.append(cum_variance)
            n_components_list.append(n_comp)
            
            # Perform clustering to calculate silhouette score
            if features_pca_test.shape[0] > n_comp:  # Ensure we have enough samples
                n_clusters = min(5, features_pca_test.shape[0]-1)
                if n_clusters > 1:  # Need at least 2 clusters
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(features_pca_test)
                    
                    # Calculate silhouette score if we have multiple clusters
                    if len(np.unique(cluster_labels)) > 1:
                        sil_score = silhouette_score(features_pca_test, cluster_labels)
                        silhouette_scores.append(sil_score)
                    else:
                        silhouette_scores.append(0)
                else:
                    silhouette_scores.append(0)
            else:
                silhouette_scores.append(0)
            
        
        # Create visualization
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=(
                "Explained Variance vs Components", 
                "Silhouette Score vs Components"
            ),
            shared_xaxes=True
        )
        
        # Add variance plot
        fig.add_trace(
            go.Scatter(
                x=n_components_list, 
                y=variance_ratios, 
                mode='lines+markers',
                name='Explained Variance', 
                marker=dict(size=8),
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add silhouette plot
        fig.add_trace(
            go.Scatter(
                x=n_components_list, 
                y=silhouette_scores, 
                mode='lines+markers',
                name='Silhouette Score', 
                marker=dict(size=8),
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        # Add 0.95 variance threshold line
        fig.add_hline(
            y=0.95, 
            line=dict(color='green', dash='dash'), 
            row=1, col=1
        )
        fig.add_annotation(
            x=max_components/2, 
            y=0.96, 
            text="95% Variance", 
            showarrow=False, 
            row=1, col=1
        )
        
        # Find optimal components based on silhouette score
        if silhouette_scores:
            optimal_components = n_components_list[np.argmax(silhouette_scores)]
            
            # Add vertical line at optimal components
            fig.add_vline(
                x=optimal_components, 
                line=dict(color='green', dash='dash'), 
                row=1, col=2
            )
            fig.add_annotation(
                x=optimal_components, 
                y=max(silhouette_scores)/2, 
                text=f"Optimal: {optimal_components}", 
                showarrow=False, 
                row=1, col=2
            )
        else:
            optimal_components = component_range[0]
        
        # Update layout
        fig.update_layout(
            title='PCA Component Optimization Analysis',
            height=500, 
            width=1000,
            template='plotly_white',
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Number of Components")
        fig.update_yaxes(title_text="Explained Variance Ratio (Cumulative)", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        
        print(f"✅ Optimal number of components: {optimal_components}")
        return optimal_components, fig

    def apply_dimensionality_reduction(
        self, 
        features: np.ndarray, 
        n_components: Union[int, float] = 50, 
        method: str = 'pca'
    ) -> Tuple[np.ndarray, Any, StandardScaler]:
        """
        Apply dimensionality reduction to features
        
        Args:
            features: Feature matrix
            n_components: Number of components to keep or variance threshold (0-1)
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
            if isinstance(n_components, int):
                n_components = min(n_components, min(features.shape))
                print(f"Applying PCA to reduce dimensions from {features.shape[1]} to {n_components}...")
                reducer = PCA(n_components=n_components)
            else:
                # n_components is a variance threshold (0-1)
                print(f"Applying PCA to preserve {n_components:.1%} variance...")
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
            with tqdm(total=100, desc="t-SNE progress", unit="%") as pbar:
                reduced_features = reducer.fit_transform(features_scaled)
                pbar.update(100)  # Mark as complete
            
            print("t-SNE completed")
            
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        return reduced_features, reducer, scaler
    
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
                '',
                'Clustering Results',
                'SWIFT Processing Summary',
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
        
        # 3. SWIFT Processing Summary Table
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
            ['Model Used', f"{self.model_name}"]
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
            title='SWIFT (CLIP) Feature Extraction Dashboard',
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
            'model_used': self.model_name,
            'input_shape': self.input_shape
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
