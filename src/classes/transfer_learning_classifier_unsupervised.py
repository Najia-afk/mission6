# python
"""
Unsupervised Feature-Only Classifier with SWIFT-style visualizations
- Backbones: VGG16, ResNet50
- Supports using the whole CNN (include_top=True) or flattened conv features (include_top=False)
- Pipeline: PCA (elbow), t-SNE, KMeans, dashboard + side-by-side t-SNE plots, ARI vs. categories
"""

import os
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from tqdm.notebook import tqdm

from src.scripts.plot_ari_comparison import ari_comparison


class TransferLearningClassifierUnsupervised:
    """
    Run unsupervised analysis (no training) with SWIFT-like workflow:
    - Feature extraction from whole CNN head (default include_top=True)
    - PCA elbow search, PCA, t-SNE, KMeans
    - Dashboard + side-by-side t-SNE plots, ARI vs. categories
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 backbones: Optional[List[str]] = None,
                 use_gpu: bool = True,
                 use_include_top: bool = True):
        self.input_shape = input_shape
        self.backbones = backbones or ['VGG16', 'ResNet50']
        self.use_include_top = use_include_top

        # GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and use_gpu:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU setup error: {e}")

        # Data and results
        self.df: Optional[pd.DataFrame] = None
        self.images: Optional[List[np.ndarray]] = None
        self.results: Dict[str, Dict[str, Any]] = {}
        self.ari_scores: Dict[str, float] = {}
        self.processing_times: List[float] = []

    # -------------------------- Data prep --------------------------

    def prepare_data_from_dataframe(self,
                                    df: pd.DataFrame,
                                    image_column: str,
                                    category_column: str,
                                    image_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare image paths and ensure category column compatibility for compare_with_categories.
        Uses 'product_category' as in SWIFT plots (sanitized string).
        """
        df = df.copy()

        # Resolve image paths
        if image_dir:
            df['image_path'] = df[image_column].apply(lambda p: os.path.join(image_dir, os.path.basename(p)))
        else:
            df['image_path'] = df[image_column]

        # Normalize to a consistent category column for plotting
        df['product_category'] = df[category_column].astype(str)

        # Keep only rows with existing files
        df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)

        self.df = df
        print(f"Prepared {len(self.df)} samples for unsupervised analysis.")
        return {
            "samples": len(self.df),
            "category_col_used": 'product_category'
        }

    def _load_images(self) -> List[np.ndarray]:
        """
        Load images from self.df['image_path'] into numpy arrays (uint8 range [0,255]).
        """
        assert self.df is not None, "Call prepare_data_from_dataframe first."
        images = []
        for p in self.df['image_path'].tolist():
            try:
                img = load_img(p, target_size=self.input_shape[:2])
                arr = img_to_array(img).astype(np.float32)
                if arr.max() <= 1.0:
                    arr = arr * 255.0
                images.append(arr)
            except Exception:
                # Skip failing image
                pass
        self.images = images
        print(f"Loaded {len(images)} images for feature extraction.")
        return images

    # -------------------------- Backbone --------------------------

    def _build_backbone(self, backbone: str):
        """
        Build a backbone with configured include_top and return (model, preprocess, name).
        """
        if backbone == 'VGG16':
            if self.use_include_top:
                model = VGG16(weights='imagenet', include_top=True, input_shape=self.input_shape)
            else:
                # remove classification head, use GAP features (512-d vector)
                model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape, pooling='avg')
            preprocess = vgg16_preprocess
        elif backbone == 'ResNet50':
            if self.use_include_top:
                model = ResNet50(weights='imagenet', include_top=True, input_shape=self.input_shape)
            else:
                # remove classification head, use GAP features (2048-d vector)
                model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape, pooling='avg')
            preprocess = resnet50_preprocess
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        return model, preprocess, backbone

    def _extract_features(self, backbone: str) -> np.ndarray:
        """
        Extract features using configured backbone. If include_top=True, use model outputs;
        otherwise use pooled convolutional features (already 1D if pooling='avg').
        """
        assert self.images is not None, "Call _load_images first."
        model, preprocess, name = self._build_backbone(backbone)

        feats_batches = []
        self.processing_times = []
        bs = 8
        total_batches = (len(self.images) + bs - 1) // bs

        start_time = None
        for i in tqdm(range(0, len(self.images), bs), desc=f"Extracting {name} features", total=total_batches, unit="batch"):
            batch = np.stack(self.images[i:i+bs], axis=0)
            batch = preprocess(batch.copy())
            if start_time is None:
                start_time = tf.timestamp().numpy()

            preds = model.predict(batch, verbose=0)
            # if no pooling, flatten conv maps
            if not self.use_include_top and preds.ndim > 2:
                preds = preds.reshape(preds.shape[0], -1)

            feats_batches.append(preds)
            now = tf.timestamp().numpy()
            batch_time = (now - start_time) / preds.shape[0]
            self.processing_times.extend([batch_time] * preds.shape[0])
            start_time = now

        features = np.vstack(feats_batches) if feats_batches else np.zeros((0, 1))
        print(f"{name} features shape: {features.shape} (include_top={self.use_include_top})")
        return features

    # -------------------------- DR & clustering (SWIFT-style) --------------------------

    def find_optimal_pca_components(self, features: np.ndarray, max_components: int = 50, step_size: int = 5) -> Tuple[int, go.Figure]:
        print("🔍 Finding optimal number of PCA components...")
        n_samples, n_features = features.shape
        pca_upper = int(min(max_components, n_features, n_samples))
        if pca_upper < 2:
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Explained Variance vs Components", "Silhouette Score vs Components"))
            print(f"✅ Optimal number of components: {pca_upper}")
            return pca_upper, fig

        # Safe component candidates (deduped), always include the upper bound
        base_range = np.arange(step_size, max_components + 1, step_size)
        components = sorted(set([min(int(c), pca_upper) for c in base_range] + [pca_upper]))

        variance_ratios: List[float] = []
        silhouette_scores: List[float] = []
        n_components_list: List[int] = []

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        for n_comp in tqdm(components, desc="Testing PCA components", unit="components"):
            # PCA fit within valid range
            pca = PCA(n_components=n_comp)
            features_pca_test = pca.fit_transform(features_scaled)

            cum_variance = float(np.sum(pca.explained_variance_ratio_))
            variance_ratios.append(cum_variance)
            n_components_list.append(n_comp)

            # Quick KMeans + silhouette on PCA space
            if features_pca_test.shape[0] > n_comp:
                n_clusters = min(5, features_pca_test.shape[0] - 1)
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(features_pca_test)
                    silhouette_scores.append(float(silhouette_score(features_pca_test, labels)) if len(np.unique(labels)) > 1 else 0.0)
                else:
                    silhouette_scores.append(0.0)
            else:
                silhouette_scores.append(0.0)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Explained Variance vs Components", "Silhouette Score vs Components"),
            shared_xaxes=True
        )

        fig.add_trace(go.Scatter(x=n_components_list, y=variance_ratios, mode='lines+markers',
                                 name='Explained Variance', marker=dict(size=8), line=dict(color='blue', width=2)),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=n_components_list, y=silhouette_scores, mode='lines+markers',
                                 name='Silhouette Score', marker=dict(size=8), line=dict(color='red', width=2)),
                      row=1, col=2)

        fig.add_hline(y=0.95, line=dict(color='green', dash='dash'), row=1, col=1)
        fig.add_annotation(x=max(n_components_list)/2, y=0.96, text="95% Variance", showarrow=False, row=1, col=1)

        if silhouette_scores:
            optimal_components = n_components_list[int(np.argmax(silhouette_scores))]
            fig.add_vline(x=optimal_components, line=dict(color='green', dash='dash'), row=1, col=2)
            fig.add_annotation(x=optimal_components, y=max(silhouette_scores)/2, text=f"Optimal: {optimal_components}",
                               showarrow=False, row=1, col=2)
        else:
            optimal_components = components[0]

        fig.update_layout(title='PCA Component Optimization Analysis', height=500, width=1000,
                          template='plotly_white', showlegend=False)
        fig.update_xaxes(title_text="Number of Components")
        fig.update_yaxes(title_text="Explained Variance Ratio (Cumulative)", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)

        print(f"✅ Optimal number of components: {optimal_components}")
        return int(optimal_components), fig
    
    def apply_dimensionality_reduction(self, features: np.ndarray, n_components: Union[int, float] = 50, method: str = 'pca') -> Tuple[np.ndarray, Any, StandardScaler]:
        if features.shape[0] == 0:
            print("No features to reduce")
            return np.array([]), None, None

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        if method.lower() == 'pca':
            if isinstance(n_components, int):
                n_components = min(n_components, min(features.shape))
                print(f"Applying PCA to reduce dimensions from {features.shape[1]} to {n_components}...")
                reducer = PCA(n_components=n_components)
            else:
                print(f"Applying PCA to preserve {n_components:.1%} variance...")
                reducer = PCA(n_components=n_components)

            reduced_features = reducer.fit_transform(features_scaled)
            cumulative_variance = float(np.sum(reducer.explained_variance_ratio_))
            print(f"PCA completed: {cumulative_variance:.2%} of variance preserved")

        elif method.lower() == 'tsne':
            print(f"Applying t-SNE to reduce dimensions to {n_components}...")
            reducer = TSNE(n_components=n_components, random_state=42)
            with tqdm(total=100, desc="t-SNE progress", unit="%") as pbar:
                reduced_features = reducer.fit_transform(features_scaled)
                pbar.update(100)
            print("t-SNE completed")
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")

        return reduced_features, reducer, scaler

    def perform_clustering(self, features: np.ndarray, n_clusters: Optional[int] = None, cluster_range: Tuple[int, int] = (2, 8)) -> Dict[str, Any]:
        print("🎯 Performing clustering analysis...")

        if n_clusters is None:
            print(f"Finding optimal number of clusters in range {cluster_range}...")
            silhouette_scores: List[float] = []
            inertias: List[float] = []

            for k in tqdm(range(cluster_range[0], cluster_range[1] + 1), desc="Testing cluster counts", unit="k"):
                if k >= features.shape[0]:
                    print(f"Skipping k={k}: more clusters than samples")
                    continue
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                inertias.append(kmeans.inertia_)
                if features.shape[0] > k and k > 1:
                    silhouette_scores.append(float(silhouette_score(features, labels)))
                else:
                    silhouette_scores.append(0.0)

            if silhouette_scores:
                n_clusters = cluster_range[0] + int(np.argmax(silhouette_scores))
                print(f"Optimal number of clusters: {n_clusters} (silhouette score: {max(silhouette_scores):.3f})")
            else:
                n_clusters = cluster_range[0]
                print(f"Using default number of clusters: {n_clusters}")

        print(f"Performing KMeans clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        silhouette_avg = float(silhouette_score(features, labels)) if n_clusters > 1 and features.shape[0] > n_clusters else 0.0
        print(f"Clustering completed: {n_clusters} clusters, silhouette score: {silhouette_avg:.3f}")

        return {
            'n_clusters': n_clusters,
            'labels': labels,
            'silhouette_score': silhouette_avg,
            'inertia': kmeans.inertia_,
            'cluster_centers': kmeans.cluster_centers_
        }

    # -------------------------- Visualization (SWIFT-style) --------------------------

    def create_analysis_dashboard(self,
                                  backbone_name: str,
                                  original_features: np.ndarray,
                                  reduced_features: np.ndarray,
                                  clustering_results: Dict[str, Any],
                                  processing_times: List[float],
                                  pca_info: Optional[Any] = None) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['', 'Clustering Results', f'{backbone_name} Processing Summary', 'Processing Time Distribution'],
            specs=[[{"type": "indicator"}, {"type": "scatter"}], [{"type": "table"}, {"type": "histogram"}]]
        )

        avg_time = float(np.mean(processing_times)) if processing_times else 0.0
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=(1.0 / avg_time) if avg_time > 0 else 0.0,
                title={'text': "Processing Speed (img/sec)"},
                number={'suffix': " img/sec"},
                gauge={
                    'axis': {'range': [0, max(5, (2.0 / avg_time) if avg_time > 0 else 5)]},
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

        # Scatter of reduced features with cluster colors
        if reduced_features.shape[1] >= 2 and 'labels' in clustering_results:
            x = reduced_features[:, 0]
            y = reduced_features[:, 1]
            colors = clustering_results['labels']
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(size=8, color=colors, colorscale='Viridis', showscale=True, colorbar=dict(title='Cluster')),
                    name='Clusters'),
                row=1, col=2
            )
            if 'cluster_centers' in clustering_results and reduced_features.shape[1] >= 2:
                centers = clustering_results['cluster_centers']
                if centers.shape[1] >= 2:
                    fig.add_trace(
                        go.Scatter(
                            x=centers[:, 0], y=centers[:, 1],
                            mode='markers', marker=dict(size=14, color='red', symbol='x'), name='Cluster Centers'),
                        row=1, col=2
                    )

        variance_preserved = float(pca_info.explained_variance_ratio_.sum()) if pca_info is not None else 0.0
        compression_ratio = (original_features.shape[1] / reduced_features.shape[1]) if reduced_features.shape[1] > 0 else 0.0
        summary_data = [
            ['Original Feature Dimensions', f"{original_features.shape[1]:,}"],
            ['PCA Reduced Dimensions', f"{reduced_features.shape[1]:,}"],
            ['Samples Processed', f"{original_features.shape[0]:,}"],
            ['Compression Ratio', f"{compression_ratio:.1f}x"],
            ['Variance Preserved', f"{variance_preserved:.1%}"],
            ['Optimal Clusters', f"{clustering_results['n_clusters']}"],
            ['Silhouette Score', f"{clustering_results['silhouette_score']:.3f}"],
            ['Avg Processing Time', f"{avg_time:.3f}s/image"],
            ['Processing Speed', f"{(1/avg_time):.1f} img/sec" if avg_time > 0 else "∞"],
            ['Model Used', backbone_name]
        ]
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'], fill_color='lightblue', align='center', font=dict(size=12, color='black')),
                cells=dict(values=[[row[0] for row in summary_data], [row[1] for row in summary_data]],
                           fill_color='white', align='center', font=dict(size=11))
            ),
            row=2, col=1
        )

        fig.add_trace(go.Histogram(x=processing_times, nbinsx=20, marker_color='purple', name='Processing Times'),
                      row=2, col=2)

        fig.update_layout(title=f'{backbone_name} Feature Extraction Dashboard', template='plotly_white',
                          showlegend=False, width=1000, height=800)
        return fig

    def compare_with_categories(self, df: pd.DataFrame, tsne_features: np.ndarray, clustering_results: Dict[str, Any], backbone_name: str) -> Dict[str, Any]:
        print(f"🔍 {backbone_name} Analysis: Comparing clustering with real product categories...")

        labels = clustering_results['labels']
        cats = []
        for i in range(len(labels)):
            if i < len(df):
                cats.append(df.iloc[i]['product_category'])
            else:
                cats.append('Unknown')
        cats = np.array(cats)

        ari = float(adjusted_rand_score(cats, labels))
        print(f"📊 {backbone_name} processed {len(labels)} images")
        print(f"📂 Unique categories: {len(np.unique(cats))}")
        print(f"🎯 Adjusted Rand Index(ARI): {ari:.4f}")
        print(f"🔗 Cluster quality (Silhouette): {clustering_results['silhouette_score']:.3f}")

        # DataFrame for plots
        tsne_df = pd.DataFrame({
            't-SNE1': tsne_features[:, 0],
            't-SNE2': tsne_features[:, 1],
            'Category': cats,
            'Cluster': labels
        })

        # t-SNE by real categories
        tsne_fig = px.scatter(
            tsne_df, x='t-SNE1', y='t-SNE2', color='Category',
            title=f'🚀 {backbone_name} Features: t-SNE Visualization by Product Categories',
            hover_data={'Cluster': True},
            labels={'t-SNE1': 't-SNE Component 1', 't-SNE2': 't-SNE Component 2'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        tsne_fig.update_layout(width=1000, height=700, template='plotly_white', title_x=0.5,
                               legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
        tsne_fig.add_annotation(
            text=f"📊 {len(cats)} images • {len(np.unique(cats))} categories • {len(np.unique(labels))} clusters<br>"
                 f"🎯 ARI Score: {ari:.4f} • Silhouette Score: {clustering_results['silhouette_score']:.3f}",
            xref="paper", yref="paper", x=0.5, y=-0.1, xanchor='center', yanchor='top',
            showarrow=False, font=dict(size=12, color="gray"), bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1
        )

        # Side-by-side comparison
        print("\n📊 Creating side-by-side comparison: Real Categories vs Clusters...")
        comparison_fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['🏷️ Colored by Real Product Categories', f'🎯 Colored by {backbone_name} Clusters'],
            horizontal_spacing=0.1
        )

        # Left: categories
        for i, category in enumerate(np.unique(cats)):
            mask = cats == category
            comparison_fig.add_trace(
                go.Scatter(
                    x=tsne_features[mask, 0], y=tsne_features[mask, 1],
                    mode='markers', name=str(category),
                    marker=dict(size=8, color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)],
                                line=dict(width=1, color='white')),
                    showlegend=True, legendgroup='categories'),
                row=1, col=1
            )

        # Right: clusters
        unique_clusters = np.unique(labels)
        cluster_colors = px.colors.qualitative.Dark2
        for i, cluster in enumerate(unique_clusters):
            mask = labels == cluster
            comparison_fig.add_trace(
                go.Scatter(
                    x=tsne_features[mask, 0], y=tsne_features[mask, 1],
                    mode='markers', name=f'Cluster {cluster}',
                    marker=dict(size=8, color=cluster_colors[i % len(cluster_colors)],
                                line=dict(width=1, color='white')),
                    showlegend=True, legendgroup='clusters'),
                row=1, col=2
            )

        comparison_fig.update_layout(
            title=f'🔍 {backbone_name} Features: t-SNE Analysis Comparison',
            title_x=0.5, width=1400, height=600, template='plotly_white',
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        comparison_fig.update_xaxes(title_text="t-SNE Component 1")
        comparison_fig.update_yaxes(title_text="t-SNE Component 2")
        comparison_fig.add_annotation(
            text=f"📈 {backbone_name} Performance: ARI = {ari:.4f} • Silhouette = {clustering_results['silhouette_score']:.3f}<br>"
                 f"💡 {'Good alignment' if ari > 0.5 else 'Moderate alignment' if ari > 0.2 else 'Poor alignment'} between clusters and true categories",
            xref="paper", yref="paper", x=0.5, y=-0.12, xanchor='center', yanchor='top',
            showarrow=False, font=dict(size=12, color="gray"), bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray", borderwidth=1
        )

        print(f"🔍 {backbone_name} Side-by-Side Comparison:")
        comparison_fig.show()

        # Return results structure
        unique_cats, counts = np.unique(cats, return_counts=True)
        return {
            'ari_score': ari,
            'tsne_fig': tsne_fig,
            'comparison_fig': comparison_fig,
            'clustering_results': clustering_results,
            'categories': cats,
            'cluster_labels': labels,
            'tsne_data': tsne_df,
            'category_distribution': dict(zip(unique_cats, counts)),
            'n_categories': len(np.unique(cats)),
            'n_clusters': len(unique_clusters),
            'silhouette_score': clustering_results['silhouette_score']
        }

    # -------------------------- Pipeline --------------------------

    def run_pipeline(self,
                     backbone: str = 'VGG16',
                     pca_components: int = 150,
                     cluster_range: Tuple[int, int] = (2, 7),
                     auto_optimize_pca: bool = False,
                     pca_max_components: int = 500,
                     pca_step_size: int = 50) -> Dict[str, Any]:
        """
        Full unsupervised pipeline with SWIFT-like outputs.
        """
        if self.images is None:
            self._load_images()

        features = self._extract_features(backbone)
        if features.size == 0:
            raise RuntimeError("No features extracted. Check images/data paths.")

        elbow_fig = None
        if auto_optimize_pca:
            optimal_components, elbow_fig = self.find_optimal_pca_components(
                features, max_components=pca_max_components, step_size=pca_step_size
            )
            pca_components = int(optimal_components)

        print(f"[{backbone}] PCA reduction...")
        feats_pca, pca_obj, _ = self.apply_dimensionality_reduction(features, n_components=pca_components, method='pca')

        print(f"[{backbone}] t-SNE projection...")
        feats_tsne, tsne_obj, _ = self.apply_dimensionality_reduction(feats_pca, n_components=2, method='tsne')

        print(f"[{backbone}] Clustering...")
        cluster_results = self.perform_clustering(feats_pca, n_clusters=None, cluster_range=cluster_range)

        print(f"[{backbone}] Dashboard...")
        dashboard = self.create_analysis_dashboard(
            backbone_name=backbone,
            original_features=features,
            reduced_features=feats_pca,
            clustering_results=cluster_results,
            processing_times=self.processing_times,
            pca_info=pca_obj
        )
        dashboard.show()

        print(f"[{backbone}] ARI and visualization...")
        analysis = self.compare_with_categories(
            df=self.df,
            tsne_features=feats_tsne,
            clustering_results=cluster_results,
            backbone_name=backbone
        )

        # Store results
        self.results[backbone] = {
            "features": features,
            "pca": feats_pca,
            "tsne": feats_tsne,
            "clustering": cluster_results,
            "analysis": analysis,
            "dashboard": dashboard,
            "elbow": elbow_fig
        }
        self.ari_scores[backbone] = analysis['ari_score']

        print(f"[{backbone}] ARI: {analysis['ari_score']:.4f}, silhouette: {analysis['silhouette_score']:.3f}")
        return self.results[backbone]

    def compare_unsupervised(self) -> Any:
        """
        Plot ARI comparison across backbones using existing bar plot.
        """
        if not self.ari_scores:
            print("No ARI scores available. Run run_pipeline() for at least one backbone.")
            return None
        scores = {f"{k} Deep": v for k, v in self.ari_scores.items()}
        fig = ari_comparison(scores)
        return fig