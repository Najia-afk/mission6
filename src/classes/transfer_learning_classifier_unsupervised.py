"""
Unsupervised Feature-Only Classifier
- Extracts deep features with backbone CNNs (VGG16, ResNet50)
- Reduces dimensions (PCA), projects (t-SNE), clusters (KMeans)
- Compares clusters to real categories (ARI) using existing plotting logic
"""

import os
import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.metrics import adjusted_rand_score

# Reuse existing utilities
from .vgg16_extractor import VGG16FeatureExtractor
from src.scripts.plot_ari_comparison import ari_comparison


class TransferLearningClassifierUnsupervised:
    """
    Run unsupervised analysis (no training):
    - Backbones supported: VGG16, ResNet50
    - Returns ARI and t-SNE plots via VGG16FeatureExtractor.compare_with_categories
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 backbones: Optional[List[str]] = None,
                 use_gpu: bool = True):
        self.input_shape = input_shape
        self.backbones = backbones or ['VGG16', 'ResNet50']

        # GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and use_gpu:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU setup error: {e}")

        # Will reuse VGG16FeatureExtractor for PCA/t-SNE/clustering/plots
        self._aux = VGG16FeatureExtractor(input_shape=input_shape, layer_name='block5_pool')

        # Data and results
        self.df: Optional[pd.DataFrame] = None
        self.images: Optional[List[np.ndarray]] = None
        self.results: Dict[str, Dict[str, Any]] = {}
        self.ari_scores: Dict[str, float] = {}

    def prepare_data_from_dataframe(self,
                                    df: pd.DataFrame,
                                    image_column: str,
                                    category_column: str,
                                    image_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare image paths and ensure category column compatibility for compare_with_categories.
        """
        df = df.copy()

        # Resolve image paths
        if image_dir:
            df['image_path'] = df[image_column].apply(lambda p: os.path.join(image_dir, os.path.basename(p)))
        else:
            df['image_path'] = df[image_column]

        # Ensure expected category column exists for compare_with_categories
        # compare_with_categories expects 'product_category_tree'; fallback to provided category
        if 'product_category_tree' not in df.columns:
            df['product_category_tree'] = df[category_column].astype(str)

        # Keep only rows with existing files
        df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)

        self.df = df
        print(f"Prepared {len(self.df)} samples for unsupervised analysis.")
        return {
            "samples": len(self.df),
            "category_col_used": 'product_category_tree'
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
            except Exception as e:
                # skip failing image
                pass
        self.images = images
        print(f"Loaded {len(images)} images for feature extraction.")
        return images

    def _extract_features(self, backbone: str) -> np.ndarray:
        """
        Extract deep features using the selected backbone (include_top=False), flatten output.
        """
        assert self.images is not None, "Call _load_images first."

        if backbone == 'VGG16':
            model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
            preprocess = vgg16_preprocess
        elif backbone == 'ResNet50':
            model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
            preprocess = resnet50_preprocess
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        feats_batches = []
        bs = 8
        for i in range(0, len(self.images), bs):
            batch = np.stack(self.images[i:i+bs], axis=0)
            batch = preprocess(batch.copy())
            feats = model.predict(batch, verbose=0)
            feats = feats.reshape(feats.shape[0], -1)
            feats_batches.append(feats)

        features = np.vstack(feats_batches) if feats_batches else np.zeros((0, 1))
        print(f"{backbone} features shape: {features.shape}")
        return features

    def run_pipeline(self,
                     backbone: str = 'VGG16',
                     pca_components: int = 150,
                     cluster_range: Tuple[int, int] = (2, 7)) -> Dict[str, Any]:
        """
        Full unsupervised pipeline for a backbone:
        - feature extraction
        - PCA -> t-SNE
        - KMeans clustering
        - ARI vs category; t-SNE figs using compare_with_categories
        """
        if self.images is None:
            self._load_images()

        features = self._extract_features(backbone)
        if features.size == 0:
            raise RuntimeError("No features extracted. Check images/data paths.")

        # Dimensionality reduction and clustering via existing extractor utilities
        print(f"[{backbone}] PCA reduction...")
        feats_pca, pca_obj, _ = self._aux.apply_dimensionality_reduction(features, n_components=pca_components, method='pca')

        print(f"[{backbone}] t-SNE projection...")
        feats_tsne, tsne_obj, _ = self._aux.apply_dimensionality_reduction(feats_pca, n_components=2, method='tsne')

        print(f"[{backbone}] Clustering...")
        cluster_results = self._aux.perform_clustering(feats_pca, n_clusters=None, cluster_range=cluster_range)

        # Compare clusters vs categories (reuses existing plotting/ARI)
        print(f"[{backbone}] ARI and visualization...")
        analysis = self._aux.compare_with_categories(
            df=self.df,
            tsne_features=feats_tsne,
            clustering_results=cluster_results
        )

        # Store results
        self.results[backbone] = {
            "features": features,
            "pca": feats_pca,
            "tsne": feats_tsne,
            "clustering": cluster_results,
            "analysis": analysis
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
