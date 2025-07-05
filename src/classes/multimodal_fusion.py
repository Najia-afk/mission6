import numpy as np
from sklearn.preprocessing import StandardScaler
from .reduce_dimensions import DimensionalityReducer

class MultimodalFusion:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.reducer = DimensionalityReducer(random_state=self.random_state)

    def prepare_and_fuse(self, text_features, image_features):
        # Align features to the same number of samples
        min_samples = min(text_features.shape[0], image_features.shape[0])
        text_features_aligned = text_features[:min_samples]
        image_features_aligned = image_features[:min_samples]
        
        # Scale features before fusion
        text_scaled = self.scaler.fit_transform(text_features_aligned)
        image_scaled = self.scaler.fit_transform(image_features_aligned)
        
        # Fuse by concatenation
        fused_features = np.hstack([text_scaled, image_scaled])
        return fused_features, min_samples

    def analyze_fused_features(self, fused_features, labels):
        # Reduce dimensionality
        pca_features = self.reducer.fit_transform_pca(fused_features)
        tsne_features = self.reducer.fit_transform_tsne(fused_features)
        
        # Visualize
        pca_fig = self.reducer.plot_pca(pca_features, labels=labels)
        tsne_fig = self.reducer.plot_tsne(tsne_features, labels=labels)
        
        return pca_fig, tsne_fig
