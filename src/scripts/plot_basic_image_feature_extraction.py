import numpy as np
from typing import Any, Dict, List, Optional
from src.classes.basic_image_features import BasicImageFeatureExtractor

def run_basic_feature_demo(
    processed_images: List[Any],
    sample_size: int = 10,
    random_seed: int = 42,
    extractor_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
):
    """
    Run a lightweight classical feature extraction demo on a subset of processed images.

    Args:
        processed_images: List/array of preprocessed image arrays (H,W,C) or (H,W).
        sample_size: Number of images to use (capped by available).
        random_seed: Reproducibility.
        extractor_kwargs: Optional overrides for BasicImageFeatureExtractor constructor.
        verbose: Print summary if True.

    Returns:
        dict with:
            'summary'
            'feature_dims'
            'combined_features'
            'feature_names'
            'figure' (Plotly Figure)
            'extractor' (the fitted extractor)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if not processed_images:
        raise ValueError("processed_images is empty.")

    n_total = len(processed_images)
    k = min(sample_size, n_total)
    idx = np.random.choice(n_total, size=k, replace=False)
    sample_imgs = [processed_images[i] for i in idx]
    image_names = [f"image_{i+1}" for i in range(k)]

    if extractor_kwargs is None:
        extractor_kwargs = dict(
            sift_features=128,
            lbp_radius=1,
            lbp_points=8,
            patch_size=(16, 16),
            max_patches=25
        )

    extractor = BasicImageFeatureExtractor(**extractor_kwargs)

    feature_results = extractor.extract_features_batch(
        sample_imgs,
        image_names=image_names
    )

    combined_features, feature_names = extractor.combine_features()

    # Build feature dimension breakdown
    feature_dims = {
        'SIFT': feature_results.get('sift_features', np.zeros((0, 0))).shape[1]
                if len(feature_results.get('sift_features', [])) > 0 else 0,
        'LBP': feature_results.get('lbp_features', np.zeros((0, 0))).shape[1]
                if len(feature_results.get('lbp_features', [])) > 0 else 0,
        'GLCM': feature_results.get('glcm_features', np.zeros((0, 0))).shape[1]
                if len(feature_results.get('glcm_features', [])) > 0 else 0,
        'Gabor': feature_results.get('gabor_features', np.zeros((0, 0))).shape[1]
                if len(feature_results.get('gabor_features', [])) > 0 else 0,
        'Patches': feature_results.get('patch_features', np.zeros((0, 0))).shape[1]
                if len(feature_results.get('patch_features', [])) > 0 else 0
    }
    total_dims = sum(feature_dims.values())

    # Summary
    feature_types_present = [
        k for k, v in feature_dims.items() if v > 0
    ]
    summary = {
        'images_processed': len(feature_results.get('image_names', [])),
        'feature_matrix_shape': combined_features.shape,
        'total_features': total_dims,
        'feature_types': feature_types_present
    }

    # Visualization (uses extractor's built-in method)
    figure = extractor.create_feature_visualization()

    if verbose:
        print("\n📊 Feature Extraction Summary:")
        print(f"   Images processed: {summary['images_processed']}")
        print(f"   Combined feature matrix: {summary['feature_matrix_shape']}")
        print(f"   Feature types: {len(feature_types_present)}")
        print("\n   🎯 Feature dimensions breakdown:")
        for ft, dims in feature_dims.items():
            pct = (dims / total_dims * 100) if total_dims else 0
            print(f"      {ft}: {dims} dims ({pct:.1f}%)")
        print("\n✅ Feature extraction visualization complete.")
        print(f"   📊 Total dimensions: {summary['total_features']}")
        print(f"   🖼️ Images analyzed: {summary['images_processed']}")

    return {
        'summary': summary,
        'feature_dims': feature_dims,
        'combined_features': combined_features,
        'feature_names': feature_names,
        'figure': figure,
        'extractor': extractor
    }
