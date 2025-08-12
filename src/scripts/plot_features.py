import os
import inspect
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from src.classes.basic_image_featuresv2 import BasicImageFeatureExtractor


def _basename(p: str) -> str:
    """Safe basename for mapping names in DataFrame to processed image names."""
    return os.path.basename(p) if isinstance(p, str) else str(p)


def _build_image_lookup(
    df: pd.DataFrame,
    processed_images: Any,
    id_column: str,
    image_column: str
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Builds two lookups: by uniq_id and by image filename, to find arrays quickly."""
    by_id: Dict[str, np.ndarray] = {}
    by_image: Dict[str, np.ndarray] = {}

    # Common dict structure from ImageProcessor
    if isinstance(processed_images, dict):
        arrays = None
        names = None
        for k in ('processed_images', 'images', 'image_arrays'):
            if k in processed_images and isinstance(processed_images[k], (list, tuple, np.ndarray)):
                arrays = processed_images[k]
                break
        for k in ('image_names', 'image_paths', 'filenames', 'images'):
            if k in processed_images and isinstance(processed_images[k], (list, tuple, np.ndarray)):
                names = processed_images[k]
                break
        if arrays is not None and names is not None and len(arrays) == len(names):
            for arr, name in zip(arrays, names):
                if isinstance(arr, np.ndarray):
                    by_image[_basename(name)] = arr
        # Fallback if dict is {name: array}
        if not by_image and hasattr(processed_images, 'items'):
            try:
                if all(isinstance(v, np.ndarray) for v in processed_images.values()):
                    for k, v in processed_images.items():
                        by_image[_basename(k)] = v
            except Exception:
                pass

    # If caller passed arrays aligned with df
    elif isinstance(processed_images, (list, tuple, np.ndarray)):
        if len(processed_images) == len(df) and all(isinstance(x, np.ndarray) for x in processed_images):
            for uid, img, img_name in zip(df[id_column].astype(str), processed_images, df[image_column]):
                by_id[str(uid)] = img
                by_image[_basename(img_name)] = img

    # Generic mapping
    elif hasattr(processed_images, 'items'):
        try:
            for k, v in processed_images.items():
                if isinstance(v, np.ndarray):
                    by_image[_basename(k)] = v
        except Exception:
            pass

    return by_id, by_image


def normalize_images_by_category(
    df: pd.DataFrame,
    processed_images: Any,
    feature_extractor: BasicImageFeatureExtractor,
    num_images_per_category: Optional[int] = None,
    random_seed: Optional[int] = None,
    id_column: str = 'uniq_id',
    image_column: str = 'image',
) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
    """
    Normalize images by category and compute average feature vectors per category.
    Returns dict[category] -> {sift, lbp, glcm, gabor, patch: np.ndarray or None}
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    by_id, by_image = _build_image_lookup(df, processed_images, id_column, image_column)

    normalized_images: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    if 'product_category' not in df.columns:
        return normalized_images

    for category in df['product_category'].dropna().unique():
        cat_rows = df[df['product_category'] == category]
        cat_imgs: List[np.ndarray] = []
        cat_names: List[str] = []
        for _, row in cat_rows.iterrows():
            uid = str(row.get(id_column, ''))
            img_name = row.get(image_column, '')
            img_key = _basename(img_name)
            img = by_id.get(uid) if uid else None
            if img is None and isinstance(img_key, str):
                img = by_image.get(img_key)
            if isinstance(img, np.ndarray):
                cat_imgs.append(img)
                cat_names.append(img_key or uid)

        if not cat_imgs:
            continue

        # Optional subsample for balance/speed
        if num_images_per_category is not None and len(cat_imgs) > num_images_per_category:
            idx = np.random.choice(len(cat_imgs), num_images_per_category, replace=False)
            cat_imgs = [cat_imgs[i] for i in idx]
            cat_names = [cat_names[i] for i in idx]

        # Extract per-category batch and average each feature type
        feature_extractor.extract_features_batch(cat_imgs, image_names=cat_names)
        fr = getattr(feature_extractor, 'feature_results', {})
        normalized_images[category] = {
            'sift': np.mean(fr.get('sift_features', []), axis=0) if len(fr.get('sift_features', [])) > 0 else None,
            'lbp': np.mean(fr.get('lbp_features', []), axis=0) if len(fr.get('lbp_features', [])) > 0 else None,
            'glcm': np.mean(fr.get('glcm_features', []), axis=0) if len(fr.get('glcm_features', [])) > 0 else None,
            'gabor': np.mean(fr.get('gabor_features', []), axis=0) if len(fr.get('gabor_features', [])) > 0 else None,
            'patch': np.mean(fr.get('patch_features', []), axis=0) if len(fr.get('patch_features', [])) > 0 else None,
        }

    return normalized_images


def _build_figures_from_results(
    df: pd.DataFrame,
    combined_features: Optional[np.ndarray],
    feature_names: List[str],
    feature_results: Dict[str, Any],
    df_magnitudes: pd.DataFrame,
    image_column: str = 'image',
    category_column: str = 'product_category',
) -> Dict[str, go.Figure]:
    """Create PCA scatter, heatmap, radar, and stacked bar from extracted features."""
    figs: Dict[str, go.Figure] = {}

    # Map image name -> category
    name_to_cat = {
        _basename(row[image_column]): row[category_column]
        for _, row in df[[image_column, category_column]].dropna().iterrows()
    }

    # Feature scatter (PCA) colored by category if possible
    image_names = []
    if 'image_names' in feature_results and isinstance(feature_results['image_names'], (list, tuple)):
        image_names = [_basename(x) for x in feature_results['image_names']]

    if isinstance(combined_features, np.ndarray) and combined_features.ndim == 2 and combined_features.shape[0] > 1:
        n = combined_features.shape[0]
        if len(image_names) != n:
            image_names = [f'img_{i}' for i in range(n)]
        cats = [name_to_cat.get(_basename(x), 'Unknown') for x in image_names]
        # Reduce to 2D for plotting
        X = PCA(n_components=2, random_state=0).fit_transform(combined_features) if combined_features.shape[1] > 1 else np.c_[combined_features, np.zeros((n, 1))]
        figs['feature_viz'] = go.Figure([
            go.Scattergl(
                x=X[:, 0], y=X[:, 1], mode='markers',
                marker=dict(size=6, opacity=0.85),
                text=image_names,
                customdata=np.array(cats, dtype=object),
                hovertemplate='Name: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Category: %{customdata}',
            )
        ]).update_layout(title='PCA of Combined Image Features', xaxis_title='PC1', yaxis_title='PC2')
    else:
        figs['feature_viz'] = go.Figure([go.Scatter(y=[0], mode='markers')]).update_layout(title='No combined features')

    # Heatmap from per-category magnitudes
    if not df_magnitudes.empty:
        feat_types = [c for c in ['sift', 'lbp', 'glcm', 'gabor', 'patch'] if c in df_magnitudes.columns]
        z = df_magnitudes[feat_types].values
        y = (df_magnitudes['category'].tolist()
             if 'category' in df_magnitudes.columns
             else df_magnitudes.index.tolist())
        figs['heatmap'] = go.Figure([go.Heatmap(z=z, x=feat_types, y=y, colorscale='Viridis')]).update_layout(
            title='Feature Magnitude by Category', xaxis_title='Feature Type', yaxis_title='Category'
        )
    else:
        figs['heatmap'] = go.Figure([go.Heatmap(z=[[0]])]).update_layout(title='No category magnitudes')

    # Radar chart (normalized per-feature)
    if not df_magnitudes.empty:
        feat_types = [c for c in ['sift', 'lbp', 'glcm', 'gabor', 'patch'] if c in df_magnitudes.columns]
        vals = df_magnitudes[feat_types].replace(0, np.nan)
        vals = vals.divide(vals.max(axis=0), axis=1).fillna(0.0)
        radar = go.Figure()
        for idx, row in vals.iterrows():
            r = row.values.tolist()
            name = (df_magnitudes.loc[idx, 'category'] if 'category' in df_magnitudes.columns else str(idx))
            radar.add_trace(go.Scatterpolar(r=r, theta=feat_types, fill='toself', name=name))
        radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Feature Profile 'Shape' by Category"
        )
        figs['radar'] = radar
    else:
        figs['radar'] = go.Figure().update_layout(title='No radar (no magnitudes)')

    # Stacked bar: category contribution per feature type
    if not df_magnitudes.empty:
        feat_types = [c for c in ['sift', 'lbp', 'glcm', 'gabor', 'patch'] if c in df_magnitudes.columns]
        bar = go.Figure()
        labels = (df_magnitudes['category'].tolist()
                  if 'category' in df_magnitudes.columns
                  else df_magnitudes.index.tolist())
        data_mat = df_magnitudes[feat_types].values
        for i, label in enumerate(labels):
            bar.add_trace(go.Bar(x=feat_types, y=data_mat[i, :], name=str(label)))
        bar.update_layout(
            barmode='stack',
            title='Category Contribution to Each Feature Type',
            xaxis_title='Feature Type',
            yaxis_title='Total Feature Activation (L2 Norm)'
        )
        figs['stacked_bar'] = bar
    else:
        figs['stacked_bar'] = go.Figure().update_layout(title='No stacked bar (no magnitudes)')

    return figs


def build_category_feature_matrix(
    normalized_images: Dict[str, Dict[str, Optional[np.ndarray]]]
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Flatten per-category averaged features into a matrix [categories x concatenated_features]."""
    feature_matrix: List[List[float]] = []
    feature_names: List[str] = []
    category_names: List[str] = []

    for category, features in normalized_images.items():
        vec: List[float] = []
        local_names: List[str] = []
        for ftype, values in features.items():
            if values is not None:
                vec.extend(values.tolist() if isinstance(values, np.ndarray) else list(values))
                if not feature_names:
                    local_names.extend([f"{ftype}_{i}" for i in range(len(values))])
        if vec:
            feature_matrix.append(vec)
            category_names.append(category)
            if not feature_names and local_names:
                feature_names = local_names

    return (np.array(feature_matrix) if feature_matrix else np.empty((0, 0))), feature_names, category_names


def compute_category_feature_magnitudes(
    normalized_images: Dict[str, Dict[str, Optional[np.ndarray]]],
    include_category_col: bool = True
) -> pd.DataFrame:
    """Compute L2 norm per feature type for each category."""
    rows: List[Dict[str, Any]] = []
    for category, feats in normalized_images.items():
        row: Dict[str, Any] = {'category': category} if include_category_col else {}
        for ftype in ['sift', 'lbp', 'glcm', 'gabor', 'patch']:
            v = feats.get(ftype)
            row[ftype] = float(np.linalg.norm(v)) if isinstance(v, np.ndarray) else 0.0
        rows.append(row)
    df_mag = pd.DataFrame(rows)
    if not include_category_col and 'category' in df_mag.columns:
        df_mag = df_mag.set_index('category')
    return df_mag


def summarize_feature_dims(feature_results: Dict[str, Any]) -> Dict[str, int]:
    """Return feature dimensions per type from feature_results."""
    dims = {
        'SIFT': (feature_results.get('sift_features', np.empty((0, 0))).shape[1]
                 if len(feature_results.get('sift_features', [])) > 0 else 0),
        'LBP': (feature_results.get('lbp_features', np.empty((0, 0))).shape[1]
                if len(feature_results.get('lbp_features', [])) > 0 else 0),
        'GLCM': (feature_results.get('glcm_features', np.empty((0, 0))).shape[1]
                 if len(feature_results.get('glcm_features', [])) > 0 else 0),
        'Gabor': (feature_results.get('gabor_features', np.empty((0, 0))).shape[1]
                  if len(feature_results.get('gabor_features', [])) > 0 else 0),
        'Patches': (feature_results.get('patch_features', np.empty((0, 0))).shape[1]
                    if len(feature_results.get('patch_features', [])) > 0 else 0),
    }
    return dims


def _extract_all_features_from_processed(
    df: pd.DataFrame,
    processed_images: Any,
    feature_extractor: BasicImageFeatureExtractor,
    id_column: str = 'uniq_id',
    image_column: str = 'image',
) -> Tuple[Optional[np.ndarray], List[str], Dict[str, Any]]:
    """
    Extract features for all available processed images in a single batch,
    then combine into a single matrix (per-image).
    """
    by_id, by_image = _build_image_lookup(df, processed_images, id_column, image_column)

    arrays: List[np.ndarray] = []
    names: List[str] = []
    seen = set()

    # Prefer df order; dedupe by basename
    for _, row in df[[image_column]].dropna().iterrows():
        key = _basename(row[image_column])
        if key in by_image and key not in seen and isinstance(by_image[key], np.ndarray):
            arrays.append(by_image[key])
            names.append(key)
            seen.add(key)

    if not arrays:
        return None, [], {}

    feature_extractor.extract_features_batch(arrays, image_names=names)
    combined_features, feature_names = feature_extractor.combine_features()
    return combined_features, feature_names, feature_extractor.feature_results


def run_plotting_pipeline(
    df: pd.DataFrame,
    processed_images: Any,
    feature_extractor: BasicImageFeatureExtractor,
    num_images_per_category: int = 15,
    random_seed: int = 42,
    id_column: str = 'uniq_id',
    image_column: str = 'image',
    category_column: str = 'product_category',
    include_category_col_in_magnitudes: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrates:
    - Normalize images by category and compute averaged feature vectors.
    - Build category-level feature matrix and magnitudes.
    - Extract per-image features (one batch) and compute combined feature matrix (for PCA).
    - Build heatmap, radar, stacked bar, and PCA figures.

    Returns dict with:
      normalized_images, category_feature_matrix, category_feature_names, category_names,
      df_magnitudes, feature_results, combined_features, combined_feature_names, figures
    """
    # Category-level normalization and averages
    normalized_images = normalize_images_by_category(
        df, processed_images, feature_extractor,
        num_images_per_category=num_images_per_category,
        random_seed=random_seed,
        id_column=id_column,
        image_column=image_column,
    )

    # Category feature matrix (flattened)
    cat_matrix, cat_feat_names, cat_names = build_category_feature_matrix(normalized_images)

    # Category magnitudes (L2 per feature type)
    df_magnitudes = compute_category_feature_magnitudes(
        normalized_images, include_category_col=include_category_col_in_magnitudes
    )

    # Per-image features for PCA scatter
    combined_features, combined_feature_names, feature_results = _extract_all_features_from_processed(
        df, processed_images, feature_extractor, id_column=id_column, image_column=image_column
    )

    # Figures
    figures = _build_figures_from_results(
        df=df,
        combined_features=combined_features,
        feature_names=combined_feature_names or [],
        feature_results=feature_results or {},
        df_magnitudes=df_magnitudes,
        image_column=image_column,
        category_column=category_column,
    )

    return {
        'normalized_images': normalized_images,
        'category_feature_matrix': cat_matrix,
        'category_feature_names': cat_feat_names,
        'category_names': cat_names,
        'df_magnitudes': df_magnitudes,
        'feature_results': feature_results,
        'combined_features': combined_features,
        'combined_feature_names': combined_feature_names,
        'figures': figures,
    }


def extract_and_visualize_basic_features(
    df: pd.DataFrame,
    processed_images: Any,
    num_images_per_category: int = 15,
    random_seed: int = 42,
    id_column: str = 'uniq_id',
    image_column: str = 'image',
    category_column: str = 'product_category',
) -> Tuple[
    Dict[str, Dict[str, Optional[np.ndarray]]],
    Dict[str, Any],
    Optional[np.ndarray],
    List[str],
    pd.DataFrame,
    Dict[str, go.Figure]
]:
    """
    Notebook-friendly wrapper:
    - Instantiates a BasicImageFeatureExtractor (v2) with safe defaults.
    - Runs the full plotting pipeline.
    - Returns a tuple conforming to the notebook import usage.
    """
    # Try default constructor, fallback to explicit params if needed
    try:
        feature_extractor = BasicImageFeatureExtractor()
    except TypeError:
        feature_extractor = BasicImageFeatureExtractor(
            sift_features=128, lbp_radius=1, lbp_points=8, patch_size=(16, 16), max_patches=25
        )

    out = run_plotting_pipeline(
        df=df,
        processed_images=processed_images,
        feature_extractor=feature_extractor,
        num_images_per_category=num_images_per_category,
        random_seed=random_seed,
        id_column=id_column,
        image_column=image_column,
        category_column=category_column,
        include_category_col_in_magnitudes=True,
    )

    normalized_images = out['normalized_images']
    feature_results = out['feature_results']
    combined_features = out['combined_features']
    feature_names = out['combined_feature_names'] or []
    df_magnitudes = out['df_magnitudes']
    figs = out['figures']

    return normalized_images, feature_results, combined_features, feature_names, df_magnitudes, figs


def _pluck_arrays_and_names(processed_images: Any) -> Tuple[List[np.ndarray], List[str]]:
    """Best-effort extraction of (arrays, names) from various processed_images shapes."""
    arrays, names = [], []

    # Common dict schema
    if isinstance(processed_images, dict):
        for ak in ('processed_images', 'images', 'image_arrays'):
            if ak in processed_images and isinstance(processed_images[ak], (list, tuple, np.ndarray)):
                arrays = [a for a in processed_images[ak] if isinstance(a, np.ndarray)]
                break
        for nk in ('image_names', 'image_paths', 'filenames', 'images'):
            if nk in processed_images and isinstance(processed_images[nk], (list, tuple, np.ndarray)):
                names = [str(_basename(n)) for n in processed_images[nk]]
                break
        if names and arrays and len(names) != len(arrays):
            names = names[:len(arrays)]
        if not names and arrays:
            names = [f'image_{i+1}' for i in range(len(arrays))]
        return arrays, names

    # Sequence of arrays
    if isinstance(processed_images, (list, tuple, np.ndarray)) and all(isinstance(x, np.ndarray) for x in processed_images):
        arrays = list(processed_images)
        names = [f'image_{i+1}' for i in range(len(arrays))]
        return arrays, names

    # Generic mapping
    if hasattr(processed_images, 'items'):
        try:
            items = [(k, v) for k, v in processed_images.items() if isinstance(v, np.ndarray)]
            names = [str(_basename(k)) for k, _ in items]
            arrays = [v for _, v in items]
            return arrays, names
        except Exception:
            pass

    return [], []


def quick_sample_feature_extraction(
    processed_images: Any,
    sample_size: int = 10
) -> Dict[str, Any]:
    """
    Minimal batch extraction for a quick demo.
    Prefers legacy BasicImageFeatureExtractor if available, else v2.
    Returns dict with 'summary', 'feature_viz', 'feature_results', 'combined_features', 'feature_names'.
    """
    arrays, names = _pluck_arrays_and_names(processed_images)
    if not arrays:
        return {
            'summary': {'images_processed': 0, 'feature_matrix_shape': (0, 0), 'total_features': 0, 'feature_types': []},
            'feature_viz': go.Figure().update_layout(title='No images found'),
            'feature_results': {},
            'combined_features': None,
            'feature_names': []
        }

    # Subsample
    arrays = arrays[:sample_size]
    names = names[:sample_size]

    print(f"🔄 Extracting basic image features from {len(arrays)} images...")

    # Prefer non-v2 if present
    extractor = None
    try:
        from src.classes.basic_image_features import BasicImageFeatureExtractor as BasicImageFeatureExtractorV1
        try:
            extractor = BasicImageFeatureExtractorV1()
        except TypeError:
            extractor = BasicImageFeatureExtractorV1(
                sift_features=128, lbp_radius=1, lbp_points=8, patch_size=(16, 16), max_patches=25
            )
    except Exception:
        # Fallback to v2 already imported at module top
        try:
            extractor = BasicImageFeatureExtractor()
        except TypeError:
            extractor = BasicImageFeatureExtractor(
                sift_features=128, lbp_radius=1, lbp_points=8, patch_size=(16, 16), max_patches=25
            )

    # Extract and combine
    feature_results = extractor.extract_features_batch(arrays, image_names=names)
    combined_features, feature_names = extractor.combine_features()

    # Try extractor-provided viz; otherwise PCA fallback
    if hasattr(extractor, 'create_feature_visualization'):
        fig = extractor.create_feature_visualization()
    else:
        if isinstance(combined_features, np.ndarray) and combined_features.size > 0:
            X = combined_features
            n = X.shape[0]
            X2 = PCA(n_components=2, random_state=0).fit_transform(X) if X.shape[1] > 1 else np.c_[X, np.zeros((n, 1))]
            fig = go.Figure([
                go.Scattergl(x=X2[:, 0], y=X2[:, 1], mode='markers', text=names,
                             hovertemplate='Name: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}')
            ]).update_layout(title='PCA of Combined Image Features (Sample)')
        else:
            fig = go.Figure().update_layout(title='No combined features')

    # Summary
    total_features = int(combined_features.shape[1]) if isinstance(combined_features, np.ndarray) and combined_features.ndim == 2 else 0
    feature_types = [k for k in feature_results.keys() if k != 'image_names' and isinstance(feature_results[k], (list, np.ndarray)) and len(feature_results[k]) > 0]
    summary = {
        'images_processed': len(names),
        'feature_matrix_shape': (len(names), total_features),
        'total_features': total_features,
        'feature_types': feature_types,
    }

    print("✅ Feature extraction complete!")

    return {
        'summary': summary,
        'feature_viz': fig,
        'feature_results': extractor.feature_results,
        'combined_features': combined_features,
        'feature_names': feature_names,
    }
