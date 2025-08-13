import os
import inspect
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
import plotly.express as px

from src.classes.basic_image_featuresv2 import BasicImageFeatureExtractor

try:
    from src.classes.vgg16_extractor import VGG16FeatureExtractor  # noqa: F401
except Exception:
    VGG16FeatureExtractor = None  # Avoid import errors from unfinished class


def _basename(p: str) -> str:
    """Safe basename for mapping names in DataFrame to processed image names."""
    return os.path.basename(p) if isinstance(p, str) else str(p)


def _feature_type_from_name(name: str) -> str:
    n = name.lower()
    for k in ("sift", "lbp", "glcm", "gabor", "patch"):
        if n.startswith(k):
            return k
    return "other"


def _pluck_arrays_and_names(processed_images: Dict[str, Any]) -> Tuple[List[Dict[str, np.ndarray]], List[str]]:
    """
    Extract available classical feature dicts and their type names from ImageProcessor's output.
    Expected structure:
        processed_images['basic_features'] = {
            image_path_or_name: {
                'sift': np.ndarray(...),
                'lbp': np.ndarray(...),
                'glcm': np.ndarray(...),
                'gabor': np.ndarray(...),
                'patch_stats': np.ndarray(...)
            },
            ...
        }
    Returns:
        - arrays_by_type: list of dicts [ {image_id: feature_vec}, ... ] aligned to 'names'
        - names: list of feature type names (e.g., ['sift','lbp',...])
    """
    arrays_by_type: List[Dict[str, np.ndarray]] = []
    names: List[str] = []

    if not isinstance(processed_images, dict):
        return arrays_by_type, names

    basic = processed_images.get("basic_features")
    if not isinstance(basic, dict) or not basic:
        return arrays_by_type, names

    # Detect feature types from the first entry
    first_key = next(iter(basic))
    first_val = basic[first_key]
    if not isinstance(first_val, dict):
        # If already flattened features per image, treat as single feature type
        name = "classical"
        ft_map = {k: np.asarray(v) for k, v in basic.items()}
        return [ft_map], [name]

    types = list(first_val.keys())
    for t in types:
        type_map: Dict[str, np.ndarray] = {}
        for img_id, feats in basic.items():
            if isinstance(feats, dict) and t in feats:
                vec = feats[t]
                if vec is None:
                    continue
                type_map[img_id] = np.asarray(vec).ravel()
        if type_map:
            arrays_by_type.append(type_map)
            names.append(t)

    return arrays_by_type, names


def _combine_feature_dict(
    basic_features: Dict[str, Dict[str, np.ndarray]],
    feature_order: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """
    Combine dict-of-dicts into a single 2D matrix [N, D] with concatenated features.
    Returns:
        - X: combined features
        - feature_names: column names
        - image_ids: rows alignment
        - col_types: type name per column (same length as feature_names)
    """
    if not basic_features:
        return np.zeros((0, 0), dtype=np.float32), [], [], []

    # Determine types
    first_key = next(iter(basic_features))
    first_val = basic_features[first_key]
    if isinstance(first_val, dict):
        all_types = list(first_val.keys())
    else:
        # Already flattened single vector per image
        all_types = ["classical"]

    if feature_order:
        types = [t for t in feature_order if t in all_types]
        if not types:
            types = all_types
    else:
        types = all_types

    # Collect image ids
    image_ids = list(basic_features.keys())

    # Prepare building
    blocks: List[np.ndarray] = []
    feature_names: List[str] = []
    col_types: List[str] = []

    for t in types:
        # collect vectors, fill missing with zeros of the median size observed
        vecs: List[np.ndarray] = []
        lengths: List[int] = []
        for img_id in image_ids:
            item = basic_features[img_id]
            vec = item if t == "classical" else item.get(t, None)
            if vec is not None:
                lengths.append(np.asarray(vec).ravel().shape[0])
        L = int(np.median(lengths)) if lengths else 0

        for img_id in image_ids:
            item = basic_features[img_id]
            vec = item if t == "classical" else item.get(t, None)
            if vec is None:
                vec = np.zeros((L,), dtype=np.float32)
            else:
                vec = np.asarray(vec).ravel()
                if L and vec.shape[0] != L:
                    # pad or truncate
                    if vec.shape[0] < L:
                        pad = np.zeros((L - vec.shape[0],), dtype=vec.dtype)
                        vec = np.concatenate([vec, pad], axis=0)
                    else:
                        vec = vec[:L]
            vecs.append(vec.astype(np.float32))
        block = np.stack(vecs)
        blocks.append(block)

        # names for this block
        for i in range(block.shape[1]):
            feature_names.append(f"{t}_dim_{i}")
            col_types.append(t)

    # concatenate
    X = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(image_ids), 0), dtype=np.float32)
    return X, feature_names, image_ids, col_types


def extract_and_visualize_basic_features(
    df: pd.DataFrame,
    processed_images: Dict[str, Any],
    num_images_per_category: int = 15,
    random_seed: int = 42,
    id_column: str = "uniq_id",
    image_column: str = "image",
    category_column: str = "product_category",
) -> Tuple[List[str], Dict[str, Any], np.ndarray, List[str], pd.DataFrame, Dict[str, go.Figure]]:
    """
    Samples images by category, builds a combined classical feature matrix from processed_images,
    and returns multiple Plotly figures for analysis.

    Returns:
        normalized_images (list[str]): the image ids/paths used
        feature_results (dict): summary info
        combined_features (np.ndarray)
        feature_names (list[str])
        df_magnitudes (pd.DataFrame): per-category mean abs magnitude by feature type
        figs (dict[str, go.Figure]): 'feature_viz','heatmap','radar','stacked_bar'
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Get available arrays and names
    arrays, names = _pluck_arrays_and_names(processed_images)
    if not arrays:
        empty_figs = {
            'feature_viz': go.Figure().update_layout(title='No images found'),
            'heatmap': go.Figure().update_layout(title='No data available'),
            'radar': go.Figure().update_layout(title='No data available'),
            'stacked_bar': go.Figure().update_layout(title='No data available'),
        }
        return [], {}, np.zeros((0, 0), dtype=np.float32), [], pd.DataFrame(), empty_figs

    # Combine into single matrix
    basic = processed_images.get("basic_features", {})
    X, feature_names, image_ids, col_types = _combine_feature_dict(basic, feature_order=names)

    # Build mapping from df to category
    df_map = {}
    if isinstance(df, pd.DataFrame) and image_column in df.columns and category_column in df.columns:
        df_map = df.set_index(image_column)[category_column].to_dict()

    # Convert image_ids to plain names if they are full paths
    def to_name(s: str) -> str:
        return s.split("/")[-1]

    labels = []
    for img_id in image_ids:
        key = img_id
        if key not in df_map:
            key = to_name(img_id)
        labels.append(df_map.get(key, "Unknown"))

    # Sample by category
    image_ids_np = np.array(image_ids)
    labels_np = np.array(labels)
    sampled_idx: List[int] = []
    for cat in np.unique(labels_np):
        idx = np.where(labels_np == cat)[0]
        if idx.size == 0:
            continue
        np.random.shuffle(idx)
        n = min(num_images_per_category, idx.size)
        sampled_idx.extend(idx[:n])

    sampled_idx = np.array(sampled_idx, dtype=int)
    sampled_X = X[sampled_idx] if sampled_idx.size else X
    sampled_labels = labels_np[sampled_idx] if sampled_idx.size else labels_np
    sampled_image_ids = image_ids_np[sampled_idx] if sampled_idx.size else image_ids_np

    # Normalize and PCA for visualization
    if sampled_X.size:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xn = scaler.fit_transform(sampled_X)
        pca = PCA(n_components=2, random_state=random_seed)
        pca2 = pca.fit_transform(Xn)
    else:
        Xn = sampled_X
        pca2 = np.zeros((len(sampled_labels), 2), dtype=np.float32)

    # Scatter plot
    fig_scatter = px.scatter(
        x=pca2[:, 0], y=pca2[:, 1],
        color=sampled_labels,
        hover_name=[str(s) for s in sampled_image_ids],
        title="PCA of Classical Image Features"
    )
    fig_scatter.update_layout(legend_title_text="Category")

    # Per-category mean absolute magnitude by feature type
    df_rows: List[Dict[str, Any]] = []
    types_unique = list(dict.fromkeys(col_types))
    if sampled_X.size:
        # compute per-type magnitude
        type_slices: Dict[str, List[int]] = {}
        for i, t in enumerate(col_types):
            type_slices.setdefault(t, []).append(i)

        mags = {}
        for t, idxs in type_slices.items():
            block = np.abs(sampled_X[:, idxs])
            mags[t] = block.mean(axis=1)

        for cat in np.unique(sampled_labels):
            mask = sampled_labels == cat
            row = {"category": cat}
            for t in types_unique:
                if t in mags:
                    row[t] = float(np.mean(mags[t][mask])) if np.any(mask) else 0.0
                else:
                    row[t] = 0.0
            df_rows.append(row)

    df_mags = pd.DataFrame(df_rows) if df_rows else pd.DataFrame(columns=["category"] + types_unique)

    # Heatmap
    if not df_mags.empty:
        heat = go.Figure(
            data=go.Heatmap(
                z=df_mags[types_unique].values,
                x=types_unique,
                y=df_mags["category"].values,
                colorscale="Viridis"
            )
        ).update_layout(title="Feature Type Activation by Category")
    else:
        heat = go.Figure().update_layout(title="Feature Type Activation by Category (no data)")

    # Radar (use up to 5 categories)
    radar = go.Figure().update_layout(title="Category Radar (feature type means)")
    if not df_mags.empty:
        cats = df_mags["category"].tolist()[:5]
        for cat in cats:
            row = df_mags[df_mags["category"] == cat]
            if not row.empty:
                radar.add_trace(go.Scatterpolar(
                    r=row[types_unique].values[0].tolist(),
                    theta=types_unique,
                    fill='toself',
                    name=str(cat)
                ))
        radar.update_layout(polar=dict(radialaxis=dict(visible=True)))

    # Stacked bar of total contributions
    stack = go.Figure().update_layout(title="Global Contribution per Feature Type")
    if sampled_X.size:
        totals = []
        for t in types_unique:
            idxs = [i for i, ct in enumerate(col_types) if ct == t]
            val = float(np.mean(np.abs(sampled_X[:, idxs]))) if idxs else 0.0
            totals.append((t, val))
        if totals:
            df_tot = pd.DataFrame(totals, columns=["type", "value"]).sort_values("value", ascending=False)
            stack = px.bar(df_tot, x="type", y="value", title="Global Contribution per Feature Type")

    feature_results = {
        "images_processed": int(sampled_X.shape[0]),
        "feature_matrix_shape": tuple(sampled_X.shape),
        "total_features": int(sampled_X.shape[1]) if sampled_X.ndim == 2 else 0,
        "feature_types": types_unique,
    }

    figs = {
        "feature_viz": fig_scatter,
        "heatmap": heat,
        "radar": radar,
        "stacked_bar": stack,
    }
    return sampled_image_ids.tolist(), feature_results, sampled_X, feature_names, df_mags, figs


def quick_sample_feature_extraction(
    processed_images: Dict[str, Any],
    sample_size: int = 10,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Quick sample using a small subset of the available classical features for a fast visualization.
    Returns a dict with summary & a PCA scatter figure.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    arrays, names = _pluck_arrays_and_names(processed_images)
    if not arrays:
        return {
            "summary": {"images_processed": 0, "feature_matrix_shape": (0, 0), "total_features": 0, "feature_types": []},
            "feature_viz": go.Figure().update_layout(title="No features available"),
        }

    # Combine full, then sample rows
    basic = processed_images.get("basic_features", {})
    X, feature_names, image_ids, col_types = _combine_feature_dict(basic, feature_order=names)
    n = X.shape[0]
    if n == 0:
        return {
            "summary": {"images_processed": 0, "feature_matrix_shape": (0, 0), "total_features": 0, "feature_types": list(dict.fromkeys(col_types))},
            "feature_viz": go.Figure().update_layout(title="No features available"),
        }

    idx = np.random.choice(n, size=min(sample_size, n), replace=False)
    sX = X[idx]
    sids = [image_ids[i] for i in idx]

    if sX.size:
        scaler = StandardScaler()
        Xn = scaler.fit_transform(sX)