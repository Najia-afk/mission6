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
    processed_images: Any,
    num_images_per_category: int = 15,
    random_seed: int = 42,
    id_column: str = "uniq_id",
    image_column: str = "image",
    category_column: str = "product_category",
) -> Tuple[Dict[str, Dict[str, Optional[np.ndarray]]], Dict[str, Any], np.ndarray, List[str], pd.DataFrame, Dict[str, go.Figure]]:
    """
    Unified feature extraction & visualization helper.

    RETURN ORDER (updated):
        normalized_images (dict): {category: {feature_type: mean_vector_or_None}}
        feature_results / summary (dict)
        sampled_X (np.ndarray)
        feature_names (list[str])
        df_mags (pd.DataFrame)
        figs (dict[str, go.Figure])

    (Changed first element from sampled_image_ids list -> normalized_images dict to match
     notebook usage: normalized_images, feature_results, combined_features, feature_names, df_magnitudes, figs = ...)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Helper to build figures (shared later)
    def _build_category_figs(sampled_X, col_types, sampled_labels, sampled_image_ids,
                             normalized_images_dict) -> Tuple[pd.DataFrame, Dict[str, go.Figure]]:
        # PCA scatter (per-image)
        if sampled_X.size:
            scaler = StandardScaler()
            Xn = scaler.fit_transform(sampled_X)
            pca2 = PCA(n_components=2, random_state=random_seed).fit_transform(Xn) if sampled_X.shape[1] > 1 else np.c_[Xn, np.zeros((Xn.shape[0],))]
        else:
            pca2 = np.zeros((len(sampled_labels), 2))

        feature_viz = px.scatter(
            x=pca2[:, 0], y=pca2[:, 1],
            color=sampled_labels,
            hover_name=[str(s) for s in sampled_image_ids],
            title="PCA of Classical Image Features"
        ).update_layout(legend_title_text="Category")

        # Per-category mean magnitude per feature type from normalized_images (category means)
        rows = []
        feature_types = []
        for cat, feats in normalized_images_dict.items():
            row = {'category': cat}
            for ft, vec in feats.items():
                if vec is not None:
                    row[ft] = float(np.linalg.norm(vec))
                    if ft not in feature_types:
                        feature_types.append(ft)
                else:
                    row[ft] = 0.0
            rows.append(row)
        df_mags_local = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['category'])

        # Heatmap
        if not df_mags_local.empty:
            heat = go.Figure(data=go.Heatmap(
                z=df_mags_local[feature_types].values,
                x=feature_types,
                y=df_mags_local['category'].values,
                colorscale='Viridis'
            )).update_layout(title="Feature Type Activation by Category",
                             xaxis_title="Feature Type", yaxis_title="Category")
        else:
            heat = go.Figure().update_layout(title="Feature Type Activation by Category (no data)")

        # Radar
        radar = go.Figure().update_layout(title="Category Radar (feature type magnitudes)")
        if not df_mags_local.empty and feature_types:
            # Normalize per feature type for shape comparison
            norm_df = df_mags_local.copy()
            max_per_ft = norm_df[feature_types].replace(0, np.nan).max(axis=0)
            norm_df[feature_types] = norm_df[feature_types].divide(max_per_ft, axis=1).fillna(0.0)
            for _, r in norm_df.iterrows():
                radar.add_trace(go.Scatterpolar(
                    r=[r[ft] for ft in feature_types],
                    theta=feature_types,
                    fill='toself',
                    name=str(r['category'])
                ))
            radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))

        # Stacked bar (global contribution per feature type across sampled images)
        stacked = go.Figure().update_layout(title="Global Contribution per Feature Type")
        if sampled_X.size:
            totals = []
            for ft in feature_types:
                ft_cols = [i for i, t in enumerate(col_types) if t == ft]
                val = float(np.mean(np.abs(sampled_X[:, ft_cols]))) if ft_cols else 0.0
                totals.append((ft, val))
            if totals:
                df_tot = pd.DataFrame(totals, columns=['type', 'value']).sort_values('value', ascending=False)
                stacked = px.bar(df_tot, x='type', y='value', title="Global Contribution per Feature Type")

        figs_local = {
            'feature_viz': feature_viz,
            'heatmap': heat,
            'radar': radar,
            'stacked_bar': stacked
        }
        return df_mags_local, figs_local

    # Path A: precomputed classical features in processed_images['basic_features']
    if isinstance(processed_images, dict) and isinstance(processed_images.get('basic_features'), dict) and processed_images['basic_features']:
        arrays, names = _pluck_arrays_and_names(processed_images)
        if not arrays:
            empty = {
                'feature_viz': go.Figure().update_layout(title='No images found'),
                'heatmap': go.Figure().update_layout(title='No data available'),
                'radar': go.Figure().update_layout(title='No data available'),
                'stacked_bar': go.Figure().update_layout(title='No data available'),
            }
            return {}, {}, np.zeros((0, 0), np.float32), [], pd.DataFrame(), empty

        basic = processed_images['basic_features']
        X, feature_names, image_ids, col_types = _combine_feature_dict(basic, feature_order=names)

        # Map image_ids -> categories
        df_map = {}
        if isinstance(df, pd.DataFrame) and image_column in df.columns and category_column in df.columns:
            df_map = df.set_index(image_column)[category_column].to_dict()

        def to_name(s: str) -> str:
            return s.split('/')[-1]

        labels = []
        for img_id in image_ids:
            key = img_id if img_id in df_map else to_name(img_id)
            labels.append(df_map.get(key, 'Unknown'))

        # Per-category sampling
        image_ids_np = np.array(image_ids)
        labels_np = np.array(labels)
        sampled_idx = []
        for cat in np.unique(labels_np):
            cat_idx = np.where(labels_np == cat)[0]
            if cat_idx.size == 0:
                continue
            np.random.shuffle(cat_idx)
            k = min(num_images_per_category, cat_idx.size)
            sampled_idx.extend(cat_idx[:k])
        sampled_idx = np.array(sampled_idx, dtype=int) if sampled_idx else np.arange(len(image_ids_np))

        sampled_X = X[sampled_idx]
        sampled_labels = labels_np[sampled_idx]
        sampled_image_ids = image_ids_np[sampled_idx]

        # Build normalized category means (using sampled images only)
        normalized_images = {}
        for cat in np.unique(sampled_labels):
            mask = sampled_labels == cat
            cat_feats = {}
            for t in set(col_types):
                ft_cols = [i for i, c in enumerate(col_types) if c == t]
                if ft_cols:
                    block = sampled_X[mask][:, ft_cols]
                    cat_feats[t] = block.mean(axis=0)
                else:
                    cat_feats[t] = None
            normalized_images[cat] = cat_feats

        df_mags, figs = _build_category_figs(sampled_X, col_types, sampled_labels, sampled_image_ids, normalized_images)

        feature_results = {
            "images_processed": int(sampled_X.shape[0]),
            "feature_matrix_shape": tuple(sampled_X.shape),
            "total_features": int(sampled_X.shape[1]),
            "feature_types": list(dict.fromkeys(col_types)),
        }
        return normalized_images, feature_results, sampled_X, feature_names, df_mags, figs

    # Path B: raw list/array of image tensors aligned with df rows (perform extraction here)
    if not isinstance(df, pd.DataFrame) or id_column not in df.columns or category_column not in df.columns:
        empty = {
            'feature_viz': go.Figure().update_layout(title='Missing DataFrame / columns'),
            'heatmap': go.Figure().update_layout(title='No data available'),
            'radar': go.Figure().update_layout(title='No data available'),
            'stacked_bar': go.Figure().update_layout(title='No data available'),
        }
        return {}, {}, np.zeros((0, 0), np.float32), [], pd.DataFrame(), empty

    # Ensure iterable of images
    if isinstance(processed_images, (list, tuple, np.ndarray)):
        raw_images = list(processed_images)
    else:
        # Unsupported structure
        empty = {
            'feature_viz': go.Figure().update_layout(title='Unsupported processed_images format'),
            'heatmap': go.Figure().update_layout(title='No data available'),
            'radar': go.Figure().update_layout(title='No data available'),
            'stacked_bar': go.Figure().update_layout(title='No data available'),
        }
        return {}, {}, np.zeros((0, 0), np.float32), [], pd.DataFrame(), empty

    # Align length if mismatch
    if len(raw_images) != len(df):
        min_len = min(len(raw_images), len(df))
        df = df.iloc[:min_len].reset_index(drop=True)
        raw_images = raw_images[:min_len]

    # Build per-category selection
    sampled_arrays: List[np.ndarray] = []
    sampled_names: List[str] = []
    sampled_labels: List[str] = []
    for cat, grp in df.groupby(category_column):
        ids = grp[id_column].tolist()
        pos = grp.index.tolist()
        if not pos:
            continue
        sel_pos = pos
        if num_images_per_category and len(pos) > num_images_per_category:
            sel_pos = np.random.choice(pos, size=num_images_per_category, replace=False)
        for p in sel_pos:
            arr = raw_images[p]
            if isinstance(arr, np.ndarray):
                sampled_arrays.append(arr)
                sampled_names.append(str(df.loc[p, id_column]))
                sampled_labels.append(str(cat))

    if not sampled_arrays:
        empty = {
            'feature_viz': go.Figure().update_layout(title='No images after sampling'),
            'heatmap': go.Figure().update_layout(title='No data available'),
            'radar': go.Figure().update_layout(title='No data available'),
            'stacked_bar': go.Figure().update_layout(title='No data available'),
        }
        return {}, {}, np.zeros((0, 0), np.float32), [], pd.DataFrame(), empty

    # Extract features in one batch
    try:
        extractor = BasicImageFeatureExtractor()
    except TypeError:
        extractor = BasicImageFeatureExtractor(
            sift_features=128, lbp_radius=1, lbp_points=8, patch_size=(16, 16), max_patches=25
        )

    feature_results = extractor.extract_features_batch(sampled_arrays, image_names=sampled_names)
    combined_features, feature_names = extractor.combine_features()

    # Derive col_types from feature_names
    col_types = [_feature_type_from_name(n) for n in feature_names]

    # Build normalized category mean feature vectors
    name_to_index = {n: i for i, n in enumerate(sampled_names)}
    normalized_images: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    feature_type_keys = [k for k in ('sift_features', 'lbp_features', 'glcm_features', 'gabor_features', 'patch_features')
                         if k in feature_results and len(feature_results[k]) > 0]

    for cat in sorted(set(sampled_labels)):
        # indices for this category
        cat_indices = [i for i, lbl in enumerate(sampled_labels) if lbl == cat]
        cat_feats: Dict[str, Optional[np.ndarray]] = {}
        for fk in feature_type_keys:
            block = np.array(feature_results[fk])
            if block.size and block.ndim == 2:
                cat_feats[fk.replace('_features', '')] = block[cat_indices].mean(axis=0) if cat_indices else None
            else:
                cat_feats[fk.replace('_features', '')] = None
        normalized_images[cat] = cat_feats

    sampled_X = combined_features  # per-image matrix
    sampled_image_ids = sampled_names
    sampled_labels_arr = np.array(sampled_labels)

    # Figures + df_mags
    # Convert normalized_images keys (feature type names already simplified)
    normalized_images_simple = normalized_images
    df_mags, figs = _build_category_figs(sampled_X, col_types, sampled_labels_arr, sampled_image_ids, normalized_images_simple)

    summary = {
        "images_processed": len(sampled_names),
        "feature_matrix_shape": tuple(sampled_X.shape),
        "total_features": int(sampled_X.shape[1]),
        "feature_types": sorted({ft.replace('_features', '') for ft in feature_type_keys}),
    }
    feature_results_summary = {**feature_results, "summary": summary}

    return normalized_images, summary, sampled_X, feature_names, df_mags, figs


def quick_sample_feature_extraction(
    processed_images: Dict[str, Any],
    sample_size: int = 10,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Quick sample using a small subset of the available classical features for a fast visualization.
    Returns:
        {
          'summary': {...},
          'feature_viz': go.Figure,
          'pca': np.ndarray (2D coords),
          'sample_ids': list,
          'feature_names': list,
        }
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    arrays, names = _pluck_arrays_and_names(processed_images)
    if not arrays:
        return {
            "summary": {"images_processed": 0, "feature_matrix_shape": (0, 0), "total_features": 0, "feature_types": []},
            "feature_viz": go.Figure().update_layout(title="No features available"),
            "pca": np.zeros((0, 2)),
            "sample_ids": [],
            "feature_names": [],
        }

    basic = processed_images.get("basic_features", {})
    X, feature_names, image_ids, col_types = _combine_feature_dict(basic, feature_order=names)
    n = X.shape[0]
    if n == 0:
        return {
            "summary": {"images_processed": 0, "feature_matrix_shape": (0, 0), "total_features": 0,
                        "feature_types": list(dict.fromkeys(col_types))},
            "feature_viz": go.Figure().update_layout(title="No features available"),
            "pca": np.zeros((0, 2)),
            "sample_ids": [],
            "feature_names": feature_names,
        }

    idx = np.random.choice(n, size=min(sample_size, n), replace=False)
    sX = X[idx]
    sids = [image_ids[i] for i in idx]

    if sX.size:
        scaler = StandardScaler()
        Xn = scaler.fit_transform(sX)
        if sX.shape[1] > 1:
            pca2 = PCA(n_components=2, random_state=random_seed).fit_transform(Xn)
        else:
            pca2 = np.c_[Xn, np.zeros((Xn.shape[0],))]
    else:
        pca2 = np.zeros((len(sids), 2))

    if pca2.size:
        feature_viz = go.Figure([
            go.Scatter(
                x=pca2[:, 0],
                y=pca2[:, 1],
                mode="markers",
                text=sids,
                marker=dict(size=7, opacity=0.85),
                hovertemplate="ID: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>"
            )
        ]).update_layout(title="PCA (sampled classical features)",
                         xaxis_title="PC1", yaxis_title="PC2")
    else:
        feature_viz = go.Figure().update_layout(title="No PCA data")

    summary = {
        "images_processed": int(sX.shape[0]),
        "feature_matrix_shape": tuple(sX.shape),
        "total_features": int(sX.shape[1]),
        "feature_types": list(dict.fromkeys(col_types)),
    }

    return {
        "summary": summary,
        "feature_viz": feature_viz,
        "pca": pca2,
        "sample_ids": sids,
        "feature_names": feature_names,
    }


def debug_processed_images_structure(processed_images: Dict[str, Any]) -> None:
    """
    Print a concise summary of processed_images['basic_features'] for debugging.
    """
    if not isinstance(processed_images, dict):
        print("processed_images is not a dict")
        return
    bf = processed_images.get("basic_features")
    if not isinstance(bf, dict) or not bf:
        print("basic_features missing or empty")
        print("Top-level keys:", list(processed_images.keys())[:10])
        return
    print(f"basic_features images: {len(bf)}")
    first_key = next(iter(bf))
    print("Example image key:", first_key)
    first_val = bf[first_key]
    if isinstance(first_val, dict):
        print("Feature types:", list(first_val.keys()))
        for t, v in first_val.items():
            if v is not None:
                print(f"  {t}: shape={np.asarray(v).shape}")
    else:
        print("basic_features values are not dicts; single feature vector mode")