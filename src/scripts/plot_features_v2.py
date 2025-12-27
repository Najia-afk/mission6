"""
Lightweight classical feature sampling & visualization (v2).

Usage:
    from src.scripts.plot_featuresv2 import quick_sample_feature_extraction
    result = quick_sample_feature_extraction(processing_results, sample_size=12)
    result['feature_viz'].show()
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots


def _extract_feature_matrix(processing_results: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """
    Build (N,D) feature matrix from processing_results['basic_features'] list.
    Returns (matrix, feature_names) or (empty, []) if unavailable.
    """
    feats = processing_results.get('basic_features', [])
    if not feats:
        return np.zeros((0, 0), dtype=np.float32), []
    valid = [f for f in feats if f and f.get('success') and 'features' in f]
    if not valid:
        return np.zeros((0, 0), dtype=np.float32), []
    feature_names = valid[0].get('feature_names', []) or [f'feat_{i}' for i in range(len(valid[0]['features']))]
    X = np.vstack([np.asarray(f['features']).ravel() for f in valid])
    # Sanity align names length
    if len(feature_names) != X.shape[1]:
        feature_names = [f'feat_{i}' for i in range(X.shape[1])]
    return X.astype(np.float32), feature_names


def _quality_dataframe(processing_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a DataFrame of quality & basic stats per image.
    """
    rows = []
    stats_list = processing_results.get('processing_stats', [])
    feats_list = processing_results.get('basic_features', [])
    for i, stats in enumerate(stats_list):
        q = stats.get('quality_info', {}) or {}
        feat_meta = feats_list[i] if i < len(feats_list) else {}
        rows.append({
            'idx': i,
            'quality_score': q.get('quality_score'),
            'blur_score': q.get('blur_score'),
            'brightness': q.get('brightness'),
            'contrast': q.get('contrast'),
            'height': (stats.get('original_size') or (None, None))[0],
            'width': (stats.get('original_size') or (None, None))[1],
            'keypoints': feat_meta.get('keypoints_count'),
            'issues_count': len(q.get('issues', [])),
        })
    return pd.DataFrame(rows)


def build_processing_figures(
    processing_results: Dict[str, Any],
    max_corr_features: int = 40,
    random_state: int = 42
) -> Dict[str, go.Figure]:
    """
    Create a set of exploratory figures from processing_results.
    Returns dict of figures.
    """
    figs: Dict[str, go.Figure] = {}
    qdf = _quality_dataframe(processing_results)
    X, feature_names = _extract_feature_matrix(processing_results)

    # 1. Quality score histogram
    if not qdf.empty and qdf['quality_score'].notna().any():
        figs['quality_hist'] = px.histogram(
            qdf, x='quality_score', nbins=20,
            title='Image Quality Score Distribution'
        ).update_layout(bargap=0.05)
    else:
        figs['quality_hist'] = go.Figure().update_layout(title='Image Quality Score Distribution (no data)')

    # 2. Blur / Brightness / Contrast distributions
    for metric in ['blur_score', 'brightness', 'contrast']:
        if not qdf.empty and qdf[metric].notna().any():
            figs[f'{metric}_hist'] = px.histogram(
                qdf, x=metric, nbins=20,
                title=f'{metric.replace("_", " ").title()} Distribution'
            )
        else:
            figs[f'{metric}_hist'] = go.Figure().update_layout(title=f'{metric.title()} Distribution (no data)')

    # 3. Brightness vs Contrast scatter
    if not qdf.empty and qdf[['brightness', 'contrast']].notna().all(axis=1).any():
        figs['brightness_contrast_scatter'] = px.scatter(
            qdf, x='brightness', y='contrast',
            color='quality_score',
            title='Brightness vs Contrast (colored by quality)',
            hover_data=['blur_score', 'keypoints']
        )
    else:
        figs['brightness_contrast_scatter'] = go.Figure().update_layout(title='Brightness vs Contrast (no data)')

    # 4. Size scatter
    if not qdf.empty and qdf[['height', 'width']].notna().all(axis=1).any():
        figs['size_scatter'] = px.scatter(
            qdf, x='width', y='height',
            color='quality_score',
            title='Original Size Distribution',
            hover_data=['blur_score', 'contrast']
        )
    else:
        figs['size_scatter'] = go.Figure().update_layout(title='Original Size Distribution (no data)')

    # 5. Keypoints histogram
    if not qdf.empty and qdf['keypoints'].notna().any():
        figs['keypoints_hist'] = px.histogram(
            qdf, x='keypoints', nbins=20,
            title='Keypoints Count Distribution'
        )
    else:
        figs['keypoints_hist'] = go.Figure().update_layout(title='Keypoints Count Distribution (no data)')

    # 6. Feature variance bar (top 30)
    if X.size and X.shape[0] > 1:
        variances = np.var(X, axis=0)
        order = np.argsort(variances)[::-1]
        top_k = min(30, len(order))
        top_idx = order[:top_k]
        figs['feature_variance_bar'] = px.bar(
            x=[feature_names[i] for i in top_idx],
            y=variances[top_idx],
            title='Top Feature Variances',
            labels={'x': 'Feature', 'y': 'Variance'}
        ).update_layout(xaxis_tickangle=-60)
    else:
        figs['feature_variance_bar'] = go.Figure().update_layout(title='Top Feature Variances (no data)')

    # 7. Feature correlation heatmap (subset by variance)
    if X.size and X.shape[0] > 2 and X.shape[1] > 2:
        variances = np.var(X, axis=0)
        order = np.argsort(variances)[::-1][:max_corr_features]
        subX = X[:, order]
        if subX.shape[1] > 1:
            corr = np.corrcoef(subX, rowvar=False)
            figs['feature_corr_heatmap'] = go.Figure(
                data=go.Heatmap(
                    z=corr,
                    x=[feature_names[i] for i in order],
                    y=[feature_names[i] for i in order],
                    colorscale='RdBu',
                    zmin=-1, zmax=1,
                    colorbar=dict(title='r')
                )
            ).update_layout(title='Feature Correlation (subset)')
        else:
            figs['feature_corr_heatmap'] = go.Figure().update_layout(title='Feature Correlation (subset insufficient)')
    else:
        figs['feature_corr_heatmap'] = go.Figure().update_layout(title='Feature Correlation (no data)')

    # 8. PCA of features (all images)
    if X.size and X.shape[0] > 1:
        scaler = StandardScaler()
        Xn = scaler.fit_transform(X)
        comps = 2 if X.shape[1] > 1 else 1
        pca_coords = PCA(n_components=comps, random_state=random_state).fit_transform(Xn)
        if comps == 1:
            pca_coords = np.c_[pca_coords, np.zeros((pca_coords.shape[0],))]
        figs['feature_pca'] = px.scatter(
            x=pca_coords[:, 0], y=pca_coords[:, 1],
            title='PCA of Basic Feature Matrix',
            labels={'x': 'PC1', 'y': 'PC2'}
        )
    else:
        figs['feature_pca'] = go.Figure().update_layout(title='PCA of Basic Feature Matrix (no data)')

    return figs


def build_processing_dashboard(processing_results: Dict[str, Any]) -> go.Figure:
    """
    Compact dashboard (4 panels) using a subset of figures.
    """
    figs = build_processing_figures(processing_results)
    qdf = _quality_dataframe(processing_results)  # reuse for rebuild
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Image Quality Score Distribution',  # sync exact title
            'Brightness vs Contrast',
            'Keypoints Distribution',
            'Feature Variances (Top)'
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    if not qdf.empty and qdf['quality_score'].notna().any():
        qs = qdf.loc[qdf['quality_score'].notna(), 'quality_score']
        fig.add_trace(
            go.Histogram(
                x=qs,
                nbinsx=20,
                name='Quality Score',
                marker=dict(color='rgba(30,120,255,0.70)', line=dict(color='#FFFFFF', width=1)),
                opacity=0.9,
                hovertemplate='Quality: %{x:.3f}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        fig.update_xaxes(title_text='Quality Score', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=1)
        # Match standalone bargap
        fig.update_layout(bargap=0.05)
    else:
        fig.add_trace(
            go.Histogram(x=[],
                         name='Quality Score'),
            row=1, col=1
        )
        fig.update_xaxes(title_text='Quality Score', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=1)

    # Panel 2
    for tr in figs['brightness_contrast_scatter'].data:
        fig.add_trace(tr, row=1, col=2)
    # Panel 3
    for tr in figs['keypoints_hist'].data:
        fig.add_trace(tr, row=2, col=1)
    # Panel 4
    for tr in figs['feature_variance_bar'].data:
        fig.add_trace(tr, row=2, col=2)

    fig.update_layout(
        title='Processing Insights Dashboard',
        template='plotly_white',
        showlegend=False,
        height=800,
        width=1100
    )
    return fig


def _pluck_arrays_and_names(processed_images: Dict[str, Any]) -> Tuple[List[Dict[str, np.ndarray]], List[str]]:
    """
    Extract available classical feature dicts and their type names from ImageProcessor's output.

    Expected structure:
        processed_images['basic_features'] = {
            image_id: {
                'sift': np.ndarray,
                'lbp': np.ndarray,
                'glcm': np.ndarray,
                'gabor': np.ndarray,
                'patch': np.ndarray
            }, ...
        }

    Returns:
        arrays_by_type: list of {image_id: feature_vector}
        names: list of feature type names in same order
    """
    arrays_by_type: List[Dict[str, np.ndarray]] = []
    names: List[str] = []

    if not isinstance(processed_images, dict):
        return arrays_by_type, names

    basic = processed_images.get("basic_features")
    if not isinstance(basic, dict) or not basic:
        return arrays_by_type, names

    first_key = next(iter(basic))
    first_val = basic[first_key]
    if not isinstance(first_val, dict):
        # Already flattened (single vector per image)
        flattened = {k: np.asarray(v) for k, v in basic.items()}
        return [flattened], ["classical"]

    types = list(first_val.keys())
    for t in types:
        type_map: Dict[str, np.ndarray] = {}
        for img_id, feats in basic.items():
            if not isinstance(feats, dict):
                continue
            vec = feats.get(t)
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
    Combine dict-of-dicts into a single matrix.

    Returns:
        X             (N, D)
        feature_names list of column names
        image_ids     alignment for rows
        col_types     parallel to feature_names
    """
    if not basic_features:
        return np.zeros((0, 0), dtype=np.float32), [], [], []

    first_key = next(iter(basic_features))
    first_val = basic_features[first_key]
    all_types = list(first_val.keys()) if isinstance(first_val, dict) else ["classical"]

    if feature_order:
        types = [t for t in feature_order if t in all_types] or all_types
    else:
        types = all_types

    image_ids = list(basic_features.keys())
    blocks: List[np.ndarray] = []
    feature_names: List[str] = []
    col_types: List[str] = []

    for t in types:
        # Determine canonical length (median)
        lengths: List[int] = []
        for img_id in image_ids:
            item = basic_features[img_id]
            vec = item if t == "classical" else (item.get(t) if isinstance(item, dict) else None)
            if vec is not None:
                lengths.append(np.asarray(vec).ravel().shape[0])
        L = int(np.median(lengths)) if lengths else 0

        vecs: List[np.ndarray] = []
        for img_id in image_ids:
            item = basic_features[img_id]
            vec = item if t == "classical" else (item.get(t) if isinstance(item, dict) else None)
            if vec is None:
                vec_arr = np.zeros((L,), dtype=np.float32)
            else:
                vec_arr = np.asarray(vec).ravel()
                if L and vec_arr.shape[0] != L:
                    if vec_arr.shape[0] < L:
                        pad = np.zeros((L - vec_arr.shape[0],), dtype=vec_arr.dtype)
                        vec_arr = np.concatenate([vec_arr, pad], axis=0)
                    else:
                        vec_arr = vec_arr[:L]
            vecs.append(vec_arr.astype(np.float32))
        block = np.stack(vecs) if vecs else np.zeros((len(image_ids), 0), dtype=np.float32)
        blocks.append(block)
        feature_names.extend([f"{t}_dim_{i}" for i in range(block.shape[1])])
        col_types.extend([t] * block.shape[1])

    X = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(image_ids), 0), dtype=np.float32)
    return X, feature_names, image_ids, col_types


def quick_sample_feature_extraction(
    processed_images: Dict[str, Any],
    sample_size: int = 10,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Fast PCA visualization on a random subset of classical features.

    Returns:
        {
          'summary': {...},
          'feature_viz': go.Figure,
          'pca': np.ndarray (N,2),
          'sample_ids': list[str],
          'feature_names': list[str],
        }
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    arrays, names = _pluck_arrays_and_names(processed_images)
    if not arrays:
        return {
            "summary": {"images_processed": 0, "feature_matrix_shape": (0, 0),
                        "total_features": 0, "feature_types": []},
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
            "summary": {"images_processed": 0, "feature_matrix_shape": (0, 0),
                        "total_features": 0, "feature_types": list(dict.fromkeys(col_types))},
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