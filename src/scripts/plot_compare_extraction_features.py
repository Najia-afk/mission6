"""
Method Comparison Visualization Script
Creates a comprehensive comparison dashboard for multiple ML methods
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional

def compare_methods(
    methods_data: Dict[str, Dict[str, Any]],
    title: str = 'Method Comparison',
    height: int = 800,
    width: int = 1200
) -> go.Figure:
    """
    Create a comprehensive comparison visualization for different methods.
    
    Args:
        methods_data: Dictionary with method names as keys and metric dictionaries as values.
                     Each metric dictionary should contain keys like:
                     - 'ari_score': Adjusted Rand Index score
                     - 'silhouette_score': Silhouette score
                     - 'pca_dims': Number of PCA dimensions
                     - 'original_dims': Original feature dimensions
                     - 'categories': Number of categories
        title: Title for the figure
        height: Height of the figure
        width: Width of the figure
        
    Returns:
        plotly.graph_objects.Figure: The comparison visualization
    """
    # Extract method names and metrics
    methods = list(methods_data.keys())
    
    # Extract metrics for visualization
    ari_scores = [data.get('ari_score', 0) for data in methods_data.values()]
    silhouette_scores = [data.get('silhouette_score', 0) for data in methods_data.values()]
    feature_dims = [data.get('pca_dims', 0) for data in methods_data.values()]
    original_dims = [data.get('original_dims', 0) for data in methods_data.values()]
    categories = [data.get('categories', 0) for data in methods_data.values()]
    
    # Create subplots
    comparison_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'ARI Score Comparison',
            'Silhouette Score Comparison', 
            'Feature Dimensionality',
            'Processing Performance'
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "table"}]
        ]
    )
    
    # Define colors for methods
    method_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 1. ARI Score Comparison
    comparison_fig.add_trace(
        go.Bar(
            x=methods,
            y=ari_scores,
            name='ARI Score',
            marker_color=method_colors[:len(methods)],
            text=[f'{score:.4f}' for score in ari_scores],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # 2. Silhouette Score Comparison
    comparison_fig.add_trace(
        go.Bar(
            x=methods,
            y=silhouette_scores,
            name='Silhouette Score',
            marker_color=method_colors[2:2+len(methods)],
            text=[f'{score:.3f}' for score in silhouette_scores],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # 3. Feature Dimensionality
    comparison_fig.add_trace(
        go.Bar(
            x=methods,
            y=feature_dims,
            name='PCA Dimensions',
            marker_color=method_colors[4:4+len(methods)],
            text=feature_dims,
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # 4. Performance Table
    performance_data = [
        ['Method', 'ARI Score', 'Silhouette', 'PCA Dims', 'Original Dims', 'Categories']
    ]
    
    # Add data for each method
    for method in methods:
        data = methods_data[method]
        performance_data.append([
            method,
            f"{data.get('ari_score', 0):.4f}",
            f"{data.get('silhouette_score', 0):.3f}",
            str(data.get('pca_dims', 0)),
            str(data.get('original_dims', 0)),
            str(data.get('categories', 0))
        ])
    
    comparison_fig.add_trace(
        go.Table(
            header=dict(
                values=performance_data[0],
                fill_color='lightblue',
                align='center',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=list(zip(*performance_data[1:])),
                fill_color='white',
                align='center',
                font=dict(size=11)
            )
        ),
        row=2, col=2
    )
    
    comparison_fig.update_layout(
        title=title,
        height=height,
        width=width,
        template='plotly_white',
        showlegend=False
    )
    
    return comparison_fig