"""
Comparison Visualizer Class
Module for creating comparative analysis visualizations, particularly for ML model performance comparisons.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import numpy as np


class ComparisonVisualizer:
    """
    A class for creating comparative visualizations for machine learning results.
    Supports bar charts, heatmaps, and multi-metric comparisons.
    """
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize the comparison visualizer.
        
        Args:
            theme: Plotly theme to use for visualizations
        """
        self.theme = theme
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
        ]
    
    def create_ari_comparison_chart(self, 
                                   scores: Dict[str, float], 
                                   title: str = "Adjusted Rand Index Comparison",
                                   height: int = 500,
                                   width: int = 700) -> go.Figure:
        """
        Create a bar chart comparing Adjusted Rand Index scores across different methods.
        
        Args:
            scores: Dictionary mapping method names to ARI scores
            title: Chart title
            height: Chart height in pixels
            width: Chart width in pixels
            
        Returns:
            Plotly figure object
        """
        # Prepare data
        methods = list(scores.keys())
        values = list(scores.values())
        
        # Sort by score (descending)
        sorted_data = sorted(zip(methods, values), key=lambda x: x[1], reverse=True)
        methods_sorted, values_sorted = zip(*sorted_data)
        
        # Create color map based on performance
        colors = []
        for score in values_sorted:
            if score >= 0.6:
                colors.append('#2ca02c')  # Green for excellent
            elif score >= 0.4:
                colors.append('#ff7f0e')  # Orange for good
            elif score >= 0.2:
                colors.append('#d62728')  # Red for moderate
            else:
                colors.append('#7f7f7f')  # Gray for poor
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=methods_sorted,
            y=values_sorted,
            marker_color=colors,
            text=[f'{score:.4f}' for score in values_sorted],
            textposition='auto',
            textfont=dict(size=12, color='white'),
            hovertemplate='<b>%{x}</b><br>ARI Score: %{y:.4f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16, color='#2c3e50')
            ),
            xaxis=dict(
                title=dict(text="Embedding Method", font=dict(size=14)),
                tickfont=dict(size=12),
                tickangle=45
            ),
            yaxis=dict(
                title=dict(text="Adjusted Rand Index", font=dict(size=14)),
                tickfont=dict(size=12),
                range=[0, max(values) * 1.1]
            ),
            height=height,
            width=width,
            template=self.theme,
            showlegend=False,
            margin=dict(t=60, b=100, l=80, r=40)
        )
        
        # Add performance threshold lines
        fig.add_hline(y=0.6, line_dash="dash", line_color="green", 
                     annotation_text="Excellent (≥0.6)", annotation_position="top right")
        fig.add_hline(y=0.4, line_dash="dash", line_color="orange",
                     annotation_text="Good (≥0.4)", annotation_position="top right")
        fig.add_hline(y=0.2, line_dash="dash", line_color="red",
                     annotation_text="Moderate (≥0.2)", annotation_position="top right")
        
        return fig
    
    def create_multi_metric_comparison(self, 
                                     metrics_data: Dict[str, Dict[str, float]],
                                     title: str = "Multi-Metric Performance Comparison",
                                     height: int = 600,
                                     width: int = 800) -> go.Figure:
        """
        Create a grouped bar chart comparing multiple metrics across methods.
        
        Args:
            metrics_data: Nested dict {method: {metric: score}}
            title: Chart title
            height: Chart height
            width: Chart width
            
        Returns:
            Plotly figure object
        """
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(metrics_data).T
        
        # Create grouped bar chart
        fig = go.Figure()
        
        metrics = df.columns
        colors = self.color_palette[:len(metrics)]
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric,
                x=df.index,
                y=df[metric],
                marker_color=colors[i],
                text=[f'{score:.3f}' for score in df[metric]],
                textposition='auto'
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis=dict(title=dict(text="Method", font=dict(size=14))),
            yaxis=dict(title=dict(text="Score", font=dict(size=14))),
            height=height,
            width=width,
            template=self.theme,
            barmode='group'
        )
        
        return fig
    
    def create_ranking_table(self, 
                           scores: Dict[str, float],
                           metric_name: str = "ARI Score") -> pd.DataFrame:
        """
        Create a ranking table for the comparison results.
        
        Args:
            scores: Dictionary mapping method names to scores
            metric_name: Name of the metric being compared
            
        Returns:
            DataFrame with ranking information
        """
        # Sort by score (descending)
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        ranking_data = []
        for rank, (method, score) in enumerate(sorted_items, 1):
            if score >= 0.6:
                performance = "🟢 Excellent"
            elif score >= 0.4:
                performance = "🟡 Good"
            elif score >= 0.2:
                performance = "🟠 Moderate"
            else:
                performance = "🔴 Poor"
                
            ranking_data.append({
                'Rank': rank,
                'Method': method,
                metric_name: f"{score:.4f}",
                'Performance': performance
            })
        
        return pd.DataFrame(ranking_data)
    
    def create_radar_chart(self, 
                          metrics_data: Dict[str, Dict[str, float]],
                          title: str = "Multi-Dimensional Performance Comparison",
                          height: int = 600,
                          width: int = 700) -> go.Figure:
        """
        Create a radar chart for multi-metric comparison.
        
        Args:
            metrics_data: Nested dict {method: {metric: score}}
            title: Chart title
            height: Chart height
            width: Chart width
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        metrics = list(list(metrics_data.values())[0].keys())
        colors = self.color_palette[:len(metrics_data)]
        
        for i, (method, scores) in enumerate(metrics_data.items()):
            values = [scores[metric] for metric in metrics]
            values.append(values[0])  # Close the radar chart
            metrics_extended = metrics + [metrics[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_extended,
                fill='toself',
                name=method,
                line_color=colors[i]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=dict(text=title, x=0.5),
            height=height,
            width=width,
            template=self.theme
        )
        
        return fig
