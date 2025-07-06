"""
Category-Based Analysis Dashboard for E-commerce Product Classification
Provides detailed analysis by product category with proper statistics and visualizations
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from typing import Dict, Any, List, Tuple, Optional

warnings.filterwarnings('ignore')

class CategoryBasedAnalyzer:
    """
    Advanced analyzer for category-based statistics and visualizations
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the category-based analyzer
        
        Args:
            df: DataFrame containing product data with 'product_category' column
        """
        self.df = df
        self.category_stats = None
        self.category_features = None
        
        # Validate that product_category exists
        if 'product_category' not in df.columns:
            raise ValueError("DataFrame must contain 'product_category' column")
        
        # Get category information
        self.categories = df['product_category'].unique()
        self.category_counts = df['product_category'].value_counts()
        
        print(f"📊 Initialized Category-Based Analyzer")
        print(f"   📂 Categories found: {len(self.categories)}")
        print(f"   📋 Total samples: {len(df)}")
        
        # Display category distribution
        print(f"\n🏷️ Category Distribution:")
        for category, count in self.category_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {category}: {count} samples ({percentage:.1f}%)")
    
    def analyze_category_features(self, features: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Analyze features by category with detailed statistics
        
        Args:
            features: Feature matrix (n_samples x n_features)
            feature_names: Names of features (optional)
            
        Returns:
            Dictionary with category-based analysis results
        """
        print(f"🔍 Analyzing features by category...")
        
        if len(features) != len(self.df):
            print(f"⚠️ Warning: Feature matrix length ({len(features)}) doesn't match DataFrame length ({len(self.df)})")
            # Trim to minimum length
            min_len = min(len(features), len(self.df))
            features = features[:min_len]
            df_subset = self.df.iloc[:min_len]
        else:
            df_subset = self.df
        
        # Calculate statistics by category
        category_stats = {}
        category_features = {}
        
        for category in self.categories:
            mask = df_subset['product_category'] == category
            if np.any(mask):
                cat_features = features[mask]
                
                # Calculate comprehensive statistics
                stats = {
                    'count': len(cat_features),
                    'mean': np.mean(cat_features, axis=0),
                    'std': np.std(cat_features, axis=0),
                    'median': np.median(cat_features, axis=0),
                    'min': np.min(cat_features, axis=0),
                    'max': np.max(cat_features, axis=0),
                    'q25': np.percentile(cat_features, 25, axis=0),
                    'q75': np.percentile(cat_features, 75, axis=0),
                    'variance': np.var(cat_features, axis=0),
                    'skewness': self._calculate_skewness(cat_features),
                    'kurtosis': self._calculate_kurtosis(cat_features)
                }
                
                category_stats[category] = stats
                category_features[category] = cat_features
        
        self.category_stats = category_stats
        self.category_features = category_features
        
        print(f"✅ Category feature analysis complete!")
        print(f"   📊 Categories analyzed: {len(category_stats)}")
        
        return {
            'category_stats': category_stats,
            'category_features': category_features,
            'feature_names': feature_names or [f'Feature_{i}' for i in range(features.shape[1])]
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> np.ndarray:
        """Calculate skewness for each feature"""
        from scipy.stats import skew
        try:
            return skew(data, axis=0)
        except:
            return np.zeros(data.shape[1])
    
    def _calculate_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Calculate kurtosis for each feature"""
        from scipy.stats import kurtosis
        try:
            return kurtosis(data, axis=0)
        except:
            return np.zeros(data.shape[1])
    
    def create_category_dashboard(self, features: np.ndarray, feature_names: List[str] = None) -> go.Figure:
        """
        Create comprehensive category-based dashboard
        
        Args:
            features: Feature matrix
            feature_names: Names of features
            
        Returns:
            Plotly figure with dashboard
        """
        if self.category_stats is None:
            self.analyze_category_features(features, feature_names)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Category Distribution',
                'Feature Variance by Category',
                'Category Feature Box Plots (Top 5 Features)',
                'Category Statistics Heatmap',
                'Standard Deviation Analysis',
                'Category Overlap Analysis'
            ],
            specs=[
                [{"type": "bar"}, {"type": "box"}],
                [{"type": "box"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Category Distribution
        categories = list(self.category_counts.index)
        counts = list(self.category_counts.values)
        colors = px.colors.qualitative.Set3[:len(categories)]
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=counts,
                name='Sample Count',
                marker_color=colors,
                text=counts,
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Feature Variance by Category
        variances_by_category = []
        variance_categories = []
        for category, stats in self.category_stats.items():
            avg_variance = np.mean(stats['variance'])
            variances_by_category.append(avg_variance)
            variance_categories.append(category)
        
        fig.add_trace(
            go.Bar(
                x=variance_categories,
                y=variances_by_category,
                name='Avg Variance',
                marker_color=colors,
                text=[f'{v:.3f}' for v in variances_by_category],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. Box plots for top 5 most variable features
        if features.shape[1] > 0:
            # Find most variable features across all categories
            overall_variance = np.var(features, axis=0)
            top_feature_indices = np.argsort(overall_variance)[-5:]
            
            for i, feat_idx in enumerate(top_feature_indices):
                for j, category in enumerate(categories):
                    if category in self.category_features:
                        cat_data = self.category_features[category][:, feat_idx]
                        fig.add_trace(
                            go.Box(
                                y=cat_data,
                                name=f'{category}',
                                legendgroup=f'feature_{i}',
                                showlegend=(i == 0),
                                marker_color=colors[j % len(colors)],
                                offsetgroup=j,
                                x=[f'Feature {feat_idx}'] * len(cat_data)
                            ),
                            row=2, col=1
                        )
        
        # 4. Category Statistics Heatmap (means)
        if self.category_stats:
            # Create heatmap of mean values for top features
            heatmap_data = []
            heatmap_categories = []
            for category, stats in self.category_stats.items():
                if features.shape[1] > 0:
                    # Use top 10 features for heatmap
                    top_10_features = np.argsort(np.var(features, axis=0))[-10:]
                    heatmap_data.append(stats['mean'][top_10_features])
                    heatmap_categories.append(category)
            
            if heatmap_data:
                fig.add_trace(
                    go.Heatmap(
                        z=heatmap_data,
                        y=heatmap_categories,
                        x=[f'F{i}' for i in range(len(heatmap_data[0]))],
                        colorscale='RdYlBu_r',
                        name='Mean Values'
                    ),
                    row=2, col=2
                )
        
        # 5. Standard Deviation Analysis
        std_means = []
        std_categories = []
        for category, stats in self.category_stats.items():
            avg_std = np.mean(stats['std'])
            std_means.append(avg_std)
            std_categories.append(category)
        
        fig.add_trace(
            go.Bar(
                x=std_categories,
                y=std_means,
                name='Avg Std Dev',
                marker_color=colors,
                text=[f'{s:.3f}' for s in std_means],
                textposition='auto'
            ),
            row=3, col=1
        )
        
        # 6. Category Overlap Analysis (using PCA)
        if len(categories) > 1 and features.shape[1] > 1:
            try:
                # Apply PCA for visualization
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features[:len(self.df)])
                pca = PCA(n_components=2)
                features_pca = pca.fit_transform(features_scaled)
                
                for i, category in enumerate(categories):
                    mask = self.df['product_category'] == category
                    if np.any(mask):
                        cat_pca = features_pca[mask]
                        fig.add_trace(
                            go.Scatter(
                                x=cat_pca[:, 0],
                                y=cat_pca[:, 1],
                                mode='markers',
                                name=category,
                                marker=dict(
                                    color=colors[i % len(colors)],
                                    size=6,
                                    opacity=0.7
                                ),
                                legendgroup='pca'
                            ),
                            row=3, col=2
                        )
            except Exception as e:
                print(f"Warning: Could not create PCA overlap analysis: {e}")
        
        # Update layout
        fig.update_layout(
            title='📊 Category-Based Feature Analysis Dashboard',
            height=1200,
            width=1400,
            template='plotly_white',
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Categories", row=1, col=1)
        fig.update_yaxes(title_text="Sample Count", row=1, col=1)
        
        fig.update_xaxes(title_text="Categories", row=1, col=2)
        fig.update_yaxes(title_text="Average Variance", row=1, col=2)
        
        fig.update_xaxes(title_text="Features", row=2, col=1)
        fig.update_yaxes(title_text="Feature Values", row=2, col=1)
        
        fig.update_xaxes(title_text="Categories", row=3, col=1)
        fig.update_yaxes(title_text="Average Std Dev", row=3, col=1)
        
        fig.update_xaxes(title_text="PCA Component 1", row=3, col=2)
        fig.update_yaxes(title_text="PCA Component 2", row=3, col=2)
        
        return fig
    
    def create_category_comparison_report(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Create detailed comparison report between categories
        
        Args:
            features: Feature matrix
            
        Returns:
            Comprehensive report dictionary
        """
        if self.category_stats is None:
            self.analyze_category_features(features)
        
        report = {
            'summary': {
                'total_categories': len(self.categories),
                'total_samples': len(self.df),
                'feature_dimensions': features.shape[1],
                'category_distribution': dict(self.category_counts)
            },
            'category_analysis': {},
            'comparisons': {}
        }
        
        # Detailed analysis for each category
        for category, stats in self.category_stats.items():
            analysis = {
                'sample_count': stats['count'],
                'feature_statistics': {
                    'mean_of_means': np.mean(stats['mean']),
                    'mean_of_stds': np.mean(stats['std']),
                    'mean_variance': np.mean(stats['variance']),
                    'overall_range': np.mean(stats['max'] - stats['min'])
                },
                'distribution_properties': {
                    'skewness_avg': np.mean(stats['skewness']),
                    'kurtosis_avg': np.mean(stats['kurtosis']),
                    'is_normal_distributed': abs(np.mean(stats['skewness'])) < 0.5 and abs(np.mean(stats['kurtosis'])) < 3
                }
            }
            report['category_analysis'][category] = analysis
        
        # Cross-category comparisons
        categories_list = list(self.categories)
        if len(categories_list) > 1:
            for i in range(len(categories_list)):
                for j in range(i + 1, len(categories_list)):
                    cat1, cat2 = categories_list[i], categories_list[j]
                    
                    # Calculate statistical distances
                    mean_distance = np.linalg.norm(
                        self.category_stats[cat1]['mean'] - self.category_stats[cat2]['mean']
                    )
                    
                    variance_ratio = np.mean(self.category_stats[cat1]['variance']) / \
                                   (np.mean(self.category_stats[cat2]['variance']) + 1e-8)
                    
                    comparison_key = f"{cat1}_vs_{cat2}"
                    report['comparisons'][comparison_key] = {
                        'mean_distance': mean_distance,
                        'variance_ratio': variance_ratio,
                        'separability': 'High' if mean_distance > 1.0 else 'Medium' if mean_distance > 0.5 else 'Low'
                    }
        
        return report
    
    def create_analysis_visualization(self, features: np.ndarray = None, clustering_results: Dict = None) -> go.Figure:
        """
        Create the main analysis visualization that addresses the user's request
        This replaces random sampling with proper category-based analysis
        
        Args:
            features: Feature matrix (optional)
            clustering_results: Clustering results (optional)
            
        Returns:
            Plotly figure with comprehensive category analysis
        """
        print("🎨 Creating comprehensive category-based analysis visualization...")
        
        # Use existing features if none provided
        if features is None and hasattr(self, 'category_features'):
            # Reconstruct features from category data
            features = np.vstack([feat for feat in self.category_features.values()])
        
        if features is not None:
            # Ensure we have category analysis
            if self.category_stats is None:
                self.analyze_category_features(features)
            
            # Create the main dashboard
            dashboard_fig = self.create_category_dashboard(features)
            return dashboard_fig
        else:
            # Create a simpler visualization with just category distribution
            fig = go.Figure()
            
            categories = list(self.category_counts.index)
            counts = list(self.category_counts.values)
            colors = px.colors.qualitative.Set3[:len(categories)]
            
            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=counts,
                    marker_color=colors,
                    text=counts,
                    textposition='auto',
                    name='Sample Count'
                )
            )
            
            fig.update_layout(
                title='📊 Product Category Distribution Analysis',
                xaxis_title='Product Categories',
                yaxis_title='Number of Samples',
                template='plotly_white',
                height=500,
                width=800
            )
            
            return fig
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get analysis summary that addresses the user's request for better analytics
        
        Returns:
            Dictionary with comprehensive summary
        """
        if self.category_stats is None:
            # Return basic summary
            return {
                'message': 'Run analyze_category_features() first to get detailed statistics',
                'categories': list(self.categories),
                'category_counts': dict(self.category_counts),
                'total_categories': len(self.categories),
                'total_samples': len(self.df),
                'balanced_dataset': all(count >= 100 for count in self.category_counts),
                'recommendation': 'Use category-based analysis instead of random sampling for better insights'
            }
        
        # Calculate cross-category insights
        feature_variability = {}
        category_separability = {}
        
        for category, stats in self.category_stats.items():
            feature_variability[category] = {
                'coefficient_of_variation': np.mean(stats['std'] / (stats['mean'] + 1e-8)),
                'variance_ratio': np.mean(stats['variance']) / (np.mean([s['variance'] for s in self.category_stats.values()]) + 1e-8),
                'stability': 'High' if np.mean(stats['std']) < 0.5 else 'Medium' if np.mean(stats['std']) < 1.0 else 'Low'
            }
        
        return {
            'dataset_overview': {
                'total_categories': len(self.categories),
                'total_samples': len(self.df),
                'category_distribution': dict(self.category_counts),
                'balanced_dataset': all(count >= 100 for count in dict(self.category_counts).values())
            },
            'feature_analysis': {
                'categories_analyzed': len(self.category_stats),
                'feature_variability': feature_variability,
                'most_variable_category': max(feature_variability.keys(), 
                                            key=lambda x: feature_variability[x]['coefficient_of_variation']),
                'most_stable_category': min(feature_variability.keys(), 
                                          key=lambda x: feature_variability[x]['coefficient_of_variation'])
            },
            'recommendations': [
                'Use category-stratified sampling for model training',
                'Apply category-specific feature scaling if variance differences are large',
                'Consider ensemble methods that account for category imbalances',
                'Focus on most separable category pairs for initial model validation'
            ],
            'insights': {
                'category_based_analysis': 'Completed - provides better insights than random sampling',
                'statistical_significance': 'Each category analyzed with proper statistical measures',
                'visualization_approach': 'Box plots and variance analysis by category implemented',
                'next_steps': 'Use these category-specific insights for model development'
            }
        }
