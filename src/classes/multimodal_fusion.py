"""
Multimodal Fusion Class for combining text and image features
Handles feature-level and decision-level fusion strategies
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


class MultimodalFusion:
    """
    Comprehensive multimodal fusion for text and image features
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the multimodal fusion system
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.fusion_strategies = {}
        self.fusion_results = {}
        self.ensemble_strategies = {}
        self.ensemble_results = {}
        
    def prepare_features(self, text_features, image_features_deep, image_features_basic=None):
        """
        Prepare and align features for fusion
        
        Args:
            text_features (np.ndarray): Text feature matrix
            image_features_deep (np.ndarray): Deep image features
            image_features_basic (np.ndarray): Basic image features (optional)
            
        Returns:
            tuple: Aligned and normalized features
        """
        print("Preparing features for multimodal fusion...")
        
        # Determine minimum sample size for alignment
        min_samples = min(len(text_features), len(image_features_deep))
        if image_features_basic is not None:
            min_samples = min(min_samples, len(image_features_basic))
        
        print(f"Aligning to {min_samples} samples...")
        
        # Align sample sizes
        text_aligned = text_features[:min_samples]
        image_deep_aligned = image_features_deep[:min_samples]
        image_basic_aligned = image_features_basic[:min_samples] if image_features_basic is not None else None
        
        # Normalize features
        scaler_text = StandardScaler()
        scaler_image_deep = StandardScaler()
        
        text_normalized = scaler_text.fit_transform(text_aligned)
        image_deep_normalized = scaler_image_deep.fit_transform(image_deep_aligned)
        
        print(f"Text features normalized: {text_normalized.shape}")
        print(f"Deep image features normalized: {image_deep_normalized.shape}")
        
        if image_basic_aligned is not None:
            scaler_image_basic = StandardScaler()
            image_basic_normalized = scaler_image_basic.fit_transform(image_basic_aligned)
            print(f"Basic image features normalized: {image_basic_normalized.shape}")
            return text_normalized, image_deep_normalized, image_basic_normalized, min_samples
        
        return text_normalized, image_deep_normalized, None, min_samples
    
    def create_fusion_strategies(self, text_features, image_features_deep, image_features_basic=None):
        """
        Create different feature fusion strategies
        
        Args:
            text_features (np.ndarray): Normalized text features
            image_features_deep (np.ndarray): Normalized deep image features
            image_features_basic (np.ndarray): Normalized basic image features (optional)
            
        Returns:
            dict: Dictionary of fusion strategies
        """
        print("Creating fusion combinations:")
        
        # Strategy 1: Text + Deep Image Features
        fusion_text_deep = np.concatenate([text_features, image_features_deep], axis=1)
        self.fusion_strategies['Text_Deep'] = fusion_text_deep
        
        # Strategy 2: Text + Basic Image Features (if available)
        if image_features_basic is not None:
            fusion_text_basic = np.concatenate([text_features, image_features_basic], axis=1)
            self.fusion_strategies['Text_Basic'] = fusion_text_basic
            
            # Strategy 3: Text + Deep + Basic Image Features
            fusion_all = np.concatenate([text_features, image_features_deep, image_features_basic], axis=1)
            self.fusion_strategies['Text_Deep_Basic'] = fusion_all
        
        # Strategy 4: Weighted fusion (example with text emphasis)
        text_weight = 0.3
        image_weight = 0.7
        
        weighted_text = text_features * text_weight
        weighted_image_deep = image_features_deep * image_weight
        
        fusion_weighted = np.concatenate([weighted_text, weighted_image_deep], axis=1)
        self.fusion_strategies['Weighted_Text_Deep'] = fusion_weighted
        
        print("   Fusion strategies created:")
        for strategy, features in self.fusion_strategies.items():
            print(f"   - {strategy}: {features.shape}")
        
        return self.fusion_strategies
    
    def analyze_fusion_strategies(self, optimal_clusters=3):
        """
        Analyze each fusion strategy with clustering
        
        Args:
            optimal_clusters (int): Number of clusters to use
            
        Returns:
            dict: Analysis results for each strategy
        """
        print("=== MULTIMODAL CLUSTERING ANALYSIS ===")
        
        for strategy_name, fused_features in self.fusion_strategies.items():
            print(f"\nAnalyzing {strategy_name}:")
            
            # Apply PCA for dimensionality reduction
            n_components = min(min(fused_features.shape) - 1, 50)
            pca_fusion = PCA(n_components=n_components)
            fused_features_pca = pca_fusion.fit_transform(fused_features)
            
            # Clustering
            kmeans_fusion = KMeans(n_clusters=optimal_clusters, random_state=self.random_state, n_init=10)
            cluster_labels_fusion = kmeans_fusion.fit_predict(fused_features_pca)
            
            # Calculate metrics
            silhouette_fusion = silhouette_score(fused_features_pca, cluster_labels_fusion)
            
            # t-SNE for visualization (on reduced features)
            perplexity = min(15, len(fused_features_pca) // 4)
            tsne_fusion = TSNE(
                n_components=2, 
                perplexity=perplexity, 
                random_state=self.random_state, 
                n_iter=1000
            )
            fused_tsne = tsne_fusion.fit_transform(fused_features_pca)
            
            # Store results
            self.fusion_results[strategy_name] = {
                'features_shape': fused_features.shape,
                'pca_shape': fused_features_pca.shape,
                'silhouette_score': silhouette_fusion,
                'cluster_labels': cluster_labels_fusion,
                'tsne_coords': fused_tsne,
                'variance_explained': pca_fusion.explained_variance_ratio_.sum(),
                'n_components': n_components,
                'pca_model': pca_fusion,
                'kmeans_model': kmeans_fusion
            }
            
            print(f"   Original shape: {fused_features.shape}")
            print(f"   PCA shape: {fused_features_pca.shape}")
            print(f"   Silhouette score: {silhouette_fusion:.3f}")
            print(f"   Variance explained: {pca_fusion.explained_variance_ratio_.sum():.3f}")
        
        return self.fusion_results
    
    def create_performance_comparison(self, baseline_scores=None):
        """
        Create performance comparison between strategies
        
        Args:
            baseline_scores (dict): Baseline scores for single modalities
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        # Create comparison DataFrame
        multimodal_comparison = []
        for strategy, results in self.fusion_results.items():
            multimodal_comparison.append({
                'Strategy': strategy,
                'Total_Dimensions': results['features_shape'][1],
                'PCA_Dimensions': results['pca_shape'][1],
                'Silhouette_Score': results['silhouette_score'],
                'Variance_Explained': results['variance_explained']
            })
        
        multimodal_df = pd.DataFrame(multimodal_comparison)
        
        # Add baseline comparisons if provided
        if baseline_scores:
            baseline_comparisons = []
            for name, score_info in baseline_scores.items():
                baseline_comparisons.append({
                    'Strategy': name,
                    'Total_Dimensions': score_info.get('dimensions', 0),
                    'PCA_Dimensions': score_info.get('dimensions', 0),
                    'Silhouette_Score': score_info.get('score', 0),
                    'Variance_Explained': score_info.get('variance', 1.0)
                })
            
            baseline_df = pd.DataFrame(baseline_comparisons)
            complete_comparison_df = pd.concat([baseline_df, multimodal_df], ignore_index=True)
        else:
            complete_comparison_df = multimodal_df
        
        return complete_comparison_df
    
    def implement_ensemble_fusion(self, text_features, image_features_deep, image_features_basic=None, optimal_clusters=3):
        """
        Implement ensemble decision fusion strategies
        
        Args:
            text_features (np.ndarray): Text features
            image_features_deep (np.ndarray): Deep image features
            image_features_basic (np.ndarray): Basic image features (optional)
            optimal_clusters (int): Number of clusters
            
        Returns:
            dict: Ensemble results
        """
        print("=== ENSEMBLE DECISION FUSION ===")
        print("Creating ensemble decision fusion framework...")
        
        # Get individual modality predictions (cluster assignments)
        text_clusters = KMeans(n_clusters=optimal_clusters, random_state=self.random_state).fit_predict(text_features)
        image_deep_clusters = KMeans(n_clusters=optimal_clusters, random_state=self.random_state).fit_predict(image_features_deep)
        
        if image_features_basic is not None:
            image_basic_clusters = KMeans(n_clusters=optimal_clusters, random_state=self.random_state).fit_predict(image_features_basic)
        
        # Decision fusion strategies
        print("\n1. Implementing Decision Fusion Strategies:")
        
        # Strategy 1: Text + Deep Image
        predictions_text_deep = [text_clusters, image_deep_clusters]
        
        self.ensemble_strategies['Majority_Text_Deep'] = self._majority_voting(predictions_text_deep)
        self.ensemble_strategies['Weighted_Text_Deep'] = self._weighted_voting(predictions_text_deep, [0.3, 0.7])
        
        # Strategy 2: All modalities (if basic features available)
        if image_features_basic is not None:
            predictions_all = [text_clusters, image_deep_clusters, image_basic_clusters]
            
            self.ensemble_strategies['Majority_All'] = self._majority_voting(predictions_all)
            self.ensemble_strategies['Weighted_All'] = self._weighted_voting(predictions_all, [0.2, 0.6, 0.2])
        
        print(f"   Created {len(self.ensemble_strategies)} ensemble strategies")
        
        # Evaluate ensemble strategies
        print("\n2. Evaluating Ensemble Performance:")
        
        # Use the best fusion features for evaluation
        best_fusion_strategy = max(self.fusion_results.keys(), 
                                 key=lambda x: self.fusion_results[x]['silhouette_score'])
        best_fusion_features = self.fusion_results[best_fusion_strategy]['tsne_coords']
        
        for strategy_name, ensemble_pred in self.ensemble_strategies.items():
            # Calculate silhouette score
            if len(np.unique(ensemble_pred)) > 1:
                ensemble_silhouette = silhouette_score(best_fusion_features, ensemble_pred)
            else:
                ensemble_silhouette = 0.0
            
            self.ensemble_results[strategy_name] = {
                'silhouette_score': ensemble_silhouette,
                'n_clusters': len(np.unique(ensemble_pred)),
                'predictions': ensemble_pred
            }
            
            print(f"   {strategy_name}: {ensemble_silhouette:.3f} (clusters: {len(np.unique(ensemble_pred))})")
        
        return self.ensemble_results
    
    def _majority_voting(self, predictions_list):
        """Simple majority voting fusion"""
        ensemble_predictions = []
        for i in range(len(predictions_list[0])):
            votes = [pred[i] for pred in predictions_list]
            ensemble_predictions.append(max(set(votes), key=votes.count))
        return np.array(ensemble_predictions)
    
    def _weighted_voting(self, predictions_list, weights):
        """Weighted voting based on given weights"""
        weights = np.array(weights) / np.sum(weights)  # Normalize weights
        ensemble_predictions = []
        
        for i in range(len(predictions_list[0])):
            weighted_votes = {}
            for j, pred in enumerate(predictions_list):
                vote = pred[i]
                if vote not in weighted_votes:
                    weighted_votes[vote] = 0
                weighted_votes[vote] += weights[j]
            ensemble_predictions.append(max(weighted_votes, key=weighted_votes.get))
        
        return np.array(ensemble_predictions)
    
    def get_best_approaches(self):
        """
        Get ranking of all fusion approaches
        
        Returns:
            dict: Sorted approaches by performance
        """
        all_approaches = {}
        
        # Add feature fusion results
        for strategy, results in self.fusion_results.items():
            all_approaches[f"Feature_{strategy}"] = results['silhouette_score']
        
        # Add ensemble fusion results
        for strategy, results in self.ensemble_results.items():
            all_approaches[f"Ensemble_{strategy}"] = results['silhouette_score']
        
        # Sort by performance
        sorted_approaches = dict(sorted(all_approaches.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_approaches
    
    def create_multimodal_dashboard(self, comparison_df, best_strategy_info=None):
        """
        Create comprehensive multimodal dashboard
        
        Args:
            comparison_df (pd.DataFrame): Performance comparison dataframe
            best_strategy_info (dict): Information about best strategy
            
        Returns:
            plotly.graph_objects.Figure: Dashboard figure
        """
        print("=== CREATING MULTIMODAL DASHBOARD ===")
        
        # Create comprehensive multimodal dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Strategy Performance Comparison', 'Best Strategy t-SNE Visualization',
                'Dimensionality Analysis', 'Improvement Analysis',
                'Fusion Strategy Details', 'Multimodal vs Unimodal Performance'
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "bar"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Strategy Performance Comparison (Bar Chart)
        strategies = comparison_df['Strategy']
        scores = comparison_df['Silhouette_Score']
        colors = ['#ff6b6b' if 'Only' in s else '#4ecdc4' for s in strategies]
        
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=scores,
                marker_color=colors,
                text=[f'{s:.3f}' for s in scores],
                textposition='auto',
                name='Silhouette Scores'
            ),
            row=1, col=1
        )
        
        # 2. Best Strategy t-SNE Visualization
        if best_strategy_info and 'tsne_coords' in best_strategy_info:
            best_tsne = best_strategy_info['tsne_coords']
            best_clusters = best_strategy_info['cluster_labels']
            
            # Create scatter for each cluster
            for cluster_id in np.unique(best_clusters):
                mask = best_clusters == cluster_id
                fig.add_trace(
                    go.Scatter(
                        x=best_tsne[mask, 0],
                        y=best_tsne[mask, 1],
                        mode='markers',
                        marker=dict(size=8, opacity=0.7),
                        name=f'Cluster {cluster_id}',
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # 3. Dimensionality Analysis (Scatter Plot)
        fig.add_trace(
            go.Scatter(
                x=comparison_df['Total_Dimensions'],
                y=comparison_df['Silhouette_Score'],
                mode='markers+text',
                marker=dict(
                    size=comparison_df['PCA_Dimensions']*2,
                    color=comparison_df['Silhouette_Score'],
                    colorscale='viridis',
                    showscale=False,
                    opacity=0.7
                ),
                text=comparison_df['Strategy'],
                textposition='top center',
                name='Dim vs Performance'
            ),
            row=2, col=1
        )
        
        # 4. Improvement Analysis
        unimodal_strategies = comparison_df[comparison_df['Strategy'].str.contains('Only', na=False)]
        multimodal_strategies = comparison_df[~comparison_df['Strategy'].str.contains('Only', na=False)]
        
        if len(unimodal_strategies) > 0 and len(multimodal_strategies) > 0:
            fig.add_trace(
                go.Bar(
                    x=['Unimodal (Best)', 'Multimodal (Best)', 'Multimodal (Average)'],
                    y=[
                        unimodal_strategies['Silhouette_Score'].max(),
                        multimodal_strategies['Silhouette_Score'].max(),
                        multimodal_strategies['Silhouette_Score'].mean()
                    ],
                    marker_color=['#ff9999', '#66b3ff', '#99ff99'],
                    text=[
                        f"{unimodal_strategies['Silhouette_Score'].max():.3f}",
                        f"{multimodal_strategies['Silhouette_Score'].max():.3f}",
                        f"{multimodal_strategies['Silhouette_Score'].mean():.3f}"
                    ],
                    textposition='auto',
                    name='Performance Comparison'
                ),
                row=2, col=2
            )
        
        # 5. Fusion Strategy Details Table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Strategy', 'Total Dims', 'PCA Dims', 'Silhouette', 'Variance Exp.'],
                    fill_color='lightblue',
                    align='center',
                    font=dict(size=10)
                ),
                cells=dict(
                    values=[
                        comparison_df['Strategy'],
                        comparison_df['Total_Dimensions'],
                        comparison_df['PCA_Dimensions'],
                        [f"{s:.3f}" for s in comparison_df['Silhouette_Score']],
                        [f"{v:.3f}" for v in comparison_df['Variance_Explained']]
                    ],
                    fill_color='white',
                    align='center',
                    font=dict(size=9)
                )
            ),
            row=3, col=1
        )
        
        # 6. Modality Contribution Analysis
        if len(unimodal_strategies) > 0:
            text_performance = unimodal_strategies[unimodal_strategies['Strategy'].str.contains('Text', na=False)]['Silhouette_Score']
            text_score = text_performance.iloc[0] if len(text_performance) > 0 else 0.25
            
            modality_performance = {
                'Text Only': text_score,
                'Image Only (Best)': unimodal_strategies['Silhouette_Score'].max(),
                'Multimodal (Best)': multimodal_strategies['Silhouette_Score'].max() if len(multimodal_strategies) > 0 else 0
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(modality_performance.keys()),
                    y=list(modality_performance.values()),
                    marker_color=['#ff7f7f', '#7f7fff', '#7fff7f'],
                    text=[f'{v:.3f}' for v in modality_performance.values()],
                    textposition='auto',
                    name='Modality Contribution'
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Comprehensive Multimodal Fusion Analysis Dashboard',
            template='plotly_white',
            showlegend=False,
            width=1200,
            height=1000,
            font=dict(size=10)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Fusion Strategy", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=1)
        fig.update_xaxes(title_text="t-SNE 1", row=1, col=2)
        fig.update_yaxes(title_text="t-SNE 2", row=1, col=2)
        fig.update_xaxes(title_text="Total Dimensions", row=2, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=2, col=1)
        fig.update_xaxes(title_text="Approach Type", row=2, col=2)
        fig.update_yaxes(title_text="Performance", row=2, col=2)
        fig.update_xaxes(title_text="Modality", row=3, col=2)
        fig.update_yaxes(title_text="Silhouette Score", row=3, col=2)
        
        return fig
    
    def get_summary_report(self):
        """
        Generate comprehensive summary report
        
        Returns:
            dict: Summary report
        """
        # Get best approaches
        sorted_approaches = self.get_best_approaches()
        best_approach = list(sorted_approaches.keys())[0] if sorted_approaches else "None"
        best_score = list(sorted_approaches.values())[0] if sorted_approaches else 0.0
        
        summary = {
            'fusion_strategies_tested': len(self.fusion_strategies),
            'ensemble_strategies_tested': len(self.ensemble_strategies),
            'total_approaches': len(sorted_approaches),
            'best_approach': best_approach,
            'best_score': best_score,
            'feature_fusion_count': len(self.fusion_results),
            'ensemble_fusion_count': len(self.ensemble_results),
            'approach_ranking': sorted_approaches
        }
        
        # Add strategy-specific details
        summary['fusion_details'] = {}
        for strategy, results in self.fusion_results.items():
            summary['fusion_details'][strategy] = {
                'silhouette_score': results['silhouette_score'],
                'total_dimensions': results['features_shape'][1],
                'pca_dimensions': results['pca_shape'][1],
                'variance_explained': results['variance_explained']
            }
        
        summary['ensemble_details'] = {}
        for strategy, results in self.ensemble_results.items():
            summary['ensemble_details'][strategy] = {
                'silhouette_score': results['silhouette_score'],
                'n_clusters': results['n_clusters']
            }
        
        return summary


class MultimodalFusionClassifier:
    """
    Late fusion classifier for combining text and image embeddings.
    Concatenates embeddings from both modalities and trains a fusion head.
    """
    
    def __init__(self, text_feature_dim=512, image_feature_dim=4096, num_classes=7, fusion_method='late'):
        """
        Initialize the fusion classifier.
        
        Args:
            text_feature_dim (int): Dimension of text embeddings
            image_feature_dim (int): Dimension of image embeddings
            num_classes (int): Number of classification classes
            fusion_method (str): 'late' for concatenation
        """
        self.text_feature_dim = text_feature_dim
        self.image_feature_dim = image_feature_dim
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32, verbose=1):
        """
        Train the fusion classifier.
        
        Args:
            X_train (np.ndarray): Training features (concatenated text + image)
            y_train (np.ndarray): Training labels (one-hot encoded)
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            verbose (int): Verbosity level
            
        Returns:
            dict: Training metrics
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train MLP classifier
        self.model = MLPClassifier(
            hidden_layer_sizes=(512, 256),
            max_iter=epochs,
            batch_size=batch_size,
            learning_rate_init=0.001,
            random_state=42,
            verbose=verbose if verbose > 0 else 0
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)
        
        metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'num_classes': self.num_classes
        }
        
        if verbose > 0:
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
        
        return metrics
