"""
Enhanced metrics module for comprehensive model evaluation.
Provides per-class metrics, macro/micro F1, and other evaluation tools.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class EnhancedMetrics:
    """Comprehensive metrics evaluation for multi-class classification."""
    
    def __init__(self, y_true, y_pred, y_pred_proba=None, class_names=None):
        """
        Initialize metrics evaluator.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            class_names: List of class names (optional)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.class_names = class_names
        self.num_classes = len(np.unique(y_true))
        
    def get_per_class_metrics(self):
        """Calculate per-class metrics."""
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, average=None
        )
        
        metrics_df = pd.DataFrame({
            'Class': self.class_names or range(self.num_classes),
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        return metrics_df
    
    def get_macro_micro_f1(self):
        """Calculate macro and micro F1 scores."""
        macro_f1 = f1_score(self.y_true, self.y_pred, average='macro')
        micro_f1 = f1_score(self.y_true, self.y_pred, average='micro')
        weighted_f1 = f1_score(self.y_true, self.y_pred, average='weighted')
        
        return {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'weighted_f1': weighted_f1
        }
    
    def get_classification_report(self):
        """Get detailed classification report."""
        return classification_report(
            self.y_true, self.y_pred,
            target_names=self.class_names or None,
            digits=4,
            output_dict=True
        )
    
    def plot_per_class_metrics(self):
        """Plot per-class metrics comparison."""
        df = self.get_per_class_metrics()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['Class'],
            y=df['Precision'],
            name='Precision',
            marker_color='#636EFA'
        ))
        fig.add_trace(go.Bar(
            x=df['Class'],
            y=df['Recall'],
            name='Recall',
            marker_color='#EF553B'
        ))
        fig.add_trace(go.Bar(
            x=df['Class'],
            y=df['F1-Score'],
            name='F1-Score',
            marker_color='#00CC96'
        ))
        
        fig.update_layout(
            title='Per-Class Metrics Comparison',
            barmode='group',
            xaxis_title='Class',
            yaxis_title='Score',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_confusion_matrix_enhanced(self):
        """Plot confusion matrix with annotations."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=self.class_names or [f'Class {i}' for i in range(self.num_classes)],
            y=self.class_names or [f'Class {i}' for i in range(self.num_classes)],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 12},
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='True',
            height=600
        )
        
        return fig
    
    def plot_f1_comparison(self):
        """Plot F1 scores comparison."""
        f1_scores = self.get_macro_micro_f1()
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(f1_scores.keys()),
                y=list(f1_scores.values()),
                marker_color=['#636EFA', '#EF553B', '#00CC96'],
                text=[f'{v:.4f}' for v in f1_scores.values()],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='F1 Score Comparison (Macro vs Micro vs Weighted)',
            yaxis_title='F1 Score',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def get_summary_dict(self):
        """Get comprehensive summary dictionary."""
        per_class = self.get_per_class_metrics()
        f1_dict = self.get_macro_micro_f1()
        
        return {
            'per_class_metrics': per_class.to_dict(),
            'macro_f1': f1_dict['macro_f1'],
            'micro_f1': f1_dict['micro_f1'],
            'weighted_f1': f1_dict['weighted_f1'],
            'accuracy': np.mean(self.y_true == self.y_pred)
        }
