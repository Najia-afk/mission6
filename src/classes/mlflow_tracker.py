"""
MLflow experiment tracking integration.
Automates logging and tracking of experiments.
"""

import mlflow
import mlflow.keras
import mlflow.sklearn
import json
from datetime import datetime
import os


class MLflowTracker:
    """MLflow integration for experiment tracking."""
    
    def __init__(self, experiment_name, tracking_uri=None):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking server URI (optional)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or 'file:./mlruns'
        
        # Set tracking server
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)
        print(f"✅ MLflow experiment: {experiment_name}")
    
    def start_run(self, run_name=None):
        """Start a new MLflow run."""
        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id
        print(f"✅ Started MLflow run: {self.run_id}")
        return self.run
    
    def end_run(self):
        """End current MLflow run."""
        mlflow.end_run()
        print(f"✅ Ended MLflow run: {self.run_id}")
    
    def log_params(self, params):
        """Log parameters."""
        mlflow.log_params(params)
        print(f"✅ Logged {len(params)} parameters")
    
    def log_metrics(self, metrics, step=None):
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model, artifact_path='model'):
        """Log Keras model."""
        mlflow.keras.log_model(model, artifact_path=artifact_path)
        print(f"✅ Logged model to {artifact_path}")
    
    def log_artifacts(self, local_dir, artifact_path=None):
        """Log artifacts (files/directories)."""
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
        print(f"✅ Logged artifacts from {local_dir}")
    
    def log_training_history(self, history):
        """Log training history as metrics."""
        if hasattr(history, 'history'):
            history = history.history
        
        for epoch, losses in enumerate(zip(*history.values())):
            metrics_dict = {key: val for key, val in zip(history.keys(), losses)}
            self.log_metrics(metrics_dict, step=epoch)
    
    def log_config(self, config_dict):
        """Log configuration as artifact."""
        config_file = '/tmp/config.json'
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        mlflow.log_artifact(config_file, artifact_path='configs')
        os.remove(config_file)
        print("✅ Logged configuration")
    
    def log_evaluation_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Log evaluation metrics from predictions."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
        }
        
        mlflow.log_metrics(metrics)
        print("✅ Logged evaluation metrics")
        
        return metrics
    
    def get_best_run(self, metric_name='accuracy', maximize=True):
        """Get best run by metric."""
        experiment = mlflow.get_experiment(self.experiment_id)
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        
        if runs.empty:
            return None
        
        if maximize:
            best_idx = runs[f'metrics.{metric_name}'].idxmax()
        else:
            best_idx = runs[f'metrics.{metric_name}'].idxmin()
        
        return runs.iloc[best_idx]
    
    def compare_runs(self, metrics_to_compare=None):
        """Compare all runs in experiment."""
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        
        if metrics_to_compare is None:
            metrics_to_compare = ['accuracy', 'f1_macro', 'f1_micro']
        
        comparison_cols = [f'metrics.{m}' for m in metrics_to_compare if f'metrics.{m}' in runs.columns]
        
        return runs[['run_id', 'start_time'] + comparison_cols].dropna()
    
    def get_run_summary(self):
        """Get summary of current run."""
        run = mlflow.get_run(self.run_id)
        
        summary = {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
            'status': run.info.status,
            'params': run.data.params,
            'metrics': run.data.metrics
        }
        
        return summary
    
    def print_summary(self):
        """Print run summary."""
        summary = self.get_run_summary()
        print(f"""
        ═══════════════════════════════════════════════════
        MLflow Run Summary
        ═══════════════════════════════════════════════════
        Run ID: {summary['run_id']}
        Status: {summary['status']}
        Start Time: {summary['start_time']}
        
        Parameters:
        {json.dumps(summary['params'], indent=2)}
        
        Metrics:
        {json.dumps(summary['metrics'], indent=2)}
        """)
