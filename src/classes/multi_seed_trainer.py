"""
Multi-seed training framework for reproducibility.
Trains models with multiple random seeds and aggregates results.
"""

import numpy as np
import tensorflow as tf
import random
import pandas as pd
from datetime import datetime


class MultiSeedTrainer:
    """Framework for multi-seed training with aggregated metrics."""
    
    def __init__(self, model_builder, num_seeds=3):
        """
        Initialize multi-seed trainer.
        
        Args:
            model_builder: Function that builds and returns a model
            num_seeds: Number of seeds to run
        """
        self.model_builder = model_builder
        self.num_seeds = num_seeds
        self.results = []
        self.seed_models = []
    
    def set_seed(self, seed):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
    
    def train_seed(self, seed, X_train, y_train, X_val=None, y_val=None,
                   epochs=10, batch_size=32, **kwargs):
        """
        Train a single model with given seed.
        
        Args:
            seed: Random seed
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            **kwargs: Additional arguments for model training
        """
        self.set_seed(seed)
        
        model = self.model_builder()
        
        val_data = None
        if X_val is not None:
            val_data = (X_val, y_val)
        
        history = model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            **kwargs
        )
        
        return model, history
    
    def run_all_seeds(self, X_train, y_train, X_val=None, y_val=None,
                      X_test=None, y_test=None, **kwargs):
        """
        Run training for all seeds.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            X_test: Test data
            y_test: Test labels
            **kwargs: Additional arguments for model training
        """
        
        print(f"\n🔄 Starting multi-seed training ({self.num_seeds} seeds)...")
        
        for seed_idx in range(self.num_seeds):
            seed = 42 + seed_idx
            print(f"\n  Seed {seed_idx+1}/{self.num_seeds} (seed={seed})...")
            
            model, history = self.train_seed(
                seed, X_train, y_train,
                X_val=X_val, y_val=y_val,
                **kwargs
            )
            
            self.seed_models.append(model)
            
            # Evaluate
            if X_test is not None:
                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
                print(f"  Test Accuracy: {test_acc:.4f}")
                
                self.results.append({
                    'seed': seed,
                    'test_accuracy': test_acc,
                    'test_loss': test_loss,
                    'val_accuracy': history.history.get('val_accuracy', [-1])[-1],
                    'train_accuracy': history.history.get('accuracy', [-1])[-1]
                })
        
        return self._aggregate_results()
    
    def _aggregate_results(self):
        """Aggregate results across seeds."""
        results_df = pd.DataFrame(self.results)
        
        aggregated = {
            'mean_test_accuracy': results_df['test_accuracy'].mean(),
            'std_test_accuracy': results_df['test_accuracy'].std(),
            'mean_val_accuracy': results_df['val_accuracy'].mean(),
            'std_val_accuracy': results_df['val_accuracy'].std(),
            'min_test_accuracy': results_df['test_accuracy'].min(),
            'max_test_accuracy': results_df['test_accuracy'].max(),
            'all_results': results_df
        }
        
        return aggregated
    
    def ensemble_predict(self, X):
        """Make ensemble predictions using all trained models."""
        predictions = []
        for model in self.seed_models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def get_summary(self):
        """Get training summary."""
        if not self.results:
            return "No training results yet."
        
        results_df = pd.DataFrame(self.results)
        
        summary = f"""
        ═══════════════════════════════════════════════════
        Multi-Seed Training Summary ({self.num_seeds} seeds)
        ═══════════════════════════════════════════════════
        
        Test Accuracy:
          Mean:  {results_df['test_accuracy'].mean():.4f}
          Std:   {results_df['test_accuracy'].std():.4f}
          Min:   {results_df['test_accuracy'].min():.4f}
          Max:   {results_df['test_accuracy'].max():.4f}
        
        Validation Accuracy:
          Mean:  {results_df['val_accuracy'].mean():.4f}
          Std:   {results_df['val_accuracy'].std():.4f}
        
        Per-Seed Results:
        {results_df.to_string()}
        """
        
        return summary
    
    def print_summary(self):
        """Print summary to console."""
        print(self.get_summary())
