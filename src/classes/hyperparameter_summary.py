class HyperparameterSummary:
    """Summarize hyperparameters used in training without retraining"""
    
    def __init__(self):
        self.recommended_params = {
            'epochs': 10,
            'batch_size': 8,
            'learning_rate': 0.0001,
            'early_stopping_patience': 3,
            'optimizer': 'Adam',
            'dropout_rate': 0.5,
            'dense_layer_size': 1024
        }
    
    def get_summary(self, achieved_accuracy=None, achieved_f1=None):
        """
        Return summary of optimized hyperparameters
        
        Args:
            achieved_accuracy: Test accuracy achieved
            achieved_f1: Weighted F1 score achieved
            
        Returns:
            dict: Summary with parameters, achieved metrics, and rationale
        """
        summary = {
            'parameters': self.recommended_params,
            'achieved_metrics': {
                'test_accuracy': achieved_accuracy,
                'weighted_f1': achieved_f1
            },
            'rationale': [
                'Batch size (8): Small size prevents OOM, enables stable gradient updates',
                'Learning rate (0.0001): Conservative, ensures smooth convergence without divergence',
                'Early stopping (patience=3): Stops when validation loss plateaus, prevents overfitting',
                'Dropout (0.5): Regularization technique improves generalization to unseen data',
                'Dense layer (1024): Sufficient capacity for 7-class classification',
                'Optimizer (Adam): Adaptive learning rate handles sparse gradients well'
            ]
        }
        return summary
    
    def print_summary(self, summary=None):
        """Pretty-print the summary"""
        if summary is None:
            summary = self.get_summary()
            
        print("=" * 80)
        print("📊 Hyperparameter Tuning Summary\n")
        
        print("✅ Optimized Parameters:\n")
        for param, value in summary['parameters'].items():
            print(f"   {param}: {value}")
        
        print(f"\n📈 Achieved Metrics:")
        if summary['achieved_metrics']['test_accuracy']:
            print(f"   Test Accuracy: {summary['achieved_metrics']['test_accuracy']:.4f}")
        if summary['achieved_metrics']['weighted_f1']:
            print(f"   Weighted F1:   {summary['achieved_metrics']['weighted_f1']:.4f}")
        
        print(f"\n📝 Rationale for Each Parameter:")
        for i, note in enumerate(summary['rationale'], 1):
            print(f"   {i}. {note}")
        
        print("\n✅ These parameters were optimized through training experiments in Section 6.")
        print("=" * 80)
