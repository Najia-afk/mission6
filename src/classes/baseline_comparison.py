import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score


class BaselineComparison:
    """Compare model against baseline classifiers"""
    
    def __init__(self, classifier):
        """
        Initialize baseline comparison
        
        Args:
            classifier: TransferLearningClassifier instance
        """
        summary = classifier.get_summary()
        if not summary['best_model']:
             raise ValueError("Classifier has no trained models")
        best_model_name = summary['best_model']['name']
        
        self.best_model = classifier.models[best_model_name]
        self.X_test = classifier.X_test
        # Convert one-hot to integer labels for sklearn metrics
        self.y_test = np.argmax(classifier.y_test, axis=1)
        self.label_encoder = classifier.label_encoder
        
    def compare(self, X_train, y_train):
        """
        Compare best model vs baselines
        
        Args:
            X_train: Training features
            y_train: Training labels (can be strings or integers)
            
        Returns:
            dict: Results for each model
        """
        # If y_train is strings, encode it
        if len(y_train) > 0 and isinstance(y_train[0], str):
             y_train = self.label_encoder.transform(y_train)

        # Get predictions from best model
        y_pred_probs = self.best_model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        model_acc = accuracy_score(self.y_test, y_pred)
        model_f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # Baseline classifiers
        baselines = {
            'Random': DummyClassifier(strategy='uniform', random_state=42),
            'Stratified': DummyClassifier(strategy='stratified', random_state=42),
            'Most Frequent': DummyClassifier(strategy='most_frequent'),
        }
        
        results = {
            'Best Model (Transfer Learning)': {
                'accuracy': model_acc,
                'f1_weighted': model_f1
            }
        }
        
        for name, clf in baselines.items():
            clf.fit(X_train, y_train)
            y_baseline = clf.predict(self.X_test)
            results[name] = {
                'accuracy': accuracy_score(self.y_test, y_baseline),
                'f1_weighted': f1_score(self.y_test, y_baseline, average='weighted')
            }
        
        return results
