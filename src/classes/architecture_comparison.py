import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import f1_score


class ArchitectureComparison:
    """Compare different classifier architectures on extracted features"""
    
    def __init__(self, classifier):
        """
        Initialize architecture comparison
        
        Args:
            classifier: TransferLearningClassifier instance
        """
        self.classifier = classifier
        self.num_classes = classifier.num_classes
        self.label_encoder = classifier.label_encoder
        
    def get_architectures(self):
        """
        Define classifier head architectures
        
        Returns:
            dict: Architecture definitions
        """
        return {
            'Simple (2 layers)': [
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
            ],
            'Medium (3 layers)': [
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
            ],
            'Deep (4 layers)': [
                layers.Dense(1024, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
            ]
        }
    
    def compare(self, X_train, y_train, X_val, y_val, epochs=1):
        """
        Train and compare different architectures
        
        Args:
            X_train: Training images (N, 224, 224, 3)
            y_train: Training labels (strings or one-hot)
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs (default 1 for quick demo)
            
        Returns:
            dict: Validation accuracy for each architecture
        """
        # 1. Extract features using the base model from classifier
        # We need the base model (VGG16) without the top.
        full_model = self.classifier.models.get('base_vgg16') or self.classifier.models.get('augmented_vgg16')
        if not full_model:
             # Fallback: try to find any model
             if self.classifier.models:
                 full_model = list(self.classifier.models.values())[0]
             else:
                 raise ValueError("No trained model found in classifier")
        
        # Find the global average pooling layer
        pooling_layer = None
        for layer in full_model.layers:
            if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                pooling_layer = layer
                break
        
        if not pooling_layer:
             raise ValueError("Could not find GlobalAveragePooling2D layer in base model")
        
        feature_extractor = tf.keras.Model(inputs=full_model.input, outputs=pooling_layer.output)
        
        print("Extracting features from training images...")
        X_train_features = feature_extractor.predict(X_train, verbose=1)
        print("Extracting features from validation images...")
        X_val_features = feature_extractor.predict(X_val, verbose=1)
        
        # 2. Encode labels to one-hot if they are strings
        if len(y_train) > 0 and isinstance(y_train[0], str):
             y_train_int = self.label_encoder.transform(y_train)
             y_train_enc = tf.keras.utils.to_categorical(y_train_int, num_classes=self.num_classes)
             
             y_val_int = self.label_encoder.transform(y_val)
             y_val_enc = tf.keras.utils.to_categorical(y_val_int, num_classes=self.num_classes)
        else:
             y_train_enc = y_train
             y_val_enc = y_val

        results = {}
        
        print("Training different architectures...\n")
        
        for arch_name, dense_layers in self.get_architectures().items():
            print(f"Testing {arch_name}...")
            
            # Build model
            inputs = tf.keras.Input(shape=(X_train_features.shape[1],))
            x = inputs
            for layer in dense_layers:
                x = layer(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Quick training
            history = model.fit(
                X_train_features, y_train_enc,
                validation_data=(X_val_features, y_val_enc),
                epochs=epochs,
                batch_size=32,
                verbose=0
            )
            
            val_acc = history.history['val_accuracy'][-1]
            
            # Calculate F1
            y_pred_probs = model.predict(X_val_features, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(y_val_enc, axis=1)
            
            val_f1 = f1_score(y_true, y_pred, average='weighted')
            
            results[arch_name] = val_acc
            print(f"  ✓ Val Accuracy: {val_acc:.4f}")
            print(f"  ✓ Val F1 Score: {val_f1:.4f}\n")
        
        return results
