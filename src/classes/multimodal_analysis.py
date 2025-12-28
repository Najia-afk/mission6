import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm
from src.classes.advanced_embeddings import AdvancedTextEmbeddings


class MultimodalAnalysis:
    """Analyze multimodal fusion of text + image features"""
    
    def __init__(self, classifier):
        """
        Initialize multimodal analysis
        
        Args:
            classifier: TransferLearningClassifier instance
        """
        self.classifier = classifier
        self.num_classes = classifier.num_classes
        self.label_encoder = classifier.label_encoder
        
    def extract_vgg16_batch(self, images, batch_size=64):
        """
        Extract VGG16 features in batches to avoid OOM
        
        Args:
            images: Image array (N, 224, 224, 3)
            batch_size: Batch size for processing (default 64)
            
        Returns:
            np.ndarray: VGG16 features (N, 512)
        """
        vgg = tf.keras.applications.VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        vgg.trainable = False
        
        features = []
        num_batches = (len(images) + batch_size - 1) // batch_size
        
        print(f"Extracting VGG16 features from {len(images)} images in batches of {batch_size}...\n")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch = images[start_idx:end_idx]
            
            batch_feat = vgg(batch, training=False)
            batch_feat = tf.keras.layers.GlobalAveragePooling2D()(batch_feat)
            features.append(batch_feat.numpy())
            
            if (i + 1) % 5 == 0 or end_idx == len(images):
                print(f"  ✓ Processed {end_idx}/{len(images)} images")
        
        print()
        return np.concatenate(features, axis=0)
    
    def extract_text_features(self, product_names):
        """
        Extract USE text embeddings
        
        Args:
            product_names: List of product names
            
        Returns:
            np.ndarray: USE embeddings (N, 512)
        """
        print("Extracting USE text embeddings...\n")
        adv_emb = AdvancedTextEmbeddings()
        features = adv_emb.fit_transform_use(product_names)
        print(f"  ✓ USE features extracted: {features.shape}\n")
        return features
    
    def fuse_modalities(self, text_features, image_features):
        """
        Fuse text and image features via concatenation
        
        Args:
            text_features: USE embeddings (N, 512)
            image_features: VGG16 features (N, 512)
            
        Returns:
            np.ndarray: Fused features (N, 1024)
        """
        fused = np.concatenate([text_features, image_features], axis=1)
        print(f"Fused features (TEXT + IMAGE): {fused.shape}\n")
        return fused
    
    def evaluate_fusion(self, images, labels, text_descriptions):
        """
        Compare fusion performance vs single modality
        
        Args:
            images: Test images (N, 224, 224, 3)
            labels: Test labels (strings or integers)
            text_descriptions: Test text descriptions
            
        Returns:
            dict: Comparison results
        """
        # 1. Extract Image Features
        # Use base model from classifier
        full_model = self.classifier.models.get('base_vgg16') or self.classifier.models.get('augmented_vgg16')
        if not full_model:
             if self.classifier.models:
                 full_model = list(self.classifier.models.values())[0]
             else:
                 raise ValueError("No trained model found in classifier")
        
        pooling_layer = None
        for layer in full_model.layers:
            if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                pooling_layer = layer
                break
        
        if not pooling_layer:
             raise ValueError("Could not find GlobalAveragePooling2D layer")
             
        feature_extractor = tf.keras.Model(inputs=full_model.input, outputs=pooling_layer.output)
        
        print("Extracting image features...")
        # Use manual batching with tqdm for notebook-friendly progress bar
        batch_size = 32
        features_list = []
        num_batches = int(np.ceil(len(images) / batch_size))
        
        for i in tqdm(range(num_batches), desc="Extracting image features"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch = images[start_idx:end_idx]
            batch_preds = feature_extractor.predict(batch, verbose=0)
            features_list.append(batch_preds)
            
        image_features = np.concatenate(features_list, axis=0)
        
        # 2. Extract Text Features
        print("Extracting text features...")
        adv_emb = AdvancedTextEmbeddings()
        text_features = adv_emb.fit_transform_use(text_descriptions)
        
        # 3. Fuse
        fused_features = np.concatenate([text_features, image_features], axis=1)
        
        # 4. Prepare Labels
        if len(labels) > 0 and isinstance(labels[0], str):
             y_encoded = self.label_encoder.transform(labels)
        else:
             y_encoded = labels # Assume integers
             
        # 5. Evaluate
        # Train simple classifier on fused features (using a split of the provided data)
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, f1_score
        
        print("Training simple classifier on fused features (80/20 split of provided data)...\n")
        
        scaler = StandardScaler()
        X_fused_scaled = scaler.fit_transform(fused_features)
        
        split_idx = int(0.8 * len(X_fused_scaled))
        X_train_f = X_fused_scaled[:split_idx]
        y_train_f = y_encoded[:split_idx]
        X_test_f = X_fused_scaled[split_idx:]
        y_test_f = y_encoded[split_idx:]
        
        clf = SVC(kernel='rbf', C=1.0, random_state=42)
        clf.fit(X_train_f, y_train_f)
        
        y_pred_fused = clf.predict(X_test_f)
        
        fusion_accuracy = accuracy_score(y_test_f, y_pred_fused)
        fusion_f1 = f1_score(y_test_f, y_pred_fused, average='weighted')
        
        # Get image-only accuracy on the same subset for fair comparison
        # We use SVM on image features to be fair
        clf_img = SVC(kernel='rbf', C=1.0, random_state=42)
        # Re-fit scaler on image features
        scaler_img = StandardScaler()
        X_img_scaled = scaler_img.fit_transform(image_features)
        
        clf_img.fit(X_img_scaled[:split_idx], y_encoded[:split_idx])
        y_pred_img = clf_img.predict(X_img_scaled[split_idx:])
        image_accuracy = accuracy_score(y_test_f, y_pred_img)
        
        results = {
            'image_only_accuracy': image_accuracy,
            'fusion_accuracy': fusion_accuracy,
            'fusion_f1': fusion_f1,
            'improvement': (fusion_accuracy - image_accuracy) / image_accuracy * 100 if image_accuracy > 0 else 0,
            'y_true': y_test_f,
            'y_pred': y_pred_fused
        }
        
        print(f"✓ Image-only (SVM on VGG features): {results['image_only_accuracy']:.4f}")
        print(f"✓ Fusion (Text+Image): {results['fusion_accuracy']:.4f}")
        print(f"✓ Fusion F1 Score: {results['fusion_f1']:.4f}")
        print(f"✓ Improvement: {results['improvement']:+.1f}%\n")
        
        return results
