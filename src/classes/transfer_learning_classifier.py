"""
Transfer Learning Classifier for E-commerce Product Image Classification
Uses pre-trained models (VGG16, ResNet50, etc.) with transfer learning
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple, Union, Optional
import time
import glob
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.notebook import tqdm


class TQDMProgressBar(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epochs_pbar = None
        self.batch_pbar = None
        self.epoch_start_time = None
        self.last_batch_start = None
        self.batch_times = []
        self.steps_in_epoch = 0

    def on_train_begin(self, logs=None):
        total_epochs = self.params.get('epochs', 0)
        if total_epochs:
            self.epochs_pbar = tqdm(total=total_epochs, desc="Training epochs", unit="epoch")

    def _compute_steps(self):
        steps = self.params.get('steps', None)
        if steps is None:
            samples = self.params.get('samples', 0)
            batch_size = self.params.get('batch_size', 32)
            steps = int(np.ceil(samples / max(1, batch_size))) if samples else 0
        return steps or 0

    def on_epoch_begin(self, epoch, logs=None):
        self.steps_in_epoch = self._compute_steps()
        self.batch_times = []
        self.epoch_start_time = time.time()
        if self.steps_in_epoch:
            self.batch_pbar = tqdm(
                total=self.steps_in_epoch,
                desc=f"Epoch {epoch + 1}/{self.params.get('epochs', '?')}",
                unit="batch",
                leave=False
            )

    def on_train_batch_begin(self, batch, logs=None):
        self.last_batch_start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if self.batch_pbar is not None:
            self.batch_pbar.update(1)
        if self.last_batch_start is not None:
            self.batch_times.append(time.time() - self.last_batch_start)

    def on_epoch_end(self, epoch, logs=None):
        # Close batch bar
        if self.batch_pbar is not None:
            self.batch_pbar.close()
            self.batch_pbar = None

        # Update epoch bar
        if self.epochs_pbar is not None:
            self.epochs_pbar.update(1)

        # Compose single compact summary line that mimics Keras' last-batch line
        logs = logs or {}
        acc = logs.get('accuracy', None)
        loss = logs.get('loss', None)
        val_acc = logs.get('val_accuracy', None)
        val_loss = logs.get('val_loss', None)

        epoch_secs = time.time() - (self.epoch_start_time or time.time())
        avg_ms_per_step = (np.mean(self.batch_times) * 1000.0) if self.batch_times else 0.0

        steps_txt = f"{self.steps_in_epoch}/{self.steps_in_epoch}" if self.steps_in_epoch else ""
        bar_txt = "━━━━━━━━━━━━━━━━━━━━"  # decorative bar
        time_txt = f"{int(epoch_secs)}s {int(avg_ms_per_step)}ms/step"

        metrics_parts = []
        if acc is not None:
            metrics_parts.append(f"accuracy: {acc:.4f}")
        if loss is not None:
            metrics_parts.append(f"loss: {loss:.4f}")
        if val_acc is not None:
            metrics_parts.append(f"val_accuracy: {val_acc:.4f}")
        if val_loss is not None:
            metrics_parts.append(f"val_loss: {val_loss:.4f}")

        line = f"{steps_txt} {bar_txt} {time_txt} - " + " - ".join(metrics_parts)
        # Print cleanly without breaking tqdm bars
        tqdm.write(line)

    def on_train_end(self, logs=None):
        if self.epochs_pbar is not None:
            self.epochs_pbar.close()
            self.epochs_pbar = None


class TransferLearningClassifier:
    """
    Transfer Learning Classifier for image classification
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: Optional[int] = None,
                 base_model_name: str = 'VGG16',
                 weights: str = 'imagenet',
                 use_gpu: bool = True,
                 random_state: int = 42,
                 model_dir: str = 'models'):
        """
        Initialize the Transfer Learning Classifier
        
        Args:
            input_shape: Input shape for the model (height, width, channels)
            num_classes: Number of classes for classification
            base_model_name: Name of the base model ('VGG16', 'ResNet50', etc.)
            weights: Pre-trained weights to use
            use_gpu: Whether to use GPU for training
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model_name = base_model_name
        self.weights = weights
        self.random_state = random_state
        self.model_dir = model_dir
        
        # Set up GPU configuration
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and use_gpu:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Error setting up GPU: {e}")
        
        # Dataset attributes
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.class_names = None
        self.label_encoder = None
        
        # Model attributes
        self.models = {}
        self.histories = {}
        self.evaluation_results = {}
        
        print(f"🔧 Transfer Learning Classifier initialized")
        print(f"   📊 Input shape: {self.input_shape}")
        print(f"   🎯 GPU Available: {len(gpus)}")
    
    def prepare_data_from_dataframe(self,
                                   df: pd.DataFrame,
                                   image_column: str,
                                   category_column: str,
                                   test_size: float = 0.2,
                                   val_size: float = 0.2,
                                   random_state: int = 42,
                                   image_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare data from a DataFrame containing image paths and categories.
        """
        print("🔄 Preparing data from DataFrame...")
        
        df = df.copy()

        # --- Corrected Path Handling Logic ---
        # Determine the directory to use for images.
        final_image_dir = image_dir
        if not final_image_dir:
            # If no directory is passed, try a default location.
            default_dir = 'dataset/Flipkart/Images'
            if os.path.exists(default_dir):
                final_image_dir = default_dir
                print(f"   📁 Using default image directory: {final_image_dir}")

        # If we have a directory, construct the full path safely.
        if final_image_dir:
            # Use os.path.basename to prevent creating a double path.
            # This takes only the filename part of the input path.
            df['image_path'] = df[image_column].apply(
                lambda p: os.path.join(final_image_dir, os.path.basename(p))
            )
        else:
            # If no directory is found or provided, use the column as-is.
            df['image_path'] = df[image_column]
            print("   ⚠️ No image directory specified or found. Using raw paths from DataFrame.")

        # Encode categories
        self.label_encoder = LabelEncoder()
        df['label_encoded'] = self.label_encoder.fit_transform(df[category_column])
        self.class_names = self.label_encoder.classes_
        
        if self.num_classes is None:
            self.num_classes = len(self.class_names)
        
        # Split data
        train_val_df, self.test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df[category_column]
        )
        
        self.train_df, self.val_df = train_test_split(
            train_val_df, test_size=val_size, random_state=random_state, stratify=train_val_df[category_column]
        )
        
        # Print summary
        print(f"   📋 Categories found: {self.class_names.tolist()}")
        print(f"   🎯 Number of classes: {self.num_classes}")
        print(f"   📊 Train samples: {len(self.train_df)}")
        print(f"   📊 Validation samples: {len(self.val_df)}")
        print(f"   📊 Test samples: {len(self.test_df)}")
        
        # Verify image paths exist
        sample_image = self.train_df.iloc[0]['image_path']
        if not os.path.exists(sample_image):
            print(f"   ⚠️ Warning: Sample image path does not exist: {sample_image}")
            print(f"   🔍 Please check your image paths or provide correct image_dir")
        
        return {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "train_size": len(self.train_df),
            "val_size": len(self.val_df),
            "test_size": len(self.test_df)
        }

    def _load_and_preprocess_images(self, df: pd.DataFrame, image_column: str = 'image_path') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess images from DataFrame
        
        Args:
            df: DataFrame containing image paths
            image_column: Name of the column containing image paths
            
        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []
        
        if len(df) == 0:
            print(f"   ⚠️ Warning: Empty DataFrame provided to load_and_preprocess_images")
            return np.array([]), np.array([])
        
        print(f"   🖼️ Loading {len(df)} images...")
        failed_count = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading & preprocessing", unit="img"):
            try:
                image_path = row[image_column]
                
                # Try alternative paths if image doesn't exist
                if not os.path.exists(image_path):
                    alt_paths = [
                        # Try dataset/Flipkart/Images directory
                        os.path.join('dataset/Flipkart/Images', os.path.basename(image_path)),
                        # Try with just the filename in current directory
                        os.path.basename(image_path)
                    ]
                    
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path):
                            image_path = alt_path
                            break
                
                # If image still doesn't exist after trying alternatives
                if not os.path.exists(image_path):
                    failed_count += 1
                    continue
                    
                # Load and preprocess image
                img = load_img(image_path, target_size=self.input_shape[:2])
                img = img_to_array(img)
                
                # Apply preprocessing based on model type
                if self.base_model_name == 'VGG16':
                    img = vgg16_preprocess(img)
                else:
                    img = img / 255.0  # Simple normalization for other models
                
                images.append(img)
                labels.append(row['label_encoded'])
            except Exception as e:
                failed_count += 1
                if failed_count <= 20:  # Limit the number of error messages
                    print(f"   ⚠️ Error loading image {row[image_column]}: {e}")
                elif failed_count == 21:
                    print(f"   ⚠️ Too many errors, suppressing further messages...")
        
        # Check if we have any images
        if len(images) == 0:
            print(f"   ❌ No images could be loaded successfully! ({failed_count} failures)")
            print(f"   🔧 Generating synthetic data for demonstration purposes...")
            
            # Generate synthetic data as a fallback
            synthetic_images = []
            synthetic_labels = []
            
            # Create synthetic data with the right shape
            for i in tqdm(range(min(100, len(df))), desc="Synthetic data", unit="img"):
                # Create a synthetic image with the right shape, filled with random data
                synthetic_img = np.random.rand(*self.input_shape) * 2.0 - 1.0  # Range [-1, 1] for VGG16
                synthetic_images.append(synthetic_img)
                
                # Use the original label if available, otherwise use a random label
                if i < len(df):
                    synthetic_labels.append(df.iloc[i]['label_encoded'])
                else:
                    synthetic_labels.append(np.random.randint(0, self.num_classes))
            
            return np.array(synthetic_images), to_categorical(synthetic_labels, num_classes=self.num_classes)
        else:
            print(f"   ✅ Successfully loaded {len(images)} images ({failed_count} failures)")
        
        return np.array(images), to_categorical(labels, num_classes=self.num_classes)

    def prepare_arrays_method(self) -> Dict[str, Any]:
        """
        Prepare arrays for training using standard train/val/test split
        
        Returns:
            Dictionary with data summary
        """
        print("🔄 Preparing data using arrays method...")
        
        # Load and preprocess images
        self.X_train, self.y_train = self._load_and_preprocess_images(self.train_df)
        self.X_val, self.y_val = self._load_and_preprocess_images(self.val_df)
        self.X_test, self.y_test = self._load_and_preprocess_images(self.test_df)
        
        # Print summary
        print(f"   📊 Train set: {self.X_train.shape if self.X_train.size > 0 else '(0,)'}")
        print(f"   📊 Validation set: {self.X_val.shape if self.X_val.size > 0 else '(0,)'}")
        print(f"   📊 Test set: {self.X_test.shape if self.X_test.size > 0 else '(0,)'}")
        
        # Check if we have enough data
        if self.X_train.size == 0 or self.X_val.size == 0 or self.X_test.size == 0:
            print("   ⚠️ Warning: One or more datasets are empty!")
        
        return {
            "X_train_shape": self.X_train.shape if self.X_train.size > 0 else (0,),
            "X_val_shape": self.X_val.shape if self.X_val.size > 0 else (0,),
            "X_test_shape": self.X_test.shape if self.X_test.size > 0 else (0,),
        }

    def create_base_model(self, show_backbone_summary: bool = False) -> tf.keras.Model:
        """
        Create a base model with pre-trained weights.
        Optionally display the backbone summary before attaching the head.
        """
        print(f"🔧 Creating base model with {self.base_model_name}...")
        # Instantiate backbone
        if self.base_model_name == 'VGG16':
            base_model = VGG16(
                weights=self.weights,
                include_top=False,
                input_shape=self.input_shape,
                name=self.base_model_name.lower()
            )
        elif self.base_model_name == 'ResNet50':
            base_model = ResNet50(
                weights=self.weights,
                include_top=False,
                input_shape=self.input_shape,
                name=self.base_model_name.lower()
            )
        else:
            raise ValueError(f"Unsupported base model: {self.base_model_name}")

        # Freeze backbone
        base_model.trainable = False

        if show_backbone_summary:
            print("=== Backbone Summary (Frozen) ===")
            base_model.summary(line_length=80)

        # Build head
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=predictions)

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("   ✅ Base model created and compiled.")
        model.summary()
        return model

    
    def _calculate_ari_score(self, model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Calculate the Adjusted Rand Index for a given model's predictions.
        
        Args:
            model: The trained Keras model.
            X_test: The test data features.
            y_test: The one-hot encoded true labels for the test data.
            
        Returns:
            The ARI score as a float.
        """
        if X_test is None or y_test is None or X_test.size == 0:
            return float('nan')
            
        # Get model predictions
        y_pred = model.predict(X_test, verbose=0)
        
        # Convert predictions and true labels from one-hot to single class indices
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate and return ARI score
        return adjusted_rand_score(y_true_classes, y_pred_classes)
    
    def create_augmented_model(self) -> tf.keras.Model:
        """
        Create a model with data augmentation and fine-tuning.
        
        Returns:
            Keras model
        """
        print(f"🔧 Creating augmented model with {self.base_model_name} for fine-tuning...")
        
        # Create the base model structure first
        model = self.create_base_model()
        
        # Find the actual base model layer within the full model by its name.
        # This is robust and avoids the error. The default names are 'vgg16', 'resnet50', etc.
        try:
            base_model_layer = model.get_layer(self.base_model_name.lower())
        except ValueError:
            print(f"   ❌ Error: Could not find layer named '{self.base_model_name.lower()}' in the model. Aborting fine-tuning.")
            return model

        # Unfreeze some top layers of the base model for fine-tuning
        if self.base_model_name == 'VGG16':
            # Unfreeze the top convolutional layers (e.g., block5)
            for layer in base_model_layer.layers[-4:]:
                layer.trainable = True
        elif self.base_model_name == 'ResNet50':
            # Unfreeze the top convolutional layers
            for layer in base_model_layer.layers[-10:]:
                layer.trainable = True
        
        # Re-compile the model with a lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("   ✅ Model re-compiled for fine-tuning with a lower learning rate.")
        model.summary()
        return model
    
    
    def train_model(self,
                   model_name: str,
                   model: tf.keras.Model,
                   epochs: int = 20,
                   batch_size: int = 32,
                   patience: int = 5,
                   use_generators: bool = False) -> Dict[str, Any]:
        """
        Train a model
        
        Args:
            model_name: Name to identify the model
            model: Keras model to train
            epochs: Number of epochs to train
            batch_size: Batch size for training
            patience: Patience for early stopping
            use_generators: Whether to use data generators for training
            
        Returns:
            Dictionary with training results
        """
        print(f"🔄 Training model: {model_name}...")
        
        # Check if data is available
        if self.X_train.size == 0 or self.X_val.size == 0:
            print("   ❌ Error: Cannot train model because training or validation data is empty")
            return {
                'error': 'No training data available',
                'model': model,
                'history': None,
                'evaluation': {
                    'loss': float('nan'),
                    'accuracy': float('nan')
                },
                'training_time': 0
            }
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"{model_name}_best.keras")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
            ModelCheckpoint(filepath=model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
            TQDMProgressBar()
        ]
        
        
        # Start training
        start_time = time.time()
        
        try:
            if use_generators:
                # Create data generators
                train_datagen = ImageDataGenerator(
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest'
                )
                
                val_datagen = ImageDataGenerator()
                
                # Create generators
                train_generator = train_datagen.flow(
                    self.X_train, self.y_train,
                    batch_size=batch_size
                )
                
                val_generator = val_datagen.flow(
                    self.X_val, self.y_val,
                    batch_size=batch_size
                )
                
                # Train model
                history = model.fit(
                    train_generator,
                    validation_data=val_generator,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=0  # suppress Keras batch logs; we print a compact epoch-end line instead
                )
            else:
                # Train without generators
                history = model.fit(
                    self.X_train, self.y_train,
                    validation_data=(self.X_val, self.y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=0  # suppress Keras batch logs; we print a compact epoch-end line instead
                )
                
            # Load best weights
            if os.path.exists(model_path):
                model.load_weights(model_path)
        except Exception as e:
            print(f"   ❌ Error during training: {e}")
            return {
                'error': str(e),
                'model': model,
                'history': None,
                'evaluation': {
                    'loss': float('nan'),
                    'accuracy': float('nan')
                },
                'training_time': time.time() - start_time
            }
        
        # Calculate training time
        training_time = time.time() - start_time
        
       # Evaluate model and calculate ARI if test data is available
        if self.X_test.size > 0:
            evaluation = model.evaluate(self.X_test, self.y_test, verbose=0)
            ari_score = self._calculate_ari_score(model, self.X_test, self.y_test)
        else:
            print("   ⚠️ No test data available for evaluation")
            evaluation = [float('nan'), float('nan')]  # Loss, accuracy
            ari_score = float('nan')

        # Save model and history
        self.models[model_name] = model
        self.histories[model_name] = history.history
        
        # Save evaluation results
        self.evaluation_results[model_name] = {
            'loss': evaluation[0],
            'accuracy': evaluation[1],
            'training_time': training_time,
            'ari_score': ari_score 
        }
        
        print(f"✅ Training completed in {training_time:.2f}s")
        print(f"   📊 Test accuracy: {evaluation[1]:.4f}")
        print(f"   📊 ARI Score: {ari_score:.4f}")
        
        return {
            'model': model,
            'history': history.history,
            'evaluation': {
                'loss': evaluation[0],
                'accuracy': evaluation[1]
            },
            'training_time': training_time
        }
    
    def compare_models(self) -> go.Figure:
        """
        Compare models based on evaluation metrics
        
        Returns:
            Plotly figure comparing models
        """
        print("📊 Comparing models...")
        
        if not self.evaluation_results:
            print("⚠️ No models to compare. Train models first.")
            return None
        
        # Create figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Test Accuracy', 'Training Time'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Get data
        model_names = list(self.evaluation_results.keys())
        accuracies = [result['accuracy'] * 100 for result in self.evaluation_results.values()]
        training_times = [result['training_time'] for result in self.evaluation_results.values()]
        
        # Add traces
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=accuracies,
                name='Test Accuracy (%)',
                marker_color='royalblue',
                text=[f"{acc:.2f}%" for acc in accuracies],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=training_times,
                name='Training Time (s)',
                marker_color='lightgreen',
                text=[f"{time:.2f}s" for time in training_times],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Model Comparison',
            height=500,
            width=900,
            showlegend=False
        )
        
        return fig
    
    def plot_training_history(self, model_name: Optional[str] = None) -> go.Figure:
        """
        Plot training history for a model
        
        Args:
            model_name: Name of the model to plot history for. If None, plots all models.
            
        Returns:
            Plotly figure with training history
        """
        print(f"📊 Plotting training history...")
        
        if not self.histories:
            print("⚠️ No training history available. Train models first.")
            return None
        
        if model_name is not None and model_name not in self.histories:
            print(f"⚠️ Model '{model_name}' not found.")
            return None
        
        # Create figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training and Validation Accuracy', 'Training and Validation Loss'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        if model_name is not None:
            # Plot history for specific model
            history = self.histories[model_name]
            
            # Accuracy
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history['accuracy']) + 1)),
                    y=history['accuracy'],
                    mode='lines',
                    name=f'{model_name} - Training Accuracy',
                    line=dict(color='royalblue')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history['val_accuracy']) + 1)),
                    y=history['val_accuracy'],
                    mode='lines',
                    name=f'{model_name} - Validation Accuracy',
                    line=dict(color='lightblue', dash='dash')
                ),
                row=1, col=1
            )
            
            # Loss
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history['loss']) + 1)),
                    y=history['loss'],
                    mode='lines',
                    name=f'{model_name} - Training Loss',
                    line=dict(color='firebrick')
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history['val_loss']) + 1)),
                    y=history['val_loss'],
                    mode='lines',
                    name=f'{model_name} - Validation Loss',
                    line=dict(color='lightcoral', dash='dash')
                ),
                row=1, col=2
            )
        else:
            # Plot history for all models with different colors
            colors = ['royalblue', 'firebrick', 'green', 'purple', 'orange']
            
            for i, (name, history) in enumerate(self.histories.items()):
                color_idx = i % len(colors)
                
                # Accuracy
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(history['accuracy']) + 1)),
                        y=history['accuracy'],
                        mode='lines',
                        name=f'{name} - Training Accuracy',
                        line=dict(color=colors[color_idx])
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(history['val_accuracy']) + 1)),
                        y=history['val_accuracy'],
                        mode='lines',
                        name=f'{name} - Validation Accuracy',
                        line=dict(color=colors[color_idx], dash='dash')
                    ),
                    row=1, col=1
                )
                
                # Loss
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(history['loss']) + 1)),
                        y=history['loss'],
                        mode='lines',
                        name=f'{name} - Training Loss',
                        line=dict(color=colors[color_idx])
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(history['val_loss']) + 1)),
                        y=history['val_loss'],
                        mode='lines',
                        name=f'{name} - Validation Loss',
                        line=dict(color=colors[color_idx], dash='dash')
                    ),
                    row=1, col=2
                )
        
        # Update layout
        fig.update_layout(
            title='Training History',
            height=500,
            width=1000,
            legend=dict(orientation='h', y=-0.2)
        )
        
        fig.update_xaxes(title_text='Epoch', row=1, col=1)
        fig.update_xaxes(title_text='Epoch', row=1, col=2)
        fig.update_yaxes(title_text='Accuracy', row=1, col=1)
        fig.update_yaxes(title_text='Loss', row=1, col=2)
        
        return fig
    

    
    def plot_confusion_matrix(self, model_name: str) -> go.Figure:
        """
        Plot confusion matrix for a model
        
        Args:
            model_name: Name of the model to plot confusion matrix for
            
        Returns:
            Plotly figure with confusion matrix
        """
        print(f"📊 Plotting confusion matrix for {model_name}...")
        
        if model_name not in self.models:
            print(f"⚠️ Model '{model_name}' not found.")
            return None
        
        # Get model and predictions
        model = self.models[model_name]
        y_pred = model.predict(self.X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=cm_norm,
            x=[self.class_names[i] for i in range(len(self.class_names))],
            y=[self.class_names[i] for i in range(len(self.class_names))],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverinfo="x+y+z+text"
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Confusion Matrix - {model_name}",
            xaxis=dict(title='Predicted Label'),
            yaxis=dict(title='True Label'),
            height=800
        )
        
        return fig
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of classifier results, including the best model.
        
        Returns:
            Dictionary with summary information
        """
        summary = {
            "data": {
                "num_classes": self.num_classes,
                "class_names": self.class_names.tolist() if self.class_names is not None else None,
                "train_size": len(self.train_df) if self.train_df is not None else 0,
                "val_size": len(self.val_df) if self.val_df is not None else 0,
                "test_size": len(self.test_df) if self.test_df is not None else 0
            },
            "models": {
                name: {
                    "accuracy": result["accuracy"],
                    "loss": result["loss"],
                    "training_time": result["training_time"]
                }
                for name, result in self.evaluation_results.items()
            },
            "best_model": None  # Initialize best_model as None
        }
        
        # Find the best model based on test accuracy
        if self.evaluation_results:
            # Find the model name with the highest accuracy
            best_model_name = max(self.evaluation_results, key=lambda k: self.evaluation_results[k]['accuracy'])
            best_model_stats = self.evaluation_results[best_model_name]
            
            # Get the peak validation accuracy from the training history for the best model
            best_val_accuracy = 0
            if best_model_name in self.histories and 'val_accuracy' in self.histories[best_model_name]:
                best_val_accuracy = max(self.histories[best_model_name]['val_accuracy'])

            summary["best_model"] = {
                "name": best_model_name,
                "test_accuracy": best_model_stats["accuracy"],
                "test_loss": best_model_stats["loss"],
                "val_accuracy": best_val_accuracy,
                "training_time": best_model_stats["training_time"]
            }
        
        return summary
    

    def plot_prediction_examples(self,
                                 model_name: str,
                                 num_correct: int = 4,
                                 num_incorrect: int = 4,
                                 uniq_id: Optional[str] = None) -> go.Figure:
        """
        Show prediction examples.
        If uniq_id provided: show only that sample (if in test set).
        Else: random correct / incorrect samples.
        """
        print(f"🖼️ Visualizing prediction examples for model: {model_name}")
        if (model_name not in self.models):
            print(f"   ❌ Model '{model_name}' not found.")
            return go.Figure()

        model = self.models[model_name]
        y_pred = model.predict(self.X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)

        if uniq_id:
            print(f"   🔍 Looking for uniq_id={uniq_id}")
            target_indices = self.test_df.index[self.test_df['uniq_id'] == uniq_id].tolist()
            if not target_indices:
                print(f"   ❌ uniq_id '{uniq_id}' not found in test set.")
                return go.Figure()
            # Position within test_df (sequential integer position)
            try:
                pos = list(self.test_df.index).index(target_indices[0])
            except ValueError:
                print(f"   ❌ Could not map uniq_id '{uniq_id}' to position.")
                return go.Figure()
            all_indices = np.array([pos])
            plot_title = f"Prediction for uniq_id: {uniq_id}"
        else:
            correct_indices = np.where(y_pred_classes == y_true_classes)[0]
            incorrect_indices = np.where(y_pred_classes != y_true_classes)[0]
            sel_correct = np.random.choice(correct_indices,
                                           min(num_correct, len(correct_indices)),
                                           replace=False) if len(correct_indices) else []
            sel_incorrect = np.random.choice(incorrect_indices,
                                             min(num_incorrect, len(incorrect_indices)),
                                             replace=False) if len(incorrect_indices) else []
            all_indices = np.concatenate([sel_correct, sel_incorrect])
            plot_title = "Model Prediction Analysis (Correct vs Incorrect)"

        if len(all_indices) == 0:
            print("   ⚠️ No examples to display.")
            return go.Figure()

        num_plots = len(all_indices)
        # Layout: up to 4 per row
        cols = 4 if num_plots > 2 else num_plots
        cols = max(1, cols)
        rows = (num_plots + cols - 1) // cols
        fig = make_subplots(rows=rows, cols=cols)

        original_images = []
        for idx in all_indices:
            img_path = self.test_df.iloc[idx]['image_path']
            original_images.append(Image.open(img_path).resize((224, 224)))

        def visualize_preprocess(img):
            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            arr = vgg16_preprocess(arr)
            arr = arr[0]
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            return Image.fromarray(arr.astype('uint8'))

        # Add traces + annotations
        for i, (test_idx, pil_img) in enumerate(zip(all_indices, original_images)):
            row = (i // cols) + 1
            col = (i % cols) + 1
            fig.add_trace(go.Image(z=np.array(pil_img)), row=row, col=col)
            true_label = self.class_names[y_true_classes[test_idx]]
            pred_label = self.class_names[y_pred_classes[test_idx]]
            title_color = "green" if true_label == pred_label else "red"

            # Compute the center/top of the subplot in paper coords
            axis_num = i + 1
            suffix = "" if axis_num == 1 else str(axis_num)
            xaxis = getattr(fig.layout, f"xaxis{suffix}")
            yaxis = getattr(fig.layout, f"yaxis{suffix}")
            xmid = (xaxis.domain[0] + xaxis.domain[1]) / 2.0
            ytop = min(0.98, yaxis.domain[1] + 0.03)  # a bit above the image, but inside the figure

            fig.add_annotation(
                text=f"<b>True:</b> {true_label}<br><b>Pred:</b> {pred_label}",
                font=dict(color=title_color, size=12),
                x=xmid, y=ytop,
                xref="paper", yref="paper",
                showarrow=False,
                align="center",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1
            )

        # Precompute all transformations as numpy arrays
        original_arrays = [np.array(im) for im in original_images]

        contrast_arrays = [np.array(ImageEnhance.Contrast(im).enhance(2.0)) for im in original_images]

        grayscale_arrays = [np.array(ImageOps.grayscale(im).convert("RGB")) for im in original_images]

        blur_arrays = [np.array(im.filter(ImageFilter.GaussianBlur(radius=2))) for im in original_images]

        vgg_arrays = []
        for im in original_images:
            arr = img_to_array(im)
            arr_batch = np.expand_dims(arr.copy(), axis=0)
            arr_batch = vgg16_preprocess(arr_batch)
            arr_v = arr_batch[0]
            arr_v = (arr_v - arr_v.min()) / (arr_v.max() - arr_v.min() + 1e-8) * 255
            vgg_arrays.append(arr_v.astype('uint8'))

        trace_indices = list(range(num_plots))

        buttons = [
            dict(label="Original",
                 method="restyle",
                 args=[{"z": original_arrays}, trace_indices]),
            dict(label="High Contrast",
                 method="restyle",
                 args=[{"z": contrast_arrays}, trace_indices]),
            dict(label="Grayscale",
                 method="restyle",
                 args=[{"z": grayscale_arrays}, trace_indices]),
            dict(label="Blur",
                 method="restyle",
                 args=[{"z": blur_arrays}, trace_indices]),
            dict(label="VGG16 Input",
                 method="restyle",
                 args=[{"z": vgg_arrays}, trace_indices]),
        ]

        fig.update_layout(
            updatemenus=[dict(type="buttons",
                              direction="right",
                              x=0.5, xanchor="center",
                              y=1.1, yanchor="top",
                              buttons=buttons)],
            title_text=plot_title,
            height=280 * rows + 120,
            #width=260 * cols,
            margin=dict(t=140)
        )
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        return fig