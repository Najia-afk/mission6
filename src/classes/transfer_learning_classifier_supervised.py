"""
Supervised Transfer Learning Classifier
- Moved from TransferLearningClassifier
- Trains/evaluates CNN heads on frozen backbones (VGG16, ResNet50)
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
from sklearn.metrics import adjusted_rand_score, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple, Union, Optional
import time
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from tqdm.notebook import tqdm

# Progress bar callback
class TQDMProgressBar(tf.keras.callbacks.Callback):
    # ...existing code...
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
        if self.batch_pbar is not None:
            self.batch_pbar.close()
            self.batch_pbar = None
        if self.epochs_pbar is not None:
            self.epochs_pbar.update(1)
        logs = logs or {}
        acc = logs.get('accuracy', None)
        loss = logs.get('loss', None)
        val_acc = logs.get('val_accuracy', None)
        val_loss = logs.get('val_loss', None)
        epoch_secs = time.time() - (self.epoch_start_time or time.time())
        avg_ms_per_step = (np.mean(self.batch_times) * 1000.0) if self.batch_times else 0.0
        steps_txt = f"{self.steps_in_epoch}/{self.steps_in_epoch}" if self.steps_in_epoch else ""
        bar_txt = "━━━━━━━━━━━━━━━━━━━━"
        time_txt = f"{int(epoch_secs)}s {int(avg_ms_per_step)}ms/step"
        parts = []
        if acc is not None: parts.append(f"accuracy: {acc:.4f}")
        if loss is not None: parts.append(f"loss: {loss:.4f}")
        if val_acc is not None: parts.append(f"val_accuracy: {val_acc:.4f}")
        if val_loss is not None: parts.append(f"val_loss: {val_loss:.4f}")
        from tqdm import tqdm as _tqdm
        _tqdm.write(f"{steps_txt} {bar_txt} {time_txt} - " + " - ".join(parts))
    def on_train_end(self, logs=None):
        if self.epochs_pbar is not None:
            self.epochs_pbar.close()
            self.epochs_pbar = None


class TransferLearningClassifierSupervised:
    """
    Supervised transfer learning pipeline for image classification.
    API mirrors the original TransferLearningClassifier.
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: Optional[int] = None,
                 base_model_name: str = 'VGG16',
                 weights: str = 'imagenet',
                 use_gpu: bool = True,
                 random_state: int = 42,
                 model_dir: str = 'models'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model_name = base_model_name
        self.weights = weights
        self.random_state = random_state
        self.model_dir = model_dir
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and use_gpu:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Error setting up GPU: {e}")
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
        self.models = {}
        self.histories = {}
        self.evaluation_results = {}
        print(f"🔧 Transfer Learning Classifier (supervised) initialized")

    def prepare_data_from_dataframe(self, df: pd.DataFrame, image_column: str, category_column: str,
                                    test_size: float = 0.2, val_size: float = 0.2,
                                    random_state: int = 42, image_dir: Optional[str] = None) -> Dict[str, Any]:
        print("🔄 Preparing data from DataFrame...")
        df = df.copy()
        final_image_dir = image_dir
        if not final_image_dir:
            default_dir = 'dataset/Flipkart/Images'
            if os.path.exists(default_dir):
                final_image_dir = default_dir
                print(f"   📁 Using default image directory: {final_image_dir}")
        if final_image_dir:
            df['image_path'] = df[image_column].apply(lambda p: os.path.join(final_image_dir, os.path.basename(p)))
        else:
            df['image_path'] = df[image_column]
            print("   ⚠️ No image directory specified or found. Using raw paths from DataFrame.")
        self.label_encoder = LabelEncoder()
        df['label_encoded'] = self.label_encoder.fit_transform(df[category_column])
        self.class_names = self.label_encoder.classes_
        if self.num_classes is None:
            self.num_classes = len(self.class_names)
        train_val_df, self.test_df = train_test_split(df, test_size=test_size, random_state=random_state,
                                                      stratify=df[category_column])
        self.train_df, self.val_df = train_test_split(train_val_df, test_size=val_size, random_state=random_state,
                                                      stratify=train_val_df[category_column])
        print(f"   📋 Categories found: {self.class_names.tolist()}")
        print(f"   🎯 Number of classes: {self.num_classes}")
        print(f"   📊 Train/Val/Test: {len(self.train_df)}/{len(self.val_df)}/{len(self.test_df)}")
        return {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "train_size": len(self.train_df),
            "val_size": len(self.val_df),
            "test_size": len(self.test_df)
        }

    def _load_and_preprocess_images(self, df: pd.DataFrame, image_column: str = 'image_path') -> Tuple[np.ndarray, np.ndarray]:
        images, labels = [], []
        if len(df) == 0:
            print("   ⚠️ Warning: Empty DataFrame provided")
            return np.array([]), np.array([])
        print(f"   🖼️ Loading {len(df)} images...")
        failed = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading & preprocessing", unit="img"):
            try:
                path = row[image_column]
                if not os.path.exists(path):
                    alt = os.path.join('dataset/Flipkart/Images', os.path.basename(path))
                    if os.path.exists(alt): path = alt
                if not os.path.exists(path):
                    failed += 1; continue
                img = load_img(path, target_size=self.input_shape[:2])
                arr = img_to_array(img)
                if self.base_model_name == 'VGG16':
                    arr = vgg16_preprocess(arr)
                else:
                    arr = arr / 255.0
                images.append(arr)
                labels.append(row['label_encoded'])
            except Exception:
                failed += 1
        if len(images) == 0:
            print(f"   ❌ No images loaded ({failed} failures)")
            return np.array([]), np.array([])
        print(f"   ✅ Loaded {len(images)} images ({failed} failures)")
        return np.array(images), to_categorical(labels, num_classes=self.num_classes)

    def prepare_arrays_method(self) -> Dict[str, Any]:
        print("🔄 Preparing data using arrays method...")
        self.X_train, self.y_train = self._load_and_preprocess_images(self.train_df)
        self.X_val, self.y_val = self._load_and_preprocess_images(self.val_df)
        self.X_test, self.y_test = self._load_and_preprocess_images(self.test_df)
        return {
            "X_train_shape": self.X_train.shape if self.X_train.size else (0,),
            "X_val_shape": self.X_val.shape if self.X_val.size else (0,),
            "X_test_shape": self.X_test.shape if self.X_test.size else (0,)
        }

    def create_base_model(self, show_backbone_summary: bool = False) -> tf.keras.Model:
        print(f"🔧 Creating base model with {self.base_model_name}...")
        if self.base_model_name == 'VGG16':
            base_model = VGG16(weights=self.weights, include_top=False, input_shape=self.input_shape,
                               name=self.base_model_name.lower())
        elif self.base_model_name == 'ResNet50':
            base_model = ResNet50(weights=self.weights, include_top=False, input_shape=self.input_shape,
                                  name=self.base_model_name.lower())
        else:
            raise ValueError(f"Unsupported base model: {self.base_model_name}")
        base_model.trainable = False
        if show_backbone_summary:
            print("=== Backbone Summary (Frozen) ===")
            base_model.summary(line_length=80)
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("   ✅ Base model created and compiled.")
        return model

    def _calculate_ari_score(self, model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> float:
        if X_test is None or y_test is None or X_test.size == 0:
            return float('nan')
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        return adjusted_rand_score(y_true_classes, y_pred_classes)

    def create_augmented_model(self) -> tf.keras.Model:
        print(f"🔧 Creating augmented model with {self.base_model_name} for fine-tuning...")
        model = self.create_base_model()
        try:
            base_model_layer = model.get_layer(self.base_model_name.lower())
        except ValueError:
            print(f"   ❌ Could not find layer '{self.base_model_name.lower()}'.")
            return model
        if self.base_model_name == 'VGG16':
            for layer in base_model_layer.layers[-4:]:
                layer.trainable = True
        elif self.base_model_name == 'ResNet50':
            for layer in base_model_layer.layers[-10:]:
                layer.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        print("   ✅ Model re-compiled for fine-tuning.")
        return model

    def train_model(self, model_name: str, model: tf.keras.Model,
                    epochs: int = 20, batch_size: int = 32, patience: int = 5,
                    use_generators: bool = False) -> Dict[str, Any]:
        print(f"🔄 Training model: {model_name}...")
        if self.X_train.size == 0 or self.X_val.size == 0:
            print("   ❌ No training/validation data.")
            return {'error': 'No data', 'model': model, 'history': None,
                    'evaluation': {'loss': float('nan'), 'accuracy': float('nan')}, 'training_time': 0}
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"{model_name}_best.keras")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
            ModelCheckpoint(filepath=model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
            TQDMProgressBar()
        ]
        start = time.time()
        if use_generators:
            train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                               shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
            val_datagen = ImageDataGenerator()
            train_gen = train_datagen.flow(self.X_train, self.y_train, batch_size=batch_size)
            val_gen = val_datagen.flow(self.X_val, self.y_val, batch_size=batch_size)
            history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks, verbose=0)
        else:
            history = model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val),
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
        if os.path.exists(model_path):
            model.load_weights(model_path)
        training_time = time.time() - start
        if self.X_test.size > 0:
            evaluation = model.evaluate(self.X_test, self.y_test, verbose=0)
            ari_score = self._calculate_ari_score(model, self.X_test, self.y_test)
        else:
            evaluation = [float('nan'), float('nan')]
            ari_score = float('nan')
        self.models[model_name] = model
        self.histories[model_name] = history.history
        self.evaluation_results[model_name] = {
            'loss': evaluation[0], 'accuracy': evaluation[1],
            'training_time': training_time, 'ari_score': ari_score
        }
        print(f"✅ Training completed in {training_time:.2f}s")
        print(f"   📊 Test accuracy: {evaluation[1]:.4f}")
        print(f"   📊 ARI Score: {ari_score:.4f}")
        return {'model': model, 'history': history.history,
                'evaluation': {'loss': evaluation[0], 'accuracy': evaluation[1]},
                'training_time': training_time}

    def compare_models(self) -> go.Figure:
        if not self.evaluation_results:
            print("⚠️ No models to compare."); return None
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Test Accuracy', 'Training Time'),
                            specs=[[{'type': 'bar'}, {'type': 'bar'}]])
        names = list(self.evaluation_results.keys())
        accs = [v['accuracy'] * 100 for v in self.evaluation_results.values()]
        times = [v['training_time'] for v in self.evaluation_results.values()]
        fig.add_trace(go.Bar(x=names, y=accs, marker_color='royalblue',
                             text=[f"{a:.2f}%" for a in accs], textposition='auto'), 1, 1)
        fig.add_trace(go.Bar(x=names, y=times, marker_color='lightgreen',
                             text=[f"{t:.2f}s" for t in times], textposition='auto'), 1, 2)
        fig.update_layout(title='Model Comparison', height=500, width=900, showlegend=False)
        return fig

    def plot_training_history(self, model_name: Optional[str] = None) -> go.Figure:
        if not self.histories:
            print("⚠️ No training history found."); return None
        # Select model if not provided: prefer lowest val_loss, fallback to highest val_accuracy
        def best_by_val_loss():
            best, best_val = None, float('inf')
            for name, h in self.histories.items():
                if 'val_loss' in h and len(h['val_loss']):
                    cur = min(h['val_loss'])
                    if cur < best_val:
                        best, best_val = name, cur
            return best
        def best_by_val_acc():
            best, best_val = None, -1.0
            for name, h in self.histories.items():
                if 'val_accuracy' in h and len(h['val_accuracy']):
                    cur = max(h['val_accuracy'])
                    if cur > best_val:
                        best, best_val = name, cur
            return best
        model_name = model_name or best_by_val_loss() or best_by_val_acc() or list(self.histories.keys())[0]
        h = self.histories.get(model_name, None)
        if h is None:
            print(f"⚠️ Unknown model '{model_name}'."); return None
        epochs = list(range(1, len(h.get('loss', [])) + 1))
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'Accuracy'))
        fig.add_trace(go.Scatter(x=epochs, y=h.get('loss', []), name='Train Loss', mode='lines'), 1, 1)
        fig.add_trace(go.Scatter(x=epochs, y=h.get('val_loss', []), name='Val Loss', mode='lines'), 1, 1)
        fig.add_trace(go.Scatter(x=epochs, y=h.get('accuracy', []), name='Train Acc', mode='lines'), 1, 2)
        fig.add_trace(go.Scatter(x=epochs, y=h.get('val_accuracy', []), name='Val Acc', mode='lines'), 1, 2)
        fig.update_layout(title=f"Training History: {model_name}", height=500, width=900)
        fig.update_xaxes(title_text='Epoch', row=1, col=1)
        fig.update_xaxes(title_text='Epoch', row=1, col=2)
        fig.update_yaxes(title_text='Loss', row=1, col=1)
        fig.update_yaxes(title_text='Accuracy', row=1, col=2)
        return fig

    def plot_confusion_matrix(self, model_name: str) -> go.Figure:
        if model_name not in self.models:
            print(f"⚠️ Model '{model_name}' not found."); return None
        if self.X_test is None or self.y_test is None or self.X_test.size == 0:
            print("⚠️ No test data to compute confusion matrix."); return None
        model = self.models[model_name]
        y_prob = model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        ztext = [[str(v) for v in row] for row in cm]
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=self.class_names, y=self.class_names, colorscale='Blues',
            text=ztext, texttemplate="%{text}", hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>"
        ))
        fig.update_layout(
            title=f"Confusion Matrix: {model_name}",
            xaxis_title="Predicted",
            yaxis_title="True",
            yaxis_autorange='reversed',
            height=600, width=700
        )
        return fig

    def get_summary(self) -> Dict[str, Any]:
        if not self.evaluation_results:
            return {
                'num_models': 0,
                'models': {},
                'best_model': None,
                'num_classes': self.num_classes,
                'class_names': self.class_names.tolist() if self.class_names is not None else []
            }
        # Determine best by min val_loss across histories
        best_name = None
        best_val_loss = float('inf')
        best_val_acc = -1.0
        for name, h in self.histories.items():
            if 'val_loss' in h and len(h['val_loss']):
                v = min(h['val_loss'])
                if v < best_val_loss:
                    best_val_loss = v
                    best_name = name
        if best_name is None:
            # fallback to best val_accuracy
            for name, h in self.histories.items():
                if 'val_accuracy' in h and len(h['val_accuracy']):
                    v = max(h['val_accuracy'])
                    if v > best_val_acc:
                        best_val_acc = v
                        best_name = name
        # Gather best metrics
        best_eval = self.evaluation_results.get(best_name, {}) if best_name else {}
        summary = {
            'num_models': len(self.evaluation_results),
            'models': self.evaluation_results,
            'best_model': {
                'name': best_name,
                'val_loss': best_val_loss if best_name else None,
                'val_accuracy': (max(self.histories[best_name]['val_accuracy'])
                                 if best_name and 'val_accuracy' in self.histories[best_name] and len(self.histories[best_name]['val_accuracy']) else None),
                'test_accuracy': best_eval.get('accuracy', None),
                'ari_score': best_eval.get('ari_score', None),
            } if best_name else None,
            'num_classes': self.num_classes,
            'class_names': self.class_names.tolist() if self.class_names is not None else []
        }
        return summary

    def plot_prediction_examples(self, model_name: str, num_correct: int = 4, num_incorrect: int = 4,
                                 uniq_id: Optional[str] = None) -> go.Figure:
        if model_name not in self.models:
            print(f"⚠️ Model '{model_name}' not found."); return None
        if self.X_test is None or self.y_test is None or self.X_test.size == 0 or self.test_df is None:
            print("⚠️ Missing test data for examples."); return None
        model = self.models[model_name]
        y_prob = model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        correct_idx = np.where(y_pred == y_true)[0].tolist()
        incorrect_idx = np.where(y_pred != y_true)[0].tolist()
        correct_idx = correct_idx[:num_correct]
        incorrect_idx = incorrect_idx[:num_incorrect]
        indices = correct_idx + incorrect_idx
        if not indices:
            print("⚠️ No examples to show."); return None
        n = len(indices)
        cols = min(4, n)
        rows = int(np.ceil(n / cols))
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[
            f"T:{self.class_names[y_true[i]]} | P:{self.class_names[y_pred[i]]}" for i in indices
        ])
        def load_rgb(path):
            try:
                with Image.open(path) as im:
                    return np.array(im.convert('RGB'))
            except Exception:
                return np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8)
        for idx, i in enumerate(indices):
            r = idx // cols + 1
            c = idx % cols + 1
            path = self.test_df.iloc[i]['image_path']
            img = load_rgb(path)
            fig.add_trace(go.Image(z=img), row=r, col=c)
            # border color: green for correct, red for incorrect
            color = 'green' if i in correct_idx else 'red'
            fig.update_xaxes(showticklabels=False, row=r, col=c)
            fig.update_yaxes(showticklabels=False, row=r, col=c)
            fig.layout.annotations[idx].font = dict(color=color)
        fig.update_layout(height=250 * rows, width=250 * cols,
                          title=f"Prediction Examples: {model_name}")
        return fig
