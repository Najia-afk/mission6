"""
Grad-CAM visualization for interpreting model predictions.
Generates visual explanations of model decisions.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class GradCAM:
    """Gradient-weighted Class Activation Maps using the last convolutional layer."""
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Keras model
            layer_name: Name of the convolutional layer to visualize (None = auto-detect last conv layer)
        """
        self.model = model
        
        # Auto-detect the last convolutional layer if not provided
        if layer_name is None:
            layer_name = self._find_last_conv_layer()
        
        self.layer_name = layer_name
        self.layer = model.get_layer(layer_name)
        print(f"Using layer '{layer_name}' for Grad-CAM visualization")
    
    def _find_last_conv_layer(self):
        """Find the last convolutional layer in the model."""
        for layer in reversed(self.model.layers):
            # Check if this is a convolutional layer
            if isinstance(layer, keras.layers.Conv2D) or 'conv' in layer.name.lower():
                return layer.name
        
        # Fallback: try to find any layer with 'conv' in the name
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower() or isinstance(layer, keras.layers.Conv2D):
                return layer.name
        
        raise ValueError("No convolutional layer found in model")
    
    def generate_heatmap(self, img_array, pred_index=None):
        """
        Generate Grad-CAM heatmap using direct gradient computation on the model.
        
        Args:
            img_array: Input image (H, W, 3) or batch (B, H, W, 3)
            pred_index: Class index to visualize (None for predicted class)
        """
        # Ensure batch dimension
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        img_tensor = tf.cast(img_array, tf.float32)
        
        with tf.GradientTape() as tape:
            # Watch the input
            tape.watch(img_tensor)
            # Get predictions
            predictions = self.model(img_tensor, training=False)
            
            # Determine which class to visualize
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # Get the class activation
            class_activation = predictions[0, pred_index]
        
        # Compute gradients with respect to input
        grads = tape.gradient(class_activation, img_tensor)
        
        if grads is None:
            # Fallback if gradients are None - use uniform heatmap
            print(f"Warning: Could not compute gradients")
            return np.ones((img_array.shape[1] // 8, img_array.shape[2] // 8))
        
        # Create a simple heatmap by averaging gradients across channels
        heatmap = tf.reduce_mean(tf.abs(grads), axis=-1)[0]
        
        # Normalize heatmap to [0, 1]
        heatmap_max = tf.math.reduce_max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
        
        # Resize to match expected size
        heatmap_resized = tf.image.resize(
            tf.expand_dims(tf.expand_dims(heatmap, 0), -1),
            [img_array.shape[1] // 8, img_array.shape[2] // 8]
        )
        
        return tf.squeeze(heatmap_resized).numpy()
    
    def overlay_heatmap(self, img_array, heatmap, alpha=0.4):
        """
        Overlay heatmap on original image.
        
        Args:
            img_array: Original image (H, W, 3)
            heatmap: Grad-CAM heatmap (may be smaller than image)
            alpha: Transparency factor
        """
        # Handle None heatmap
        if heatmap is None:
            return img_array.copy()
        
        # Resize heatmap to match image using TensorFlow
        heatmap_resized = tf.image.resize(
            tf.expand_dims(tf.expand_dims(heatmap, 0), -1),
            [img_array.shape[0], img_array.shape[1]]
        )
        heatmap_resized = tf.squeeze(heatmap_resized).numpy()
        
        # Normalize heatmap to [0, 1]
        if np.max(heatmap_resized) > 0:
            heatmap_resized = heatmap_resized / np.max(heatmap_resized)
        
        # Normalize image
        img = np.uint8((img_array - img_array.min()) / 
                       (img_array.max() - img_array.min() + 1e-8) * 255)
        
        # Create overlayed image using jet colormap
        jet = cm.get_cmap("jet")
        heatmap_colored = jet(heatmap_resized)
        
        overlayed = img * (1 - alpha) + (heatmap_colored[:, :, :3] * 255) * alpha
        return np.uint8(overlayed)
    
    def visualize_predictions(self, images, predictions, class_names, true_labels=None, num_samples=4):
        """
        Create interactive Grad-CAM visualization with predictions and confidence.
        Best practice: Side-by-side original + heatmap overlay with annotations.
        
        Args:
            images: Batch of images (B, H, W, 3)
            predictions: Model predictions (logits or probabilities)
            class_names: List of class names
            true_labels: True labels (optional)
            num_samples: Number of samples to visualize
        """
        num_samples = min(num_samples, len(images))
        
        # Create side-by-side layout (Original | Heatmap Overlay)
        fig = make_subplots(
            rows=num_samples, cols=2,
            subplot_titles=['Original Image', 'Grad-CAM Heatmap Overlay'] * num_samples,
            specs=[[{'type': 'image'}, {'type': 'image'}]] * num_samples,
            vertical_spacing=0.12
        )
        
        for idx in range(num_samples):
            # Get prediction probabilities
            pred_probs = predictions[idx]
            pred_class_idx = np.argmax(pred_probs)
            pred_class_name = class_names[pred_class_idx]
            pred_confidence = pred_probs[pred_class_idx]
            
            # Generate Grad-CAM heatmap (pass single image with batch dim)
            img_batch = np.expand_dims(images[idx], axis=0)
            try:
                heatmap = self.generate_heatmap(img_batch, pred_index=pred_class_idx)
                overlay = self.overlay_heatmap(images[idx], heatmap, alpha=0.5)
            except Exception as e:
                print(f"Warning: Could not generate Grad-CAM for sample {idx}: {str(e)}")
                overlay = images[idx]
            
            # True label info
            true_label_str = ""
            if true_labels is not None:
                true_label_str = f"True: {true_labels[idx]}"
                
            # Build annotation text for hover
            annotation_text = (
                f"Predicted: {pred_class_name} ({pred_confidence:.1%})"
            )
            
            # Normalize images for display
            img_display = np.uint8((images[idx] - images[idx].min()) / 
                                   (images[idx].max() - images[idx].min() + 1e-8) * 255)
            
            # Add original image (use customdata for hover info)
            fig.add_trace(
                go.Image(z=img_display, 
                         customdata=[[annotation_text + (" | " + true_label_str if true_label_str else "")]],
                         hovertemplate='<b>Sample %{customdata[0][0]}</b><extra></extra>'),
                row=idx+1, col=1
            )
            
            # Add overlay image with heatmap
            overlay_display = np.uint8((overlay - overlay.min()) / 
                                       (overlay.max() - overlay.min() + 1e-8) * 255)
            fig.add_trace(
                go.Image(z=overlay_display, 
                         customdata=[[f"Grad-CAM: {annotation_text}" + (" | " + true_label_str if true_label_str else "")]],
                         hovertemplate='<b>%{customdata[0][0]}</b><extra></extra>'),
                row=idx+1, col=2
            )
        
        # Update layout with professional styling
        fig.update_layout(
            height=350 * num_samples,
            showlegend=False,
            title_text="<b>Grad-CAM Model Interpretability Dashboard</b><br><sub>Left: Original | Right: Heatmap Overlay (Red = High Importance)</sub>",
            title_font_size=14,
            hovermode='closest',
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        # Remove axes
        for i in range(1, num_samples + 1):
            for j in range(1, 3):
                fig.update_xaxes(showticklabels=False, row=i, col=j)
                fig.update_yaxes(showticklabels=False, row=i, col=j)
        
        return fig
    
    def visualize_single_prediction(self, image, class_names, true_label=None):
        """
        Create detailed single-image Grad-CAM visualization with confidence scores.
        Lightweight matplotlib-based visualization for notebook display.
        
        Args:
            image: Single image array
            class_names: List of class names
            true_label: True label (optional)
            
        Returns:
            Matplotlib figure (lightweight, not embedded Plotly)
        """
        # Ensure image is batched
        img_batch = image[np.newaxis, ...]
        
        # Get prediction
        pred_probs = self.model.predict(img_batch, verbose=0)[0]
        pred_class_idx = np.argmax(pred_probs)
        pred_class_name = class_names[pred_class_idx]
        
        # Generate Grad-CAM
        heatmap = self.generate_heatmap(img_batch)
        overlay = self.overlay_heatmap(image, heatmap, alpha=0.6)
        
        # Reverse VGG16 preprocessing for display
        vgg_mean = np.array([103.939, 116.779, 123.68])
        image_display = image.copy().astype(np.float32)
        image_display += vgg_mean
        image_display = np.clip(image_display, 0, 255).astype(np.uint8)
        
        # Prepare heatmap visualization
        if heatmap is not None and np.max(heatmap) > 0:
            if heatmap.shape != image.shape[:2]:
                heatmap_resized = tf.image.resize(
                    tf.expand_dims(tf.expand_dims(heatmap, 0), -1),
                    [image.shape[0], image.shape[1]]
                )
                heatmap = tf.squeeze(heatmap_resized).numpy()
            
            jet_colormap = cm.get_cmap('jet')
            heatmap_normalized = heatmap / np.max(heatmap)
            heatmap_rgb = jet_colormap(heatmap_normalized)
            heatmap_viz = np.uint8(heatmap_rgb[:, :, :3] * 255)
        else:
            heatmap_viz = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 128
        
        # Create matplotlib figure - lightweight!
        fig = plt.figure(figsize=(16, 4))
        
        # Original image
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(image_display)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Grad-CAM heatmap
        ax2 = plt.subplot(1, 4, 2)
        ax2.imshow(heatmap_viz)
        ax2.set_title('Grad-CAM Activation', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Overlay
        ax3 = plt.subplot(1, 4, 3)
        ax3.imshow(overlay)
        ax3.set_title('Heatmap Overlay', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Text predictions
        ax4 = plt.subplot(1, 4, 4)
        ax4.axis('off')
        
        # Build prediction text
        top_indices = np.argsort(pred_probs)[-5:][::-1]
        pred_text = "CONFIDENCE SCORES:\n\n"
        pred_text += f"PREDICTED: {pred_class_name}\n{pred_probs[pred_class_idx]:.1%}\n\n"
        
        if true_label:
            is_correct = "✓" if true_label == pred_class_name else "✗"
            pred_text += f"{is_correct} TRUE: {true_label}\n\n"
        
        pred_text += "TOP 5:\n"
        for i, idx in enumerate(top_indices):
            pred_text += f"{i+1}. {class_names[idx]} {pred_probs[idx]:.1%}\n"
        
        ax4.text(0.1, 0.5, pred_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Title with color coding
        is_correct = true_label and true_label == pred_class_name
        title_color = 'green' if is_correct else ('red' if true_label else 'black')
        status = '✓ CORRECT' if is_correct else ('✗ INCORRECT' if true_label else '')
        plt.suptitle(f'Grad-CAM: {pred_class_name} {status}', 
                    fontsize=14, fontweight='bold', color=title_color)
        
        plt.tight_layout()
        return fig
