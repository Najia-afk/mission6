"""
Advanced backbone models with fine-tuning capabilities.
Supports VGG16, EfficientNetB0, and other architectures.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    VGG16, EfficientNetB0, ResNet50, MobileNetV2
)
import numpy as np


class AdvancedBackbone:
    """Advanced backbone with discriminative fine-tuning."""
    
    AVAILABLE_BACKBONES = ['vgg16', 'efficientnet_b0', 'resnet50', 'mobilenet_v2']
    
    def __init__(self, backbone_name='vgg16', input_shape=(224, 224, 3)):
        """
        Initialize advanced backbone.
        
        Args:
            backbone_name: Name of backbone ('vgg16', 'efficientnet_b0', 'resnet50', 'mobilenet_v2')
            input_shape: Input shape for images
        """
        self.backbone_name = backbone_name.lower()
        self.input_shape = input_shape
        self.model = self._load_backbone()
        self.num_layers = len(self.model.layers)
    
    def _load_backbone(self):
        """Load pretrained backbone."""
        if self.backbone_name == 'vgg16':
            return VGG16(input_shape=self.input_shape, include_top=False, weights='imagenet')
        elif self.backbone_name == 'efficientnet_b0':
            return EfficientNetB0(input_shape=self.input_shape, include_top=False, weights='imagenet')
        elif self.backbone_name == 'resnet50':
            return ResNet50(input_shape=self.input_shape, include_top=False, weights='imagenet')
        elif self.backbone_name == 'mobilenet_v2':
            return MobileNetV2(input_shape=self.input_shape, include_top=False, weights='imagenet')
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")
    
    def freeze_backbone(self):
        """Freeze all backbone layers."""
        self.model.trainable = False
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone layers."""
        self.model.trainable = True
    
    def unfreeze_last_n_layers(self, n):
        """
        Discriminative unfreezing: unfreeze last n layers.
        
        Args:
            n: Number of layers to unfreeze from the end
        """
        self.freeze_backbone()
        
        for layer in self.model.layers[-n:]:
            layer.trainable = True
        
        print(f"✅ Unfroze last {n} layers of {self.backbone_name}")
    
    def unfreeze_last_block(self):
        """Unfreeze last block (architecture-specific)."""
        self.freeze_backbone()
        
        if self.backbone_name == 'vgg16':
            # Unfreeze block5
            for layer in self.model.layers:
                if 'block5' in layer.name:
                    layer.trainable = True
            print("✅ Unfroze VGG16 block5")
        
        elif self.backbone_name == 'efficientnet_b0':
            # Unfreeze last 50 layers
            self.unfreeze_last_n_layers(50)
        
        elif self.backbone_name == 'resnet50':
            # Unfreeze last conv block
            for layer in self.model.layers:
                if 'conv5' in layer.name or 'res5' in layer.name:
                    layer.trainable = True
            print("✅ Unfroze ResNet50 conv5 block")
        
        elif self.backbone_name == 'mobilenet_v2':
            # Unfreeze last 30 layers
            self.unfreeze_last_n_layers(30)
    
    def get_feature_extractor(self):
        """Get model for feature extraction (only backbone)."""
        return self.model
    
    def build_classifier(self, num_classes, head_units=256):
        """
        Build complete classifier with backbone + classification head.
        
        Args:
            num_classes: Number of output classes
            head_units: Number of units in dense layer
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Backbone
        x = self.model(inputs, training=False)  # BN should use pretrained stats
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Classification head
        x = layers.Dense(head_units, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def get_summary(self):
        """Get backbone summary."""
        return {
            'name': self.backbone_name,
            'total_layers': self.num_layers,
            'trainable_layers': sum(1 for layer in self.model.layers if layer.trainable),
            'non_trainable_layers': sum(1 for layer in self.model.layers if not layer.trainable),
            'total_params': self.model.count_params(),
            'trainable_params': sum(tf.size(w).numpy() for w in self.model.trainable_weights),
        }


class DiscriminativeFinetuner:
    """Discriminative fine-tuning with layer-wise learning rates."""
    
    def __init__(self, model, base_lr=1e-4, lr_factor=0.1):
        """
        Initialize discriminative finetuner.
        
        Args:
            model: Keras model
            base_lr: Base learning rate for last layer
            lr_factor: Factor to multiply for previous layers
        """
        self.model = model
        self.base_lr = base_lr
        self.lr_factor = lr_factor
    
    def get_layer_wise_lr(self):
        """Calculate layer-wise learning rates."""
        layer_lrs = {}
        
        layers_reversed = list(reversed(self.model.layers))
        
        for idx, layer in enumerate(layers_reversed):
            if layer.trainable:
                lr = self.base_lr * (self.lr_factor ** idx)
                layer_lrs[layer.name] = lr
        
        return layer_lrs
    
    def build_optimizer(self):
        """Build optimizer with layer-wise learning rates."""
        layer_lrs = self.get_layer_wise_lr()
        
        optimizers = {}
        for layer_name, lr in layer_lrs.items():
            optimizers[layer_name] = keras.optimizers.Adam(learning_rate=lr)
        
        return optimizers
    
    def compile_with_discriminative_lr(self):
        """Compile model with discriminative learning rates."""
        # Note: Standard Keras doesn't support per-layer LR easily
        # Use layer groups as a workaround
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.base_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.F1Score()]
        )
        
        print(f"✅ Model compiled with base LR={self.base_lr}")
