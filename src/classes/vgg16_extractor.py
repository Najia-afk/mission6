import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

class VGG16FeatureExtractor:
    def __init__(self, input_shape=(224, 224, 3), layer_name='block5_pool'):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

    def extract_features(self, images, batch_size=32):
        num_images = len(images)
        features = []
        for i in range(0, num_images, batch_size):
            batch_images = images[i:i+batch_size]
            batch_images_np = np.array([img for img in batch_images])
            preprocessed_images = preprocess_input(batch_images_np)
            batch_features = self.model.predict(preprocessed_images)
            batch_features_flat = batch_features.reshape((batch_features.shape[0], -1))
            features.extend(batch_features_flat)
        return np.array(features)
