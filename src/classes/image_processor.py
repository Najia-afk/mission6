import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def load_and_preprocess(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)
        return image

    def enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    def process_batch(self, image_paths):
        processed_images = []
        for path in image_paths:
            img = self.load_and_preprocess(path)
            if img is not None:
                processed_images.append(img)
        return {
            'processed_images': processed_images,
            'success_rate': len(processed_images) / len(image_paths) * 100
        }

    def demonstrate_processing_steps(self, image_path):
        original = cv2.imread(str(image_path))
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        preprocessed = self.load_and_preprocess(image_path)
        enhanced = self.enhance_contrast(preprocessed)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_rgb)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(preprocessed)
        axes[1].set_title('Preprocessed')
        axes[1].axis('off')

        axes[2].imshow(enhanced)
        axes[2].set_title('Contrast Enhanced')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
