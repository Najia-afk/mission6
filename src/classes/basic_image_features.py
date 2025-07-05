import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray

class BasicImageFeatureExtractor:
    def __init__(self, sift_features=128, lbp_radius=1, lbp_points=8):
        self.sift = cv2.SIFT_create(nfeatures=sift_features)
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points

    def extract_sift_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, des = self.sift.detectAndCompute(gray, None)
        if des is not None:
            return des.mean(axis=0)
        return np.zeros(self.sift.descriptorSize())

    def extract_lbp_features(self, image):
        gray = rgb2gray(image)
        lbp = local_binary_pattern(gray, self.lbp_points, self.lbp_radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_points + 3), range=(0, self.lbp_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist

    def extract_hog_features(self, image):
        gray = rgb2gray(image)
        features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
        return features

    def extract_features_batch(self, images):
        features = []
        for img in images:
            sift_feats = self.extract_sift_features(img)
            lbp_feats = self.extract_lbp_features(img)
            # HOG features can be very high dimensional, so we might want to be careful
            # hog_feats = self.extract_hog_features(img)
            combined = np.hstack([sift_feats, lbp_feats])
            features.append(combined)
        return np.array(features)
