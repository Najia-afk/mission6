"""
Image Processing Class for E-commerce Product Classification
Handles image preprocessing, feature extraction, and quality assessment
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class ImageProcessor:
    """
    Comprehensive image processing pipeline for e-commerce products
    """
    
    def __init__(self, target_size=(224, 224), quality_threshold=0.8):
        """
        Initialize the image processor
        
        Args:
            target_size (tuple): Target size for image preprocessing
            quality_threshold (float): Minimum quality threshold for images
        """
        self.target_size = target_size
        self.quality_threshold = quality_threshold
        self.processed_images = []
        self.processing_stats = {}
        self.feature_times = []
        
    def assess_image_quality(self, image_path):
        """
        Assess the quality of an image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Quality metrics
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {'quality_score': 0.0, 'issues': ['Cannot load image']}
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate quality metrics
            height, width = gray.shape
            
            # Blur detection using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Brightness assessment
            brightness = np.mean(gray)
            
            # Contrast assessment
            contrast = np.std(gray)
            
            # Size adequacy
            size_score = min(height, width) / max(self.target_size)
            
            # Overall quality score (normalized)
            quality_score = min(1.0, (
                (blur_score / 1000) * 0.3 +  # Sharpness
                (min(brightness / 128, 1.0)) * 0.2 +  # Brightness
                (min(contrast / 64, 1.0)) * 0.3 +  # Contrast
                min(size_score, 1.0) * 0.2  # Size adequacy
            ))
            
            issues = []
            if blur_score < 100:
                issues.append('Image may be blurry')
            if brightness < 50 or brightness > 200:
                issues.append('Poor lighting conditions')
            if contrast < 30:
                issues.append('Low contrast')
            if min(height, width) < min(self.target_size):
                issues.append('Image resolution too low')
                
            return {
                'quality_score': quality_score,
                'blur_score': blur_score,
                'brightness': brightness,
                'contrast': contrast,
                'size': (height, width),
                'issues': issues
            }
            
        except Exception as e:
            return {'quality_score': 0.0, 'issues': [f'Error: {str(e)}']}
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (processed_image, original_image, processing_info)
        """
        try:
            # Load and assess quality
            original = cv2.imread(image_path)
            if original is None:
                raise ValueError(f"Cannot load image: {image_path}")
                
            quality_info = self.assess_image_quality(image_path)
            
            # Basic preprocessing
            # 1. Resize
            processed = cv2.resize(original, self.target_size)
            
            # 2. Noise reduction
            processed = cv2.medianBlur(processed, 3)
            
            # 3. Normalize
            processed = processed.astype(np.float32) / 255.0
            
            # 4. Enhance contrast if needed
            if quality_info['contrast'] < 30:
                processed = cv2.convertScaleAbs(processed * 255, alpha=1.2, beta=10) / 255.0
            
            processing_info = {
                'original_size': original.shape[:2],
                'processed_size': processed.shape[:2],
                'quality_info': quality_info,
                'preprocessing_applied': ['resize', 'denoise', 'normalize'],
                'success': True
            }
            
            if quality_info['contrast'] < 30:
                processing_info['preprocessing_applied'].append('contrast_enhancement')
            
            return processed, original, processing_info
            
        except Exception as e:
            return None, None, {
                'success': False,
                'error': str(e),
                'preprocessing_applied': []
            }
    
    def extract_basic_features(self, image):
        """
        Extract basic image features (SIFT, LBP, GLCM, Gabor)
        
        Args:
            image (np.ndarray): Preprocessed image
            
        Returns:
            dict: Dictionary of extracted features
        """
        features = {}
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                gray = (image * 255).astype(np.uint8)
            
            # 1. SIFT Features
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            if descriptors is not None and len(descriptors) > 0:
                features['sift_mean'] = np.mean(descriptors, axis=0)
                features['sift_std'] = np.std(descriptors, axis=0)
                features['sift_keypoints_count'] = len(keypoints)
            else:
                features['sift_mean'] = np.zeros(128)
                features['sift_std'] = np.zeros(128)
                features['sift_keypoints_count'] = 0
            
            # 2. Local Binary Pattern (LBP)
            radius = 1
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            features['lbp_histogram'] = lbp_hist.astype(float)
            
            # 3. Gray-Level Co-occurrence Matrix (GLCM)
            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(gray, distances, angles, symmetric=True, normed=True)
            
            features['glcm_contrast'] = graycoprops(glcm, 'contrast').flatten()
            features['glcm_dissimilarity'] = graycoprops(glcm, 'dissimilarity').flatten()
            features['glcm_homogeneity'] = graycoprops(glcm, 'homogeneity').flatten()
            features['glcm_energy'] = graycoprops(glcm, 'energy').flatten()
            
            # 4. Gabor Filters
            gabor_responses = []
            frequencies = [0.1, 0.3, 0.5]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            for freq in frequencies:
                for angle in angles:
                    real, _ = cv2.getGaborKernel((21, 21), 5, angle, 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(gray, cv2.CV_8UC3, real)
                    gabor_responses.append(np.mean(filtered))
                    gabor_responses.append(np.std(filtered))
            
            features['gabor_responses'] = np.array(gabor_responses)
            
            return features
            
        except Exception as e:
            print(f"Error extracting basic features: {e}")
            # Return default features
            return {
                'sift_mean': np.zeros(128),
                'sift_std': np.zeros(128),
                'sift_keypoints_count': 0,
                'lbp_histogram': np.zeros(10),
                'glcm_contrast': np.zeros(4),
                'glcm_dissimilarity': np.zeros(4),
                'glcm_homogeneity': np.zeros(4),
                'glcm_energy': np.zeros(4),
                'gabor_responses': np.zeros(24)
            }
    
    def process_image_batch(self, image_paths, max_images=None):
        """
        Process a batch of images
        
        Args:
            image_paths (list): List of image file paths
            max_images (int): Maximum number of images to process
            
        Returns:
            dict: Processing results and statistics
        """
        if max_images:
            image_paths = image_paths[:max_images]
        
        results = {
            'processed_images': [],
            'basic_features': [],
            'processing_stats': [],
            'successful_paths': [],
            'failed_paths': []
        }
        
        print(f"Processing {len(image_paths)} images...")
        
        for i, img_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            # Preprocess image
            processed, original, processing_info = self.preprocess_image(img_path)
            
            if processing_info['success']:
                # Extract basic features
                basic_features = self.extract_basic_features(processed)
                
                results['processed_images'].append(processed)
                results['basic_features'].append(basic_features)
                results['processing_stats'].append(processing_info)
                results['successful_paths'].append(img_path)
            else:
                results['failed_paths'].append((img_path, processing_info['error']))
                print(f"  Failed: {processing_info['error']}")
        
        # Calculate summary statistics
        success_rate = len(results['successful_paths']) / len(image_paths)
        results['summary'] = {
            'total_images': len(image_paths),
            'successful': len(results['successful_paths']),
            'failed': len(results['failed_paths']),
            'success_rate': success_rate
        }
        
        print(f"\nProcessing complete!")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Successful: {len(results['successful_paths'])}")
        print(f"Failed: {len(results['failed_paths'])}")
        
        return results
    
    def create_feature_matrix(self, basic_features_list):
        """
        Create a feature matrix from basic features
        
        Args:
            basic_features_list (list): List of feature dictionaries
            
        Returns:
            tuple: (feature_matrix, feature_names)
        """
        if not basic_features_list:
            return np.array([]), []
        
        # Combine all features into a single matrix
        feature_matrix = []
        feature_names = []
        
        for features in basic_features_list:
            row = []
            
            # SIFT features (mean only for simplicity)
            sift_summary = [
                np.mean(features['sift_mean']),
                np.std(features['sift_mean']),
                features['sift_keypoints_count']
            ]
            row.extend(sift_summary)
            if not feature_names:  # Only set names once
                feature_names.extend(['sift_mean_avg', 'sift_mean_std', 'sift_keypoints'])
            
            # LBP features (histogram summary)
            lbp_summary = [
                np.mean(features['lbp_histogram']),
                np.std(features['lbp_histogram']),
                np.max(features['lbp_histogram'])
            ]
            row.extend(lbp_summary)
            if len(feature_names) == 3:  # Only set names once
                feature_names.extend(['lbp_mean', 'lbp_std', 'lbp_max'])
            
            # GLCM features (all statistics)
            row.extend(features['glcm_contrast'])
            row.extend(features['glcm_dissimilarity'])
            row.extend(features['glcm_homogeneity'])
            row.extend(features['glcm_energy'])
            if len(feature_names) == 6:  # Only set names once
                feature_names.extend([f'glcm_contrast_{i}' for i in range(4)])
                feature_names.extend([f'glcm_dissimilarity_{i}' for i in range(4)])
                feature_names.extend([f'glcm_homogeneity_{i}' for i in range(4)])
                feature_names.extend([f'glcm_energy_{i}' for i in range(4)])
            
            # Gabor features (summary statistics)
            gabor_summary = [
                np.mean(features['gabor_responses']),
                np.std(features['gabor_responses']),
                np.max(features['gabor_responses']),
                np.min(features['gabor_responses'])
            ]
            row.extend(gabor_summary)
            if len(feature_names) == 22:  # Only set names once
                feature_names.extend(['gabor_mean', 'gabor_std', 'gabor_max', 'gabor_min'])
            
            feature_matrix.append(row)
        
        return np.array(feature_matrix), feature_names
    
    def analyze_features_quality(self, feature_matrix, feature_names):
        """
        Analyze the quality of extracted features
        
        Args:
            feature_matrix (np.ndarray): Feature matrix
            feature_names (list): List of feature names
            
        Returns:
            dict: Feature quality analysis
        """
        if len(feature_matrix) == 0:
            return {'status': 'no_features', 'message': 'No features to analyze'}
        
        analysis = {}
        
        # Basic statistics
        analysis['shape'] = feature_matrix.shape
        analysis['feature_means'] = np.mean(feature_matrix, axis=0)
        analysis['feature_stds'] = np.std(feature_matrix, axis=0)
        analysis['feature_mins'] = np.min(feature_matrix, axis=0)
        analysis['feature_maxs'] = np.max(feature_matrix, axis=0)
        
        # Feature variance analysis
        feature_vars = np.var(feature_matrix, axis=0)
        analysis['low_variance_features'] = [
            feature_names[i] for i in range(len(feature_names)) 
            if feature_vars[i] < 0.01
        ]
        
        # Correlation analysis
        if feature_matrix.shape[0] > 1:
            correlation_matrix = np.corrcoef(feature_matrix.T)
            # Find highly correlated features
            high_corr_pairs = []
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    if abs(correlation_matrix[i, j]) > 0.9:
                        high_corr_pairs.append((feature_names[i], feature_names[j], correlation_matrix[i, j]))
            analysis['high_correlation_pairs'] = high_corr_pairs
        
        # Feature completeness
        missing_data = np.isnan(feature_matrix).sum(axis=0)
        analysis['features_with_missing'] = [
            feature_names[i] for i in range(len(feature_names))
            if missing_data[i] > 0
        ]
        
        # Overall quality score
        quality_factors = []
        quality_factors.append(1.0 - len(analysis['low_variance_features']) / len(feature_names))  # Variance
        quality_factors.append(1.0 - len(analysis['features_with_missing']) / len(feature_names))  # Completeness
        if 'high_correlation_pairs' in analysis:
            quality_factors.append(1.0 - len(analysis['high_correlation_pairs']) / (len(feature_names) * (len(feature_names) - 1) / 2))  # Independence
        
        analysis['overall_quality_score'] = np.mean(quality_factors)
        
        return analysis
    
    def create_processing_dashboard(self, processing_results):
        """
        Create a comprehensive processing dashboard
        
        Args:
            processing_results (dict): Results from process_image_batch
            
        Returns:
            plotly.graph_objects.Figure: Dashboard figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Processing Success Rate',
                'Image Quality Distribution',
                'Feature Extraction Summary',
                'Processing Statistics'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "table"}]
            ]
        )
        
        # 1. Success Rate Indicator
        success_rate = processing_results['summary']['success_rate']
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=success_rate * 100,
                title={'text': "Success Rate (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen" if success_rate > 0.8 else "orange" if success_rate > 0.6 else "red"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # 2. Quality Distribution
        if processing_results['processing_stats']:
            quality_scores = [
                stats['quality_info']['quality_score'] 
                for stats in processing_results['processing_stats']
                if 'quality_info' in stats
            ]
            
            if quality_scores:
                fig.add_trace(
                    go.Histogram(
                        x=quality_scores,
                        nbinsx=20,
                        name='Quality Distribution',
                        marker_color='steelblue'
                    ),
                    row=1, col=2
                )
        
        # 3. Feature Summary
        if processing_results['basic_features']:
            feature_matrix, feature_names = self.create_feature_matrix(processing_results['basic_features'])
            feature_analysis = self.analyze_features_quality(feature_matrix, feature_names)
            
            categories = ['Good Features', 'Low Variance', 'Missing Data', 'High Correlation']
            values = [
                len(feature_names) - len(feature_analysis['low_variance_features']),
                len(feature_analysis['low_variance_features']),
                len(feature_analysis['features_with_missing']),
                len(feature_analysis.get('high_correlation_pairs', []))
            ]
            
            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=values,
                    marker_color=['green', 'orange', 'red', 'blue'],
                    name='Feature Quality'
                ),
                row=2, col=1
            )
        
        # 4. Statistics Table
        stats_data = [
            ['Total Images', processing_results['summary']['total_images']],
            ['Successful', processing_results['summary']['successful']],
            ['Failed', processing_results['summary']['failed']],
            ['Success Rate', f"{processing_results['summary']['success_rate']:.1%}"],
        ]
        
        if processing_results['basic_features']:
            stats_data.extend([
                ['Feature Dimensions', feature_matrix.shape[1]],
                ['Feature Quality', f"{feature_analysis['overall_quality_score']:.2f}"]
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color='lightblue',
                           align='center'),
                cells=dict(values=[[row[0] for row in stats_data],
                                  [row[1] for row in stats_data]],
                          fill_color='white',
                          align='center')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Image Processing Dashboard',
            template='plotly_white',
            showlegend=False,
            width=1000,
            height=600
        )
        
        return fig
