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
from typing import Dict, Any, List, Tuple  # Add this import for type hints
from tqdm.notebook import tqdm  # Import tqdm for progress bars
warnings.filterwarnings('ignore')


class ImageProcessor:
    """
    Comprehensive image processing pipeline for e-commerce products
    """
    
    def __init__(self, target_size=(224, 224), quality_threshold=0.65):
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
        self.feature_cache = {}
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
    
    def extract_basic_features(self, image_input) -> Dict[str, Any]:
        """
        Extract basic features from an image using ORB descriptors and color histograms.
        
        Args:
            image_input: Either a path to image file (str) or processed image array (np.ndarray)
            
        Returns:
            Dictionary containing extracted features and metadata
        """
        try:
            # Handle both file path and processed image array
            if isinstance(image_input, str):
                # Load image from path
                image = cv2.imread(image_input)
                if image is None:
                    return {"success": False, "error": "Could not load image"}
            elif isinstance(image_input, np.ndarray):
                # Use processed image array
                image = image_input
                # Convert from float to uint8 if necessary
                if image.dtype == np.float32 or image.dtype == np.float64:
                    image = (image * 255).astype(np.uint8)
            else:
                return {"success": False, "error": "Invalid image input type"}
            
            # Ensure image has the right shape
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB if needed
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 2:
                gray = image
            else:
                return {"success": False, "error": "Invalid image format"}
            
            # Initialize ORB detector
            orb = cv2.ORB_create(nfeatures=500)
            
            # Detect keypoints and compute descriptors
            try:
                keypoints, descriptors = orb.detectAndCompute(gray, None)
            except Exception as e:
                print(f"ORB detection failed: {e}")
                keypoints, descriptors = [], None
            
            # Extract color histogram features (HSV) - only if color image
            if len(image.shape) == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
                hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
                hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
                
                # Flatten histograms
                color_features = np.concatenate([
                    hist_h.flatten(),
                    hist_s.flatten(),
                    hist_v.flatten()
                ])
            else:
                # Grayscale image - create dummy color features
                color_features = np.zeros(170)  # 50 + 60 + 60
            
            # Basic image statistics
            basic_stats = np.array([
                image.shape[0],  # height
                image.shape[1],  # width
                image.shape[2] if len(image.shape) > 2 else 1,  # channels
                np.mean(image),  # mean intensity
                np.std(image),   # std intensity
                len(keypoints)   # number of keypoints
            ])
            
            # Combine all features
            if descriptors is not None and len(descriptors) > 0:
                # Use bag of visual words approach - take mean of descriptors
                descriptor_features = np.mean(descriptors, axis=0)
                # If we have fewer than expected descriptors, pad with zeros
                if len(descriptor_features) < 32:
                    descriptor_features = np.pad(descriptor_features, 
                                            (0, 32 - len(descriptor_features)), 
                                            'constant')
                else:
                    descriptor_features = descriptor_features[:32]  # Truncate if too long
            else:
                # No descriptors found, use zeros
                descriptor_features = np.zeros(32)
            
            # Combine all features
            all_features = np.concatenate([
                basic_stats,
                color_features,
                descriptor_features
            ])
            
            return {
                "success": True,
                "features": all_features,
                "feature_names": self._get_feature_names(),
                "keypoints_count": len(keypoints),
                "image_shape": image.shape
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_feature_names(self) -> List[str]:
        """Generate feature names for basic features"""
        names = []
        
        # Basic stats
        names.extend(['height', 'width', 'channels', 'mean_intensity', 'std_intensity', 'keypoints_count'])
        
        # Color histograms
        names.extend([f'hist_h_{i}' for i in range(50)])
        names.extend([f'hist_s_{i}' for i in range(60)])
        names.extend([f'hist_v_{i}' for i in range(60)])
        
        # ORB descriptors
        names.extend([f'orb_desc_{i}' for i in range(32)])
        
        return names
    
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
        
        # Use tqdm for progress tracking instead of individual prints
        for img_path in tqdm(image_paths, desc="Processing images", unit="img"):
            # Preprocess image
            processed, original, processing_info = self.preprocess_image(img_path)
            
            if processing_info['success']:
                # Extract basic features from the processed image
                basic_features = self.extract_basic_features(processed)
                
                results['processed_images'].append(processed)
                results['basic_features'].append(basic_features)
                results['processing_stats'].append(processing_info)
                results['successful_paths'].append(img_path)
            else:
                results['failed_paths'].append((img_path, processing_info['error']))
        
        # Calculate summary statistics
        success_rate = len(results['successful_paths']) / len(image_paths) if image_paths else 0
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
        
        # Filter out failed feature extractions
        successful_features = [f for f in basic_features_list if f.get('success', False)]
        
        if not successful_features:
            print("No successful feature extractions found")
            return np.array([]), []
        
        # Extract features and names from the first successful extraction
        first_features = successful_features[0]
        feature_names = first_features.get('feature_names', [])
        
        # Build feature matrix
        feature_matrix = []
        
        for features_dict in successful_features:
            if features_dict.get('success', False):
                feature_vector = features_dict.get('features', [])
                if len(feature_vector) > 0:
                    feature_matrix.append(feature_vector)
        
        if not feature_matrix:
            print("No valid feature vectors found")
            return np.array([]), []
        
        feature_matrix = np.array(feature_matrix)
        
        # Ensure feature names match matrix dimensions
        if len(feature_names) != feature_matrix.shape[1]:
            print(f"Warning: Feature names ({len(feature_names)}) don't match matrix columns ({feature_matrix.shape[1]})")
            # Generate generic names if mismatch
            feature_names = [f'feature_{i}' for i in range(feature_matrix.shape[1])]
        
        print(f"Created feature matrix: {feature_matrix.shape}")
        print(f"Feature names: {len(feature_names)}")
        
        return feature_matrix, feature_names
    
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
                '',
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
    
    def ensure_sample_images(self, image_dir='dataset/Flipkart/Images', num_samples=20, random_seed=None):
        """
        Ensures that sample images exist for demonstration purposes.
        Creates sample images of different product categories if directory doesn't exist.
        
        Args:
            image_dir (str): Path to the image directory
            num_samples (int): Number of sample images to create
            random_seed (int): Random seed for reproducible selection of images
            
        Returns:
            dict: Information about available images
        """
        # Set random seed if provided
        if random_seed is not None:
            import random
            random.seed(random_seed)
            
        # Get list of available images
        available_images = [f for f in os.listdir(image_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Randomize the list if requested
        if random_seed is not None:
            import random
            random.shuffle(available_images)
            # Limit to requested number of samples if needed
            if num_samples and num_samples < len(available_images):
                available_images = available_images[:num_samples]
        
        return {
            'image_dir': image_dir,
            'available_images': available_images,
            'count': len(available_images),
            'sample_created': not os.path.exists(image_dir),
            'random_seed_used': random_seed
        }