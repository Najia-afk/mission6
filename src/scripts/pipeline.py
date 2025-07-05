import pandas as pd
from pathlib import Path
import json

from ..classes.preprocess_text import TextPreprocessor
from ..classes.encode_text import TextEncoder
from ..classes.reduce_dimensions import DimensionalityReducer
from ..classes.advanced_embeddings import AdvancedTextEmbeddings
from ..classes.image_processor import ImageProcessor
from ..classes.basic_image_features import BasicImageFeatureExtractor
from ..classes.vgg16_extractor import VGG16FeatureExtractor
from ..classes.multimodal_fusion import MultimodalFusion

class Mission6Pipeline:
    def __init__(self, data_path='dataset/Flipkart', output_path='output', max_images=100, max_text_samples=1000):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.max_images = max_images
        self.max_text_samples = max_text_samples
        self.output_path.mkdir(exist_ok=True)

    def run_pipeline(self):
        # Load data
        df = pd.read_csv(self.data_path / 'flipkart_com-ecommerce_sample_1050.csv')
        df = df.head(self.max_text_samples)

        # Text processing
        text_processor = TextPreprocessor()
        df['product_name_processed'] = df['product_name'].apply(text_processor.preprocess)
        df['product_category'] = df['product_category_tree'].apply(text_processor.extract_top_category)

        # Text encoding
        encoder = TextEncoder()
        encoding_results = encoder.fit_transform(df['product_name_processed'])

        # Advanced embeddings
        adv_embeddings = AdvancedTextEmbeddings()
        bert_embeddings = adv_embeddings.fit_transform_bert(df['product_name_processed'])

        # Image processing
        image_processor = ImageProcessor()
        image_dir = self.data_path / 'Images'
        image_files = list(image_dir.glob('*.jpg'))[:self.max_images]
        processing_results = image_processor.process_batch([str(p) for p in image_files])

        # Image feature extraction
        vgg16_extractor = VGG16FeatureExtractor()
        deep_features = vgg16_extractor.extract_features(processing_results['processed_images'])

        # Multimodal fusion
        fusion = MultimodalFusion()
        fused_features, min_samples = fusion.prepare_and_fuse(bert_embeddings, deep_features)
        labels = df['product_category'][:min_samples]
        pca_fig, tsne_fig = fusion.analyze_fused_features(fused_features, labels)

        # Save results
        results = {
            'text_features_shape': encoding_results['tfidf_features'].shape,
            'bert_embeddings_shape': bert_embeddings.shape,
            'image_features_shape': deep_features.shape,
            'fused_features_shape': fused_features.shape
        }
        with open(self.output_path / 'mission6_summary.txt', 'w') as f:
            f.write(json.dumps(results, indent=4))

        return {
            'visualizations': {
                'pca': pca_fig,
                'tsne': tsne_fig
            },
            'data': {
                'dataframe': df
            }
        }
