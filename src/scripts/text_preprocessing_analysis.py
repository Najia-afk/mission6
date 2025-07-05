"""
Text Preprocessing Analysis Script
Provides comprehensive analysis and visualization of text preprocessing results.
This script handles the detailed analysis logic while keeping notebooks clean.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys
import importlib


class TextPreprocessingAnalyzer:
    """
    Script-level analyzer for text preprocessing operations.
    Handles detailed analysis, statistics, and reporting.
    """
    
    def __init__(self):
        """Initialize the analyzer with fresh modules."""
        self._reload_preprocessor()
        
    def _reload_preprocessor(self):
        """Reload the TextPreprocessor module to ensure latest version."""
        if 'src.classes.preprocess_text' in sys.modules:
            importlib.reload(sys.modules['src.classes.preprocess_text'])
        
        from src.classes.preprocess_text import TextPreprocessor
        self.processor = TextPreprocessor()
    
    def analyze_sample_text(self, text: str) -> Dict:
        """
        Analyze a single text sample and return detailed statistics.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detailed preprocessing statistics
        """
        return self.processor.get_preprocessing_stats(text)
    
    def analyze_batch_texts(self, texts: List[str], n_samples: int = 10) -> pd.DataFrame:
        """
        Analyze a batch of texts and return statistics DataFrame.
        
        Args:
            texts: List of text strings to analyze
            n_samples: Number of samples to analyze
            
        Returns:
            DataFrame with batch statistics
        """
        test_samples = texts[:n_samples]
        batch_results = []
        
        for i, text in enumerate(test_samples):
            stats = self.processor.get_preprocessing_stats(text)
            batch_results.append({
                'Sample': i+1,
                'Original_Words': stats['original_words'],
                'Processed_Words': stats['processed_words'],
                'Stopwords_Removed': stats['removed_stopwords'],
                'Reduction_%': stats['reduction_percentage']
            })
        
        return pd.DataFrame(batch_results)
    
    def find_transformation_examples(self, df: pd.DataFrame, n_examples: int = 5) -> pd.DataFrame:
        """
        Find meaningful text transformation examples.
        
        Args:
            df: DataFrame with original text column
            n_examples: Number of examples to find
            
        Returns:
            DataFrame with transformation examples
        """
        transformation_examples = []
        
        for i in range(min(20, len(df))):
            original = df['product_name'].iloc[i]
            processed = df['product_name_lemmatized'].iloc[i]
            
            if len(original.split()) != len(processed.split()):
                transformation_examples.append({
                    'Original': original[:60] + ('...' if len(original) > 60 else ''),
                    'Processed': processed[:60] + ('...' if len(processed) > 60 else ''),
                    'Word_Change': f"{len(original.split())} → {len(processed.split())}"
                })
                
            if len(transformation_examples) >= n_examples:
                break
        
        if transformation_examples:
            return pd.DataFrame(transformation_examples)
        else:
            # Fallback to showing first 5 regardless
            return pd.DataFrame({
                'Original': df['product_name'].head(5).apply(lambda x: x[:60] + ('...' if len(x) > 60 else '')),
                'Processed': df['product_name_lemmatized'].head(5).apply(lambda x: x[:60] + ('...' if len(x) > 60 else ''))
            })
    
    def calculate_overall_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate overall dataset preprocessing statistics.
        
        Args:
            df: DataFrame with original and processed text columns
            
        Returns:
            Dictionary with overall statistics
        """
        total_words_before = df['product_name'].str.split().str.len().sum()
        total_words_after = df['product_name_lemmatized'].str.split().str.len().sum()
        overall_reduction = ((total_words_before - total_words_after) / total_words_before) * 100
        
        return {
            'total_words_before': total_words_before,
            'total_words_after': total_words_after,
            'overall_reduction': overall_reduction
        }
    
    def process_and_analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'product_name') -> Dict:
        """
        Complete preprocessing and analysis pipeline for a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column to process
            
        Returns:
            Dictionary with all analysis results
        """
        results = {}
        
        # Create lemmatized column
        df[f'{text_column}_lemmatized'] = df[text_column].apply(self.processor.preprocess)
        
        # Sample analysis
        sample_text = df[text_column].iloc[0]
        results['sample_analysis'] = self.analyze_sample_text(sample_text)
        
        # Batch analysis
        results['batch_analysis'] = self.analyze_batch_texts(df[text_column].tolist())
        
        # Transformation examples
        results['transformation_examples'] = self.find_transformation_examples(df)
        
        # Overall statistics
        results['overall_statistics'] = self.calculate_overall_statistics(df)
        
        # Extract top-level categories
        if 'product_category_tree' in df.columns:
            df['product_category'] = df['product_category_tree'].apply(self.processor.extract_top_category)
            results['unique_categories'] = df['product_category'].nunique()
        
        return results
    
    def generate_analysis_report(self, results: Dict, show_examples: bool = True) -> str:
        """
        Generate a formatted analysis report.
        
        Args:
            results: Analysis results from process_and_analyze_dataframe
            show_examples: Whether to include transformation examples
            
        Returns:
            Formatted analysis report string
        """
        report = []
        report.append("=== TEXT PREPROCESSING ANALYSIS ===\n")
        
        # Sample analysis
        sample = results['sample_analysis']
        report.append("📊 Sample Text Analysis:")
        report.append(f"Original: '{sample['original_text']}'")
        report.append(f"Processed: '{sample['processed_text']}'")
        report.append(f"Words: {sample['original_words']} → {sample['processed_words']} ({sample['reduction_percentage']:.1f}% reduction)")
        report.append(f"Stopwords removed: {sample['removed_stopwords']} ({sample['stopwords_percentage']:.1f}%)")
        if sample['sample_removed_words']:
            report.append(f"Sample removed words: {sample['sample_removed_words']}")
        
        # Batch statistics
        batch_df = results['batch_analysis']
        report.append(f"\n📈 Batch Statistics ({len(batch_df)} samples):")
        report.append(f"Average word reduction: {batch_df['Reduction_%'].mean():.1f}%")
        report.append(f"Average stopwords per sample: {batch_df['Stopwords_Removed'].mean():.1f}")
        report.append(f"Total words before: {batch_df['Original_Words'].sum()}")
        report.append(f"Total words after: {batch_df['Processed_Words'].sum()}")
        
        # Transformation examples
        if show_examples:
            examples_df = results['transformation_examples']
            report.append(f"\n📝 Text Transformations:")
            if len(examples_df) > 0 and 'Word_Change' in examples_df.columns:
                report.append(examples_df.to_string(index=False))
            else:
                report.append("Note: Limited transformation examples found (most text may be already clean)")
                report.append(examples_df.to_string(index=False))
        
        # Overall statistics
        overall = results['overall_statistics']
        report.append(f"\n📊 Overall Dataset Statistics:")
        report.append(f"Total words before: {overall['total_words_before']:,}")
        report.append(f"Total words after: {overall['total_words_after']:,}")
        report.append(f"Overall word reduction: {overall['overall_reduction']:.2f}%")
        
        # Categories
        if 'unique_categories' in results:
            report.append(f"\n✅ Text preprocessing completed!")
            report.append(f"📂 Extracted {results['unique_categories']} unique top-level categories")
        
        return "\n".join(report)


def run_complete_text_analysis(df: pd.DataFrame, text_column: str = 'product_name') -> Dict:
    """
    Convenience function to run complete text preprocessing analysis.
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column to process
        
    Returns:
        Dictionary with all analysis results
    """
    analyzer = TextPreprocessingAnalyzer()
    results = analyzer.process_and_analyze_dataframe(df, text_column)
    
    # Print the report
    report = analyzer.generate_analysis_report(results)
    print(report)
    
    return results


if __name__ == "__main__":
    # Example usage when script is run directly
    print("Text Preprocessing Analysis Script")
    print("Import this module and use run_complete_text_analysis(df) function")
