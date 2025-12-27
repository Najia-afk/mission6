import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import normalize
from typing import List, Dict, Any, Union, Optional
from urllib.error import URLError
import ssl
import shutil
import tempfile
import os

class AdvancedTextEmbeddings:
    """Class for generating advanced text embeddings using Word2Vec, BERT, and Universal Sentence Encoder"""
    
    def __init__(self):
        """Initialize the advanced embeddings class"""
        self.word2vec_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.use_model = None
        self.embeddings = None
        self.method = None
        
    def fit_transform_word2vec(self, texts, vector_size: int = 100, 
                            window: int = 5, min_count: int = 1, workers: int = 4) -> np.ndarray:
        """
        Train Word2Vec model and transform texts to embeddings
        """
        # Convert Series to list if needed
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        
        # Ensure all texts are strings and filter out None values
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text is not None and not pd.isna(text):
                valid_texts.append(str(text))
                valid_indices.append(i)
        
        # Tokenize texts into sentences of words
        tokenized_texts = [text.split() for text in valid_texts]
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers
        )
        
        # Generate document embeddings by averaging word vectors
        valid_embeddings = []
        for tokens in tokenized_texts:
            # Get vectors for each word in the document
            token_vecs = [self.word2vec_model.wv[word] for word in tokens 
                        if word in self.word2vec_model.wv]
            
            if token_vecs:
                # Average the vectors
                doc_vec = np.mean(token_vecs, axis=0)
            else:
                # Use zeros if no words are in vocabulary
                doc_vec = np.zeros(vector_size)
                
            valid_embeddings.append(doc_vec)
        
        # Create array of embeddings for all original indices
        full_embeddings = np.zeros((len(texts), vector_size))
        for idx, orig_idx in enumerate(valid_indices):
            full_embeddings[orig_idx] = valid_embeddings[idx]
        
        self.embeddings = full_embeddings
        self.method = "word2vec"
        
        return self.embeddings
    
    def fit_transform_bert(self, texts, model_name: str = "bert-base-uncased", 
                        max_length: int = 128) -> np.ndarray:
        """
        Generate embeddings using BERT
        
        Args:
            texts: List or Series of preprocessed texts
            model_name: Name of the pre-trained BERT model
            max_length: Maximum sequence length
            
        Returns:
            Document embeddings matrix
        """
        # Convert Series to list if needed
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        
        # Ensure all texts are strings and filter out None values
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text is not None and not pd.isna(text):
                valid_texts.append(str(text))
                valid_indices.append(i)
        
        # Load pre-trained model and tokenizer
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        
        # Set to evaluation mode
        self.bert_model.eval()
        
        embeddings = []
        batch_size = 32
        
        # Process in batches to avoid memory issues
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i+batch_size]
            
            # Tokenize texts
            encoded_input = self.bert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Generate embeddings (without gradient calculation)
            with torch.no_grad():
                model_output = self.bert_model(**encoded_input)
                
            # Use the [CLS] token as the sentence embedding
            sentence_embeddings = model_output[0][:, 0, :].numpy()
            embeddings.extend(sentence_embeddings)
        
        # Create array of embeddings for all original indices
        if embeddings:
            embedding_dim = embeddings[0].shape[0]
            full_embeddings = np.zeros((len(texts), embedding_dim))
            for idx, orig_idx in enumerate(valid_indices):
                full_embeddings[orig_idx] = embeddings[idx]
        else:
            # Handle edge case with no valid embeddings
            full_embeddings = np.array([])
        
        self.embeddings = full_embeddings
        self.method = "bert"
        
        return self.embeddings
    
    def fit_transform_use(self, texts) -> np.ndarray:
        """
        Generate embeddings using Universal Sentence Encoder
        """
        import certifi
        
        # Convert Series to list if needed
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        
        # Ensure all texts are strings and filter out None values
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text is not None and not pd.isna(text):
                valid_texts.append(str(text))
                valid_indices.append(i)
        
        # Configure SSL for macOS
        # Option 1: Use certifi's certificates
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        os.environ['SSL_CERT_FILE'] = certifi.where()
        
        # Use cached model location
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                  'cache', 'use_model')
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['TFHUB_CACHE_DIR'] = cache_dir
        
        model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        
        print(f"   📦 Using cached model directory: {cache_dir}")
        print("   ⏳ Loading Universal Sentence Encoder (this is a one-time download)...")
        
        try:
            self.use_model = hub.load(model_url)
            print("   ✅ Model loaded successfully!")
        except ValueError as e:
            # Handle corrupted/incomplete TF-Hub cache
            if ("incompatible/unknown type" in str(e)
                or "contains neither 'saved_model.pb'" in str(e)):
                print("   ⚠️ Cached model appears corrupted. Resetting cache and retrying...")
                try:
                    shutil.rmtree(cache_dir)
                except Exception:
                    pass
                os.makedirs(cache_dir, exist_ok=True)
                os.environ['TFHUB_CACHE_DIR'] = cache_dir
                self.use_model = hub.load(model_url)
                print("   ✅ Model downloaded and cached successfully!")
            else:
                raise
        except URLError as e:
            # Handle SSL issues (e.g., CERTIFICATE_VERIFY_FAILED)
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                print("   ⚠️ SSL verification failed. Retrying with unverified SSL context...")
                ssl._create_default_https_context = ssl._create_unverified_context
                self.use_model = hub.load(model_url)
                print("   ✅ Model downloaded and cached successfully!")
            else:
                raise
        
        # Generate embeddings
        valid_embeddings = self.use_model(valid_texts).numpy()
        
        # Create array of embeddings for all original indices
        if len(valid_embeddings) > 0:
            embedding_dim = valid_embeddings.shape[1]
            full_embeddings = np.zeros((len(texts), embedding_dim))
            for idx, orig_idx in enumerate(valid_indices):
                full_embeddings[orig_idx] = valid_embeddings[idx]
        else:
            # Handle edge case with no valid embeddings
            full_embeddings = np.array([])
        
        self.embeddings = full_embeddings
        self.method = "use"
        
        return self.embeddings
    
    def get_most_similar_words(self, word: str, n: int = 10) -> List[tuple]:
        """
        Get most similar words for a given word (Word2Vec only)
        
        Args:
            word: Input word
            n: Number of similar words to return
            
        Returns:
            List of (word, similarity) tuples
        """
        if self.method != "word2vec" or self.word2vec_model is None:
            raise ValueError("This method is only available for Word2Vec embeddings")
            
        if word not in self.word2vec_model.wv:
            return []
            
        return self.word2vec_model.wv.most_similar(word, topn=n)
    
    def plot_word_similarity(self, words: List[str], figsize=(12, 10)) -> None:
        """
        Plot similarity between words using Word2Vec
        
        Args:
            words: List of words to plot
            figsize: Figure size
        """
        if self.method != "word2vec" or self.word2vec_model is None:
            raise ValueError("This method is only available for Word2Vec embeddings")
        
        # Filter words that exist in the vocabulary
        valid_words = [w for w in words if w in self.word2vec_model.wv]
        
        if len(valid_words) < 2:
            print("Not enough valid words to plot similarities")
            return
            
        # Calculate similarity matrix
        n = len(valid_words)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                similarity_matrix[i, j] = self.word2vec_model.wv.similarity(valid_words[i], valid_words[j])
                
        # Plot heatmap
        plt.figure(figsize=figsize)
        plt.imshow(similarity_matrix, cmap="viridis")
        plt.colorbar(label="Cosine Similarity")
        plt.xticks(range(n), valid_words, rotation=45, ha="right")
        plt.yticks(range(n), valid_words)
        plt.title("Word Similarity Matrix")
        plt.tight_layout()
        plt.show()
    
    def compare_with_reducer(self, reducer, labels=None, figsize=(18, 6)):
        """
        Compare embeddings using PCA and t-SNE
        
        Args:
            reducer: DimensionalityReducer instance
            labels: Category labels for coloring points
            figsize: Figure size
            
        Returns:
            Dictionary with visualization figures
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Run fit_transform_* first.")
            
        # Apply PCA
        pca_results = reducer.fit_transform_pca(self.embeddings)
        # Remove the title parameter if not supported
        pca_fig = reducer.plot_pca(labels=labels)
        
        # Apply t-SNE
        tsne_results = reducer.fit_transform_tsne(self.embeddings)
        tsne_fig = reducer.plot_tsne(labels=labels)
        
        # Create silhouette plot if labels are provided
        if labels is not None:
            silhouette_fig = reducer.plot_silhouette(
                self.embeddings, 
                labels
            )
            
            # Create intercluster distance visualization
            distance_fig = reducer.plot_intercluster_distance(
                self.embeddings,
                labels
            )
            
            # Evaluate clustering
            clustering_results = reducer.evaluate_clustering(
                self.embeddings,
                labels,
                n_clusters=len(set(labels)),
                use_tsne=True
            )
            
            # Create heatmap - check if this method accepts title
            try:
                heatmap_fig = reducer.plot_cluster_category_heatmap(
                    clustering_results['cluster_distribution'],
                    title=f"Cluster Composition - {self.method.upper()} Embeddings"
                )
            except TypeError:
                # If title not accepted, call without it
                heatmap_fig = reducer.plot_cluster_category_heatmap(
                    clustering_results['cluster_distribution']
                )
            
            return {
                'pca_fig': pca_fig,
                'tsne_fig': tsne_fig,
                'silhouette_fig': silhouette_fig,
                'distance_fig': distance_fig,
                'heatmap_fig': heatmap_fig,
                'clustering_results': clustering_results
            }
        
        return {
            'pca_fig': pca_fig,
            'tsne_fig': tsne_fig
        }