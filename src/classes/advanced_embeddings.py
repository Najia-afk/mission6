from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import tensorflow_hub as hub
import torch
import numpy as np

class AdvancedTextEmbeddings:
    def __init__(self):
        self.word2vec_model = None
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def fit_transform_word2vec(self, processed_text_series, vector_size=100, window=5, min_count=1, workers=4):
        sentences = [text.split() for text in processed_text_series]
        self.word2vec_model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        
        embeddings = []
        for sentence in sentences:
            vectors = [self.word2vec_model.wv[word] for word in sentence if word in self.word2vec_model.wv]
            if vectors:
                embeddings.append(np.mean(vectors, axis=0))
            else:
                embeddings.append(np.zeros(vector_size))
        return np.array(embeddings)

    def fit_transform_bert(self, processed_text_series):
        embeddings = []
        for text in processed_text_series:
            inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        return np.array(embeddings)

    def fit_transform_use(self, processed_text_series):
        return self.use_model(processed_text_series).numpy()
