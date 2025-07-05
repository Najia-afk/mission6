from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import plotly.graph_objects as go
from wordcloud import WordCloud
import pandas as pd

class TextEncoder:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.bow_vectorizer = CountVectorizer(max_features=self.max_features)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=self.max_features)
        self.bow_features = None
        self.tfidf_features = None
        self.feature_names = None

    def fit_transform(self, text_series):
        self.bow_features = self.bow_vectorizer.fit_transform(text_series)
        self.tfidf_features = self.tfidf_vectorizer.fit_transform(text_series)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        return {
            'bow_features': self.bow_features,
            'tfidf_features': self.tfidf_features
        }

    def plot_word_cloud(self, use_tfidf=False, **kwargs):
        if use_tfidf:
            term_frequencys = self.tfidf_features.sum(axis=0).A1
        else:
            term_frequencys = self.bow_features.sum(axis=0).A1
        
        word_freq = dict(zip(self.feature_names, term_frequencys))
        
        wc = WordCloud(width=800, height=400, background_color='white', **kwargs)
        wc.generate_from_frequencies(word_freq)
        
        fig = go.Figure(go.Image(z=wc.to_array()))
        fig.update_layout(title=f'Word Cloud ({'TF-IDF' if use_tfidf else 'BoW'})')
        return fig
