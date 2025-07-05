import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import re

class TextPreprocessor:
    def __init__(self):
        self._download_nltk_resources()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def _download_nltk_resources(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except nltk.downloader.DownloadError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except nltk.downloader.DownloadError:
            nltk.download('wordnet')

    def tokenize(self, text):
        return word_tokenize(text)

    def stem(self, text):
        tokens = self.tokenize(text)
        return " ".join([self.stemmer.stem(word) for word in tokens])

    def lemmatize(self, text):
        tokens = self.tokenize(text)
        return " ".join([self.lemmatizer.lemmatize(word) for word in tokens])

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = self.tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)
    
    def extract_top_category(self, category_tree):
        return category_tree.split('>>')[0].strip()
