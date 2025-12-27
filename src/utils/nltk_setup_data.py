import ssl
import nltk
import os

def setup_nltk_data():
    """Setup NLTK data with SSL workaround"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Required NLTK packages
    packages = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'punkt_tab'
    ]
    
    # Set download directory
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Download packages
    for package in packages:
        try:
            nltk.download(package, download_dir=nltk_data_dir, quiet=True)
        except Exception as e:
            print(f"Error downloading {package}: {e}")