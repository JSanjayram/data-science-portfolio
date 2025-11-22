from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import joblib

class TextVectorizer:
    def __init__(self, vectorizer_type='tfidf'):
        self.vectorizer_type = vectorizer_type
        self.vectorizer = None
    
    def fit_transform(self, texts):
        if self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        else:
            self.vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        return self.vectorizer.transform(texts)
    
    def save(self, path):
        joblib.dump(self.vectorizer, path)
    
    def load(self, path):
        self.vectorizer = joblib.load(path)