from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from typing import Tuple
import pandas as pd
import logging
from ..utils.config import config

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering pipeline"""
    
    def __init__(self):
        self.model_config = config.get_model_config()
        self.vectorizer = None
        self.label_encoder = None
    
    def create_tfidf_features(self, texts: pd.Series, fit: bool = True) -> Tuple:
        """Create TF-IDF features"""
        if fit or self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=self.model_config['max_features'],
                stop_words='english',
                ngram_range=tuple(self.model_config['ngram_range']),
                min_df=self.model_config['min_df'],
                max_df=self.model_config['max_df']
            )
            features = self.vectorizer.fit_transform(texts)
            logger.info(f"Fitted TF-IDF vectorizer with {features.shape[1]} features")
        else:
            features = self.vectorizer.transform(texts)
        
        return features
    
    def encode_labels(self, labels: pd.Series, fit: bool = True) -> pd.Series:
        """Encode string labels to integers"""
        if fit or self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
            logger.info(f"Fitted label encoder with {len(self.label_encoder.classes_)} classes")
        else:
            encoded_labels = self.label_encoder.transform(labels)
        
        return encoded_labels
    
    def save_artifacts(self, save_path: str):
        """Save vectorizer and label encoder"""
        os.makedirs(save_path, exist_ok=True)
        
        joblib.dump(self.vectorizer, os.path.join(save_path, 'tfidf_vectorizer.pkl'))
        joblib.dump(self.label_encoder, os.path.join(save_path, 'label_encoder.pkl'))
        logger.info(f"Saved feature engineering artifacts to {save_path}")
    
    def load_artifacts(self, load_path: str):
        """Load vectorizer and label encoder"""
        self.vectorizer = joblib.load(os.path.join(load_path, 'tfidf_vectorizer.pkl'))
        self.label_encoder = joblib.load(os.path.join(load_path, 'label_encoder.pkl'))
        logger.info(f"Loaded feature engineering artifacts from {load_path}")