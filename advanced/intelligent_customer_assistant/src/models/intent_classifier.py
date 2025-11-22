from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class IntentClassifier(BaseModel):
    """Intent classification model"""
    
    def __init__(self, model_type: str = 'logistic_regression', **kwargs):
        super().__init__(f"intent_classifier_{model_type}")
        self.model_type = model_type
        self.build_model(**kwargs)
    
    def build_model(self, **kwargs):
        """Build the specified model type"""
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                **kwargs
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                **kwargs
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                random_state=42,
                probability=True,
                **kwargs
            )
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"Built {self.model_type} model")
    
    def train(self, X, y, **kwargs):
        """Train the model"""
        logger.info(f"Training {self.model_type} model with {X.shape[0]} samples")
        self.model.fit(X, y, **kwargs)
        self.is_trained = True
        logger.info("Model training completed")
    
    def predict(self, X, **kwargs):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)