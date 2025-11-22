from abc import ABC, abstractmethod
import joblib
import os
from sklearn.metrics import classification_report, accuracy_score
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def build_model(self, **kwargs):
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def train(self, X, y, **kwargs):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X, **kwargs):
        """Make predictions"""
        pass
    
    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score']
        }
        
        logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}")
        return metrics
    
    def save_model(self, save_path: str):
        """Save model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, f'{self.model_name}.pkl')
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, load_path: str):
        """Load model from disk"""
        model_path = os.path.join(load_path, f'{self.model_name}.pkl')
        self.model = joblib.load(model_path)
        self.is_trained = True
        logger.info(f"Model loaded from {model_path}")