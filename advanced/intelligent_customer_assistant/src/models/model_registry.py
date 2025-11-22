import joblib
import os
from typing import Dict, Any

class ModelRegistry:
    def __init__(self, registry_path: str = 'data/models'):
        self.registry_path = registry_path
        self.models = {}
    
    def register_model(self, name: str, model: Any, metadata: Dict = None):
        self.models[name] = {
            'model': model,
            'metadata': metadata or {}
        }
    
    def get_model(self, name: str):
        return self.models.get(name, {}).get('model')
    
    def save_model(self, name: str, model: Any):
        os.makedirs(self.registry_path, exist_ok=True)
        model_path = os.path.join(self.registry_path, f'{name}.pkl')
        joblib.dump(model, model_path)
    
    def load_model(self, name: str):
        model_path = os.path.join(self.registry_path, f'{name}.pkl')
        return joblib.load(model_path)
    
    def list_models(self):
        return list(self.models.keys())