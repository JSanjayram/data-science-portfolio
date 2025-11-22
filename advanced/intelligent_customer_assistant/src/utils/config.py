import os
import yaml
from typing import Dict, Any

class Config:
    """Configuration management class"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                '../../config/development.yaml'
            )
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def get(self, key: str, default=None):
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default
    
    def get_model_config(self):
        """Get model-specific configuration"""
        return self.get('model')
    
    def get_data_config(self):
        """Get data-specific configuration"""
        return self.get('data')

# Singleton configuration instance
config = Config()