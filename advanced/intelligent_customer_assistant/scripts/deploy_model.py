#!/usr/bin/env python3
"""
Model deployment script
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.model_registry import ModelRegistry
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def deploy_model():
    """Deploy trained model to production"""
    try:
        logger.info("Starting model deployment")
        
        registry = ModelRegistry()
        
        # Load and register model
        model = registry.load_model('intent_classifier_logistic_regression')
        registry.register_model('production_model', model)
        
        logger.info("Model deployed successfully")
        
    except Exception as e:
        logger.error(f"Model deployment failed: {str(e)}")
        raise

if __name__ == '__main__':
    deploy_model()