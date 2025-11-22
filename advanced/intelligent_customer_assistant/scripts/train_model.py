#!/usr/bin/env python3
"""
Model training script for Intelligent Customer Assistant
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from services.assistant_service import AssistantService
from utils.logger import setup_logger

logger = setup_logger(__name__)

def train_model():
    """Train and save the customer assistant model"""
    try:
        logger.info("Starting model training pipeline")
        
        # Load and preprocess data
        data_loader = DataLoader()
        data_preprocessor = DataPreprocessor()
        
        df = data_loader.create_synthetic_dataset()
        df_processed = data_preprocessor.preprocess_dataframe(df, 'customer_query')
        
        logger.info(f"Created dataset with {len(df)} samples")
        
        # Initialize and train assistant service
        assistant = AssistantService()
        assistant.initialize_model('logistic_regression')
        assistant.train_model(df_processed)
        
        # Save model artifacts
        models_dir = os.path.join('data', 'models')
        assistant.feature_engineer.save_artifacts(models_dir)
        assistant.model.save_model(models_dir)
        
        logger.info("Model training completed successfully")
        logger.info(f"Model artifacts saved to {models_dir}")
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

if __name__ == '__main__':
    train_model()