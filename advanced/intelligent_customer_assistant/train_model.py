#!/usr/bin/env python3
"""
Train the customer assistant model with comprehensive intent data
"""

import os
import sys
import pandas as pd
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.assistant_service import AssistantService
from src.utils.sqlite_database import sqlite_manager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def prepare_training_data():
    """Prepare comprehensive training data from database"""
    
    # Get intents and examples from database
    intents_query = "SELECT intent_name, examples FROM intents"
    intents_data = sqlite_manager.execute_query(intents_query)
    
    training_examples = []
    
    for intent_row in intents_data:
        intent_name = intent_row['intent_name']
        examples = intent_row['examples'].split(',')
        
        for example in examples:
            example = example.strip()
            if example:
                training_examples.append({
                    'text': example,
                    'intent': intent_name
                })
    
    # Add existing training data from database
    existing_training = sqlite_manager.execute_query("SELECT text, intent FROM training_data")
    training_examples.extend(existing_training)
    
    # Create DataFrame
    df = pd.DataFrame(training_examples)
    
    logger.info(f"Prepared {len(df)} training examples across {df['intent'].nunique()} intents")
    return df

def train_model():
    """Train the assistant model"""
    logger.info("Starting model training...")
    
    # Initialize assistant service
    assistant = AssistantService()
    assistant.initialize_model()  # Initialize the model first
    
    # Prepare training data
    training_df = prepare_training_data()
    
    # Train model
    assistant.train_model(training_df, text_column='text', label_column='intent')
    
    # Test model with sample queries
    test_queries = [
        "hello there",
        "where is my order",
        "I want to return this item",
        "how much does this cost",
        "cancel my order",
        "forgot my password",
        "website not working",
        "any discounts available",
        "goodbye"
    ]
    
    logger.info("Testing trained model:")
    for query in test_queries:
        result = assistant.predict_intent(query)
        logger.info(f"Query: '{query}' -> Intent: {result['intent']} (Confidence: {result['confidence']:.3f})")
    
    logger.info("Model training completed successfully!")
    return assistant

if __name__ == "__main__":
    try:
        trained_assistant = train_model()
        print("\nModel training completed successfully!")
        print("You can now run the application with: python run.py")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"\nTraining failed: {str(e)}")
        sys.exit(1)