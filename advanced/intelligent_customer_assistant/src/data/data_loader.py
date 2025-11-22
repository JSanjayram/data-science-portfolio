import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from src.utils.config import config

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loading and generation class"""
    
    def __init__(self):
        self.config = config.get_data_config()
        self.intents_config = config.get('intents')
    
    def create_synthetic_dataset(self) -> pd.DataFrame:
        """Create realistic customer service dataset"""
        logger.info("Creating synthetic customer service dataset")
        
        intents_data = self._get_intents_data()
        queries, intents, responses = [], [], []
        
        for intent_name, data in intents_data.items():
            num_samples = int(data['frequency'] * self.config['dataset_size'])
            
            for _ in range(num_samples):
                base_query = np.random.choice(data['queries'])
                query = self._add_variation(base_query)
                queries.append(query)
                intents.append(intent_name)
                responses.append(data['response_template'])
        
        df = pd.DataFrame({
            'customer_query': queries,
            'intent': intents,
            'response': responses
        })
        
        return df.sample(frac=1, random_state=self.config['random_state']).reset_index(drop=True)
    
    def _get_intents_data(self) -> Dict:
        """Get intents configuration"""
        return {
            'password_reset': {
                'queries': [
                    "How do I reset my password?",
                    "I forgot my password and can't login",
                    "Password reset not working",
                    "Can you help me change my password?",
                    "I'm locked out of my account"
                ],
                'frequency': 0.15,
                'response_template': "You can reset your password by visiting our password recovery page..."
            },
            'order_status': {
                'queries': [
                    "Where is my order?",
                    "When will my package arrive?",
                    "Order delivery status",
                    "Tracking my shipment",
                    "Why is my order delayed?"
                ],
                'frequency': 0.20,
                'response_template': "I can check your order status. Please provide your order number..."
            },
            'billing_issue': {
                'queries': [
                    "I was charged twice",
                    "Invoice incorrect amount",
                    "Billing problem need help",
                    "Why was I overcharged?",
                    "Payment dispute"
                ],
                'frequency': 0.15,
                'response_template': "I understand you're having billing issues..."
            },
            'product_info': {
                'queries': [
                    "Do you have this product in stock?",
                    "What are the features of product X?",
                    "Product specifications",
                    "Price comparison between models"
                ],
                'frequency': 0.15,
                'response_template': "I'd be happy to provide product information..."
            },
            'technical_support': {
                'queries': [
                    "App keeps crashing",
                    "Website not loading properly",
                    "Error message when logging in",
                    "Feature not working"
                ],
                'frequency': 0.15,
                'response_template': "I'm sorry you're experiencing technical issues..."
            },
            'return_refund': {
                'queries': [
                    "How to return a product?",
                    "Return policy information",
                    "I want to refund my purchase"
                ],
                'frequency': 0.10,
                'response_template': "Our return policy allows returns within 30 days..."
            },
            'account_management': {
                'queries': [
                    "How to update my email address?",
                    "Change account settings",
                    "Delete my account"
                ],
                'frequency': 0.10,
                'response_template': "I can help with account management..."
            }
        }
    
    def _add_variation(self, query: str) -> str:
        """Add realistic variations to queries"""
        variations = [
            lambda x: x,
            lambda x: x + " please help",
            lambda x: "Hello, " + x.lower(),
            lambda x: "I need help with " + x.lower(),
        ]
        variation_func = np.random.choice(variations)
        return variation_func(query)