# src/services/assistant_service.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import os
import joblib
from ..utils.config import config
from ..models.intent_classifier import IntentClassifier
from ..features.feature_engineering import FeatureEngineer
from ..nlp.text_preprocessor import TextPreprocessor
from ..nlp.entity_extractor import EntityExtractor
from ..data.data_loader import DataLoader
from ..data.data_preprocessor import DataPreprocessor
from ..utils.sqlite_database import SQLiteDataService, sqlite_manager

logger = logging.getLogger(__name__)

class AssistantService:
    """Main customer assistant service"""
    
    def __init__(self):
        self.config = config
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.text_preprocessor = TextPreprocessor()
        self.entity_extractor = EntityExtractor()
        self.data_loader = DataLoader()
        self.data_preprocessor = DataPreprocessor()
        self.conversation_history = []
        self.user_contexts = {}
        self.feedback_data = []
        self.is_trained = False
        
        self.response_templates = {
            'password_reset': "🔐 You can reset your password by visiting our password recovery page at [website]/reset-password. Check your email for the reset link. If you don't receive it within 5 minutes, please check your spam folder.",
            'order_status': "📦 I can check your order status! Please provide your order number and I'll look up the current delivery estimate. You can also track your order in real-time in your account dashboard.",
            'billing_issue': "💳 I understand you're having billing issues. I'm connecting you with our billing specialists. You can also email billing@company.com with your account details for faster resolution.",
            'product_info': "📱 I'd be happy to provide product information! Could you specify which product you're interested in? You can also browse our complete catalog at [website]/products with detailed specifications.",
            'technical_support': "🛠️ I'm sorry you're experiencing technical issues. Let me guide you through some troubleshooting steps: 1) Clear your browser cache 2) Restart the application 3) Check for updates. If issues persist, our tech team can help.",
            'return_refund': "📮 Our return policy allows returns within 30 days of purchase. Please visit [website]/returns to initiate the process and print your return label. Make sure the item is in original condition.",
            'account_management': "👤 I can help with account management! You can update most settings in your account dashboard under 'Profile Settings'. What specific changes would you like to make?",
            'low_confidence': "❓ I'm not quite sure I understand. Could you please rephrase your question or provide more details? I'm here to help with password resets, order status, billing, products, technical issues, returns, and account management."
        }
    
    def initialize_model(self, model_type: str = 'logistic_regression'):
        """Initialize the intent classification model"""
        self.model = IntentClassifier(model_type=model_type)
        
        # Try to load pre-trained model
        models_dir = os.path.join('data', 'models')
        if os.path.exists(models_dir):
            try:
                self.feature_engineer.load_artifacts(models_dir)
                self.model.load_model(models_dir)
                self.is_trained = True
                logger.info("Loaded pre-trained model from disk")
                return
            except Exception as e:
                logger.warning(f"Could not load pre-trained model: {str(e)}")
        
        # If no pre-trained model, train a new one
        logger.info("No pre-trained model found. Training new model...")
        self._train_new_model()
    
    def _train_new_model(self):
        """Train a new model with database data"""
        try:
            # Get training data from database
            training_data = sqlite_manager.get_dataframe("SELECT text, intent FROM training_data")
            
            if training_data.empty:
                # Fallback to synthetic data
                df = self.data_loader.create_synthetic_dataset()
                df_processed = self.data_preprocessor.preprocess_dataframe(df, 'customer_query')
                processed_texts = df_processed['processed_customer_query']
                labels = df_processed['intent']
            else:
                # Use database training data
                processed_texts = training_data['text'].apply(self.text_preprocessor.clean_text)
                labels = training_data['intent']
            
            # Create features
            X = self.feature_engineer.create_tfidf_features(processed_texts, fit=True)
            y = self.feature_engineer.encode_labels(labels, fit=True)
            
            # Train model
            self.model.train(X, y)
            self.is_trained = True
            
            # Save model artifacts
            models_dir = os.path.join('data', 'models')
            os.makedirs(models_dir, exist_ok=True)
            self.feature_engineer.save_artifacts(models_dir)
            self.model.save_model(models_dir)
            
            logger.info(f"Successfully trained model with {len(processed_texts)} examples")
            
        except Exception as e:
            logger.error(f"Failed to train new model: {str(e)}")
            raise
    
    def train_model(self, df: pd.DataFrame, text_column: str = 'customer_query', 
                   label_column: str = 'intent'):
        """Train the intent classification model with custom data"""
        logger.info("Starting model training pipeline")
        
        # Preprocess data
        processed_texts = df[text_column].apply(self.text_preprocessor.clean_text)
        
        # Create features
        X = self.feature_engineer.create_tfidf_features(processed_texts, fit=True)
        y = self.feature_engineer.encode_labels(df[label_column], fit=True)
        
        # Train model
        self.model.train(X, y)
        self.is_trained = True
        
        logger.info("Model training completed successfully")
    
    def predict_intent(self, query: str) -> Dict:
        """Predict intent for a single query"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Preprocess query
        processed_query = self.text_preprocessor.clean_text(query)
        
        # Create features
        X = self.feature_engineer.create_tfidf_features(
            pd.Series([processed_query]), 
            fit=False
        )
        
        # Predict
        probabilities = self.model.predict_proba(X)[0]
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_class_idx]
        
        # Get intent name
        predicted_intent = self.feature_engineer.label_encoder.inverse_transform(
            [predicted_class_idx]
        )[0]
        
        return {
            'intent': predicted_intent,
            'confidence': float(confidence),
            'probabilities': probabilities.tolist()
        }
    
    def generate_response(self, query: str, user_id: str = "default") -> Dict:
        """Generate response for user query"""
        try:
            # Predict intent
            intent_result = self.predict_intent(query)
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(query)
            
            # Get user context
            context = self._get_user_context(user_id)
            
            # Generate contextual response
            response = self._generate_contextual_response(
                intent_result['intent'], 
                intent_result['confidence'],
                entities,
                context
            )
            
            # Store conversation
            self._store_conversation(user_id, query, intent_result, response, entities)
            
            # Log interaction to database
            try:
                SQLiteDataService.log_interaction(
                    user_id, f"session_{user_id}", query, 
                    intent_result['intent'], intent_result['confidence'], 
                    response, 200
                )
            except Exception as e:
                logger.warning(f"Failed to log interaction: {str(e)}")
            
            return {
                'query': query,
                'predicted_intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'response': response,
                'entities': entities,
                'user_id': user_id,
                'status': 'resolved' if intent_result['confidence'] >= self.config.get_model_config()['confidence_threshold'] else 'needs_clarification'
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                'query': query,
                'predicted_intent': 'error',
                'confidence': 0.0,
                'response': "❌ I'm experiencing technical difficulties. Please try again later.",
                'entities': {},
                'user_id': user_id,
                'status': 'error'
            }
    
    def _get_user_context(self, user_id: str) -> Dict:
        """Get or create user context"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {
                'last_intent': None,
                'last_entities': {},
                'conversation_count': 0
            }
        return self.user_contexts[user_id]
    
    def _generate_contextual_response(self, intent: str, confidence: float, 
                                    entities: Dict, context: Dict) -> str:
        """Generate contextual response"""
        if confidence < self.config.get_model_config()['confidence_threshold']:
            return self.response_templates['low_confidence']
        
        base_response = self.response_templates.get(intent, self.response_templates['low_confidence'])
        
        # Add contextual enhancements
        enhancements = []
        
        if intent == 'order_status' and 'order_number' in entities:
            enhancements.append(f"📦 I found order #{entities['order_number']}!")
        
        if 'urgency' in entities and entities['urgency'] == 'high':
            enhancements.append("🚨 I understand this is urgent!")
        
        if context['last_intent'] == intent and context['conversation_count'] > 1:
            enhancements.append("Is there anything specific about this that you'd like me to clarify?")
        
        if enhancements:
            return " ".join(enhancements) + " " + base_response
        
        return base_response
    
    def _store_conversation(self, user_id: str, query: str, intent_result: Dict, 
                           response: str, entities: Dict):
        """Store conversation in history"""
        conversation_entry = {
            'user_id': user_id,
            'timestamp': pd.Timestamp.now().isoformat(),
            'query': query,
            'predicted_intent': intent_result['intent'],
            'confidence': intent_result['confidence'],
            'response': response,
            'entities': entities
        }
        
        self.conversation_history.append(conversation_entry)
        
        # Update user context
        context = self._get_user_context(user_id)
        context['last_intent'] = intent_result['intent']
        context['last_entities'] = entities
        context['conversation_count'] += 1
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.conversation_history:
            return {
                "total_queries_processed": 0,
                "resolved_queries": 0,
                "resolution_rate": "0.0%",
                "average_confidence": "0.000",
                "active_users": len(self.user_contexts)
            }
        
        total_queries = len(self.conversation_history)
        resolved_queries = len([c for c in self.conversation_history 
                              if c['confidence'] >= self.config.get_model_config()['confidence_threshold']])
        
        resolution_rate = (resolved_queries / total_queries) * 100 if total_queries > 0 else 0
        avg_confidence = np.mean([c['confidence'] for c in self.conversation_history])
        
        return {
            'total_queries_processed': total_queries,
            'resolved_queries': resolved_queries,
            'resolution_rate': f"{resolution_rate:.1f}%",
            'average_confidence': f"{avg_confidence:.3f}",
            'active_users': len(self.user_contexts)
        }