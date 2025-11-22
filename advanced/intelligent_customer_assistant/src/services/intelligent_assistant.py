"""
Intelligent Customer Assistant with advanced AI capabilities
"""

import logging
from typing import Dict, List, Any
import json
import os
from datetime import datetime

from ..models.simplified_intelligent_classifier import SimplifiedIntelligentClassifier
from ..utils.sqlite_database import SQLiteDataService, sqlite_manager
from ..nlp.intelligent_response_generator import IntelligentResponseGenerator

logger = logging.getLogger(__name__)

class IntelligentAssistant:
    """Advanced AI-powered customer assistant"""
    
    def __init__(self):
        self.classifier = SimplifiedIntelligentClassifier()
        self.response_generator = IntelligentResponseGenerator()
        self.conversation_memory = {}
        self.is_trained = False
        
        # Initialize the assistant
        self._initialize()
    
    def _initialize(self):
        """Initialize the intelligent assistant"""
        try:
            # Try to load pre-trained model
            models_dir = os.path.join('data', 'models')
            if os.path.exists(os.path.join(models_dir, 'simplified_intelligent_classifier.json')):
                self.classifier.load_model(models_dir)
                self.is_trained = True
                logger.info("Loaded pre-trained intelligent model")
            else:
                # Train with database data
                self._train_from_database()
                
        except Exception as e:
            logger.error(f"Failed to initialize intelligent assistant: {str(e)}")
            raise
    
    def _train_from_database(self):
        """Train the classifier with database data"""
        try:
            # Get training data from database
            training_data = sqlite_manager.execute_query("SELECT text, intent FROM training_data")
            
            if not training_data:
                logger.warning("No training data found in database")
                return
            
            # Train the classifier
            self.classifier.train_with_examples(training_data)
            self.is_trained = True
            
            # Save the trained model
            models_dir = os.path.join('data', 'models')
            os.makedirs(models_dir, exist_ok=True)
            self.classifier.save_model(models_dir)
            
            logger.info(f"Trained intelligent classifier with {len(training_data)} examples")
            
        except Exception as e:
            logger.error(f"Failed to train from database: {str(e)}")
            raise
    
    def process_message(self, message: str, user_id: str = "default") -> Dict[str, Any]:
        """Process user message with intelligent understanding"""
        
        if not self.is_trained:
            return self._error_response("Assistant not properly trained")
        
        try:
            # Get user context
            context = self.classifier.get_user_context(user_id)
            
            # Predict intent with advanced analysis
            prediction = self.classifier.predict_intent(message, context)
            
            # Generate intelligent response
            response_data = self.response_generator.generate_response(
                message=message,
                intent=prediction['intent'],
                confidence=prediction['confidence'],
                entities=prediction['entities'],
                sentiment=prediction['sentiment'],
                context=context,
                user_id=user_id
            )
            
            # Update conversation memory
            self._update_conversation_memory(user_id, message, prediction, response_data)
            
            # Update classifier context
            self.classifier.update_context(user_id, prediction['intent'], prediction['entities'])
            
            # Log interaction to database
            self._log_interaction(user_id, message, prediction, response_data)
            
            # Prepare final response
            final_response = {
                'message': message,
                'user_id': user_id,
                'intent': prediction['intent'],
                'confidence': prediction['confidence'],
                'response': response_data['response'],
                'response_type': response_data['response_type'],
                'entities': prediction['entities'],
                'sentiment': prediction['sentiment'],
                'explanation': prediction['explanation'],
                'suggestions': response_data.get('suggestions', []),
                'actions': response_data.get('actions', []),
                'context_used': bool(context),
                'timestamp': datetime.now().isoformat()
            }
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return self._error_response(f"Processing error: {str(e)}")
    
    def _update_conversation_memory(self, user_id: str, message: str, prediction: Dict, response_data: Dict):
        """Update conversation memory for the user"""
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []
        
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'intent': prediction['intent'],
            'confidence': prediction['confidence'],
            'entities': prediction['entities'],
            'sentiment': prediction['sentiment'],
            'response': response_data['response'],
            'response_type': response_data['response_type']
        }
        
        self.conversation_memory[user_id].append(conversation_entry)
        
        # Keep only last 20 conversations per user
        if len(self.conversation_memory[user_id]) > 20:
            self.conversation_memory[user_id] = self.conversation_memory[user_id][-20:]
    
    def _log_interaction(self, user_id: str, message: str, prediction: Dict, response_data: Dict):
        """Log interaction to database"""
        try:
            SQLiteDataService.log_interaction(
                customer_id=user_id,
                session_id=f"session_{user_id}_{datetime.now().strftime('%Y%m%d')}",
                message=message,
                intent=prediction['intent'],
                confidence=prediction['confidence'],
                response=response_data['response'],
                response_time=200  # Placeholder
            )
        except Exception as e:
            logger.warning(f"Failed to log interaction: {str(e)}")
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            'message': '',
            'user_id': 'system',
            'intent': 'error',
            'confidence': 0.0,
            'response': "I'm experiencing some technical difficulties. Please try again or contact support.",
            'response_type': 'error',
            'entities': {},
            'sentiment': {'label': 'NEUTRAL', 'score': 0.5},
            'explanation': error_message,
            'suggestions': ['Try rephrasing your question', 'Contact customer support'],
            'actions': [],
            'context_used': False,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for a user"""
        if user_id not in self.conversation_memory:
            return []
        
        return self.conversation_memory[user_id][-limit:]
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user's conversation patterns"""
        if user_id not in self.conversation_memory:
            return {'total_conversations': 0}
        
        conversations = self.conversation_memory[user_id]
        
        # Analyze conversation patterns
        intents = [conv['intent'] for conv in conversations]
        sentiments = [conv['sentiment']['label'] for conv in conversations]
        
        insights = {
            'total_conversations': len(conversations),
            'most_common_intents': self._get_most_common(intents),
            'sentiment_distribution': self._get_sentiment_distribution(sentiments),
            'average_confidence': sum(conv['confidence'] for conv in conversations) / len(conversations),
            'last_interaction': conversations[-1]['timestamp'] if conversations else None,
            'conversation_topics': list(set(intents)),
            'user_satisfaction': self._estimate_satisfaction(conversations)
        }
        
        return insights
    
    def _get_most_common(self, items: List[str], top_n: int = 3) -> List[Dict]:
        """Get most common items with counts"""
        from collections import Counter
        counter = Counter(items)
        return [{'item': item, 'count': count} for item, count in counter.most_common(top_n)]
    
    def _get_sentiment_distribution(self, sentiments: List[str]) -> Dict[str, int]:
        """Get sentiment distribution"""
        from collections import Counter
        return dict(Counter(sentiments))
    
    def _estimate_satisfaction(self, conversations: List[Dict]) -> str:
        """Estimate user satisfaction based on conversation patterns"""
        if not conversations:
            return 'unknown'
        
        # Simple heuristic based on sentiment and intent patterns
        positive_sentiments = sum(1 for conv in conversations if conv['sentiment']['label'] == 'POSITIVE')
        negative_sentiments = sum(1 for conv in conversations if conv['sentiment']['label'] == 'NEGATIVE')
        complaints = sum(1 for conv in conversations if conv['intent'] == 'complaint')
        compliments = sum(1 for conv in conversations if conv['intent'] == 'compliment')
        
        total = len(conversations)
        
        if (positive_sentiments + compliments) / total > 0.6:
            return 'high'
        elif (negative_sentiments + complaints) / total > 0.4:
            return 'low'
        else:
            return 'medium'
    
    def retrain_with_feedback(self, feedback_data: List[Dict]):
        """Retrain the model with user feedback"""
        try:
            # Add feedback to training data
            for feedback in feedback_data:
                sqlite_manager.execute_query(
                    "INSERT INTO training_data (text, intent, confidence) VALUES (?, ?, ?)",
                    (feedback['text'], feedback['correct_intent'], 0.9)
                )
            
            # Retrain the classifier
            self._train_from_database()
            
            logger.info(f"Retrained model with {len(feedback_data)} feedback examples")
            
        except Exception as e:
            logger.error(f"Failed to retrain with feedback: {str(e)}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        try:
            # Get interaction stats from database
            total_interactions = len(sqlite_manager.execute_query(
                "SELECT id FROM customer_interactions"
            ))
            
            # Get intent distribution
            intent_stats = sqlite_manager.execute_query("""
                SELECT intent, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM customer_interactions 
                WHERE intent IS NOT NULL
                GROUP BY intent
                ORDER BY count DESC
            """)
            
            # Calculate overall stats
            all_confidences = [row['avg_confidence'] for row in intent_stats if row['avg_confidence']]
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
            
            high_confidence_interactions = len([
                conf for conf in all_confidences if conf > 0.7
            ])
            
            return {
                'total_interactions': total_interactions,
                'average_confidence': round(avg_confidence, 3),
                'high_confidence_rate': f"{(high_confidence_interactions / len(all_confidences) * 100):.1f}%" if all_confidences else "0%",
                'intent_distribution': intent_stats,
                'active_users': len(self.conversation_memory),
                'model_type': 'Intelligent AI Assistant',
                'capabilities': [
                    'Semantic Understanding',
                    'Context Awareness', 
                    'Sentiment Analysis',
                    'Entity Recognition',
                    'Multi-turn Conversations'
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance stats: {str(e)}")
            return {'error': str(e)}