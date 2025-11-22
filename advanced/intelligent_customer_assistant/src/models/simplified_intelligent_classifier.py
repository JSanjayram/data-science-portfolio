"""
Simplified intelligent classifier using sentence transformers and basic NLP
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Any
import logging
import json
import os
import re
from collections import Counter

logger = logging.getLogger(__name__)

class SimplifiedIntelligentClassifier:
    """Intelligent classifier with semantic understanding"""
    
    def __init__(self):
        self.sentence_model = None
        self.intent_embeddings = {}
        self.context_history = []
        self.entity_patterns = self._load_entity_patterns()
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize AI models"""
        try:
            logger.info("Loading sentence transformer model...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            self.sentence_model = None
    
    def _load_entity_patterns(self) -> Dict:
        """Load entity recognition patterns"""
        return {
            'order_number': [
                r'order\s*#?\s*([A-Z]{2,3}\d{3,6})',
                r'order\s*number\s*([A-Z]{2,3}\d{3,6})',
                r'#([A-Z]{2,3}\d{3,6})'
            ],
            'email': [
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            ],
            'phone': [
                r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
                r'\d{10}'
            ],
            'money': [
                r'\$\d+(?:\.\d{2})?',
                r'\d+\s*dollars?'
            ],
            'product_id': [
                r'[A-Z]{3,5}\d{3,6}'
            ]
        }
    
    def train_with_examples(self, training_data: List[Dict]):
        """Train the classifier with example data"""
        logger.info(f"Training with {len(training_data)} examples")
        
        # Group examples by intent
        intent_examples = {}
        for example in training_data:
            intent = example['intent']
            text = example['text']
            
            if intent not in intent_examples:
                intent_examples[intent] = []
            intent_examples[intent].append(text)
        
        # Create embeddings for each intent
        for intent, examples in intent_examples.items():
            if self.sentence_model:
                # Use semantic embeddings
                embeddings = self.sentence_model.encode(examples)
                # Average embeddings for the intent
                self.intent_embeddings[intent] = {
                    'embedding': np.mean(embeddings, axis=0),
                    'examples': examples,
                    'keywords': self._extract_keywords(examples)
                }
            else:
                # Fallback to keyword patterns
                self.intent_embeddings[intent] = {
                    'examples': examples,
                    'keywords': self._extract_keywords(examples)
                }
        
        logger.info(f"Trained classifier for {len(intent_examples)} intents")
    
    def _extract_keywords(self, examples: List[str]) -> List[str]:
        """Extract important keywords from examples"""
        # Simple keyword extraction
        all_words = []
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        
        for example in examples:
            words = re.findall(r'\\b\\w+\\b', example.lower())
            all_words.extend([word for word in words if word not in stop_words and len(word) > 2])
        
        # Get most common words
        word_counts = Counter(all_words)
        return [word for word, count in word_counts.most_common(10)]
    
    def predict_intent(self, text: str, context: Dict = None) -> Dict:
        """Predict intent with confidence and reasoning"""
        
        # Analyze the input text
        analysis = self._analyze_text(text)
        
        # Get semantic similarity scores
        similarity_scores = self._calculate_semantic_similarity(text)
        
        # Get keyword matching scores
        keyword_scores = self._calculate_keyword_scores(text)
        
        # Combine scores
        final_scores = self._combine_scores(similarity_scores, keyword_scores, context)
        
        # Get best prediction
        if final_scores:
            best_intent = max(final_scores.items(), key=lambda x: x[1])
        else:
            best_intent = ('unknown', 0.0)
        
        # Generate explanation
        explanation = self._generate_explanation(text, analysis, best_intent[0], final_scores)
        
        return {
            'intent': best_intent[0],
            'confidence': float(best_intent[1]),
            'all_scores': {k: float(v) for k, v in final_scores.items()},
            'analysis': analysis,
            'explanation': explanation,
            'entities': analysis.get('entities', {}),
            'sentiment': analysis.get('sentiment', {'label': 'NEUTRAL', 'score': 0.5})
        }
    
    def _analyze_text(self, text: str) -> Dict:
        """Basic text analysis"""
        analysis = {
            'original_text': text,
            'cleaned_text': text.lower().strip(),
            'entities': self._extract_entities(text),
            'sentiment': self._analyze_sentiment(text),
            'linguistic_features': self._extract_linguistic_features(text)
        }
        
        return analysis
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract entities using regex patterns"""
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    if isinstance(matches[0], tuple):
                        # Handle grouped matches
                        entities[entity_type] = [''.join(match) for match in matches]
                    else:
                        entities[entity_type] = matches
        
        return entities
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Basic sentiment analysis using keywords"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'satisfied', 'perfect', 'awesome', 'brilliant', 'outstanding', 'superb']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'angry', 'frustrated', 'disappointed', 'annoyed', 'upset', 'furious', 'disgusted', 'outraged', 'livid']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return {'label': 'POSITIVE', 'score': min(0.9, 0.5 + (positive_count * 0.1))}
        elif negative_count > positive_count:
            return {'label': 'NEGATIVE', 'score': min(0.9, 0.5 + (negative_count * 0.1))}
        else:
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    def _extract_linguistic_features(self, text: str) -> Dict:
        """Extract basic linguistic features"""
        return {
            'word_count': len(text.split()),
            'char_count': len(text),
            'has_question': '?' in text,
            'has_exclamation': '!' in text,
            'has_negation': any(neg in text.lower() for neg in ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor', "don't", "won't", "can't", "shouldn't", "wouldn't", "couldn't"]),
            'is_uppercase': text.isupper(),
            'question_words': [word for word in ['what', 'when', 'where', 'why', 'how', 'who', 'which'] if word in text.lower()],
            'action_words': [word for word in ['cancel', 'return', 'refund', 'exchange', 'buy', 'purchase', 'order', 'track', 'check', 'update', 'change'] if word in text.lower()]
        }
    
    def _calculate_semantic_similarity(self, text: str) -> Dict[str, float]:
        """Calculate semantic similarity with intent examples"""
        scores = {}
        
        if not self.sentence_model:
            return scores
        
        try:
            text_embedding = self.sentence_model.encode([text])
            
            for intent, data in self.intent_embeddings.items():
                if 'embedding' in data:
                    similarity = cosine_similarity(text_embedding, [data['embedding']])[0][0]
                    scores[intent] = max(0, similarity)
                else:
                    scores[intent] = 0.0
        
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {str(e)}")
        
        return scores
    
    def _calculate_keyword_scores(self, text: str) -> Dict[str, float]:
        """Calculate keyword-based scores"""
        scores = {}
        text_lower = text.lower()
        text_words = set(re.findall(r'\\b\\w+\\b', text_lower))
        
        for intent, data in self.intent_embeddings.items():
            keywords = data.get('keywords', [])
            if keywords:
                keyword_matches = len(set(keywords) & text_words)
                scores[intent] = keyword_matches / len(keywords)
            else:
                scores[intent] = 0.0
        
        return scores
    
    def _combine_scores(self, semantic_scores: Dict, keyword_scores: Dict, context: Dict = None) -> Dict[str, float]:
        """Combine different scoring methods"""
        combined_scores = {}
        
        all_intents = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        for intent in all_intents:
            semantic_score = semantic_scores.get(intent, 0.0)
            keyword_score = keyword_scores.get(intent, 0.0)
            
            # Weighted combination
            if self.sentence_model:
                combined_score = (semantic_score * 0.7) + (keyword_score * 0.3)
            else:
                combined_score = keyword_score
            
            # Context boost
            if context and context.get('last_intent') == intent:
                combined_score *= 1.1
            
            combined_scores[intent] = combined_score
        
        # Normalize scores
        max_score = max(combined_scores.values()) if combined_scores else 1.0
        if max_score > 0:
            combined_scores = {k: v/max_score for k, v in combined_scores.items()}
        
        return combined_scores
    
    def _generate_explanation(self, text: str, analysis: Dict, predicted_intent: str, scores: Dict) -> str:
        """Generate explanation of the prediction"""
        explanations = []
        
        confidence = scores.get(predicted_intent, 0.0)
        explanations.append(f"Intent: {predicted_intent} ({confidence:.2f})")
        
        entities = analysis.get('entities', {})
        if entities:
            explanations.append(f"Entities: {list(entities.keys())}")
        
        sentiment = analysis.get('sentiment', {})
        if sentiment.get('label') != 'NEUTRAL':
            explanations.append(f"Sentiment: {sentiment['label']}")
        
        features = analysis.get('linguistic_features', {})
        if features.get('has_question'):
            explanations.append("Question detected")
        if features.get('has_negation'):
            explanations.append("Negation found")
        if features.get('action_words'):
            explanations.append(f"Actions: {features['action_words']}")
        
        return " | ".join(explanations)
    
    def update_context(self, user_id: str, intent: str, entities: Dict):
        """Update conversation context"""
        context_entry = {
            'user_id': user_id,
            'intent': intent,
            'entities': entities,
            'timestamp': len(self.context_history)
        }
        
        self.context_history.append(context_entry)
        
        # Keep only recent context
        if len(self.context_history) > 100:
            self.context_history = self.context_history[-100:]
    
    def get_user_context(self, user_id: str) -> Dict:
        """Get recent context for a user"""
        user_history = [c for c in self.context_history if c['user_id'] == user_id]
        
        if not user_history:
            return {}
        
        recent = user_history[-1]
        return {
            'last_intent': recent['intent'],
            'last_entities': recent['entities'],
            'conversation_count': len(user_history)
        }
    
    def save_model(self, path: str):
        """Save model state"""
        model_data = {
            'intent_embeddings': {k: {
                'embedding': v['embedding'].tolist() if 'embedding' in v else None,
                'examples': v['examples'],
                'keywords': v['keywords']
            } for k, v in self.intent_embeddings.items()},
            'context_history': self.context_history
        }
        
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'simplified_intelligent_classifier.json'), 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, path: str):
        """Load model state"""
        model_path = os.path.join(path, 'simplified_intelligent_classifier.json')
        if os.path.exists(model_path):
            with open(model_path, 'r') as f:
                model_data = json.load(f)
            
            for intent, data in model_data['intent_embeddings'].items():
                self.intent_embeddings[intent] = {
                    'examples': data['examples'],
                    'keywords': data['keywords']
                }
                if data['embedding']:
                    self.intent_embeddings[intent]['embedding'] = np.array(data['embedding'])
            
            self.context_history = model_data.get('context_history', [])
            logger.info(f"Loaded simplified intelligent classifier with {len(self.intent_embeddings)} intents")