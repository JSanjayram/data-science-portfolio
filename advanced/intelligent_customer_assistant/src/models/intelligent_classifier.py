"""
Advanced intelligent intent classifier using transformers and semantic understanding
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Any
import logging
import json
import os

logger = logging.getLogger(__name__)

class IntelligentClassifier:
    """Advanced AI-powered intent classifier with context understanding"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sentence_model = None
        self.nlp = None
        self.intent_embeddings = {}
        self.context_history = []
        self.entity_patterns = {}
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all AI models"""
        try:
            # Load sentence transformer for semantic similarity
            logger.info("Loading sentence transformer model...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load spaCy for NER and linguistic analysis
            logger.info("Loading spaCy model...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Installing...")
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("All AI models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            # Fallback to basic models
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Initialize fallback models if advanced ones fail"""
        logger.info("Initializing fallback models...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("Using basic spaCy model")
            self.nlp = spacy.blank("en")
    
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
                    'patterns': self._extract_patterns(examples)
                }
            else:
                # Fallback to keyword patterns
                self.intent_embeddings[intent] = {
                    'examples': examples,
                    'patterns': self._extract_patterns(examples)
                }
        
        logger.info(f"Trained classifier for {len(intent_examples)} intents")
    
    def _extract_patterns(self, examples: List[str]) -> Dict:
        """Extract linguistic patterns from examples"""
        patterns = {
            'keywords': set(),
            'entities': set(),
            'pos_patterns': [],
            'dependency_patterns': []
        }
        
        for example in examples:
            if self.nlp:
                doc = self.nlp(example.lower())
                
                # Extract keywords (excluding stop words)
                patterns['keywords'].update([token.lemma_ for token in doc 
                                           if not token.is_stop and not token.is_punct])
                
                # Extract named entities
                patterns['entities'].update([ent.label_ for ent in doc.ents])
                
                # Extract POS patterns
                pos_pattern = [token.pos_ for token in doc]
                patterns['pos_patterns'].append(pos_pattern)
        
        # Convert sets to lists for JSON serialization
        patterns['keywords'] = list(patterns['keywords'])
        patterns['entities'] = list(patterns['entities'])
        
        return patterns
    
    def predict_intent(self, text: str, context: Dict = None) -> Dict:
        """Predict intent with confidence and reasoning"""
        
        # Analyze the input text
        analysis = self._analyze_text(text)
        
        # Get semantic similarity scores
        similarity_scores = self._calculate_semantic_similarity(text)
        
        # Get pattern matching scores
        pattern_scores = self._calculate_pattern_scores(text, analysis)
        
        # Combine scores with context
        final_scores = self._combine_scores(similarity_scores, pattern_scores, context)
        
        # Get best prediction
        best_intent = max(final_scores.items(), key=lambda x: x[1])
        
        # Generate explanation
        explanation = self._generate_explanation(text, analysis, best_intent[0], final_scores)
        
        return {
            'intent': best_intent[0],
            'confidence': float(best_intent[1]),
            'all_scores': {k: float(v) for k, v in final_scores.items()},
            'analysis': analysis,
            'explanation': explanation,
            'entities': analysis.get('entities', {}),
            'sentiment': analysis.get('sentiment', {})
        }
    
    def _analyze_text(self, text: str) -> Dict:
        """Comprehensive text analysis"""
        analysis = {
            'original_text': text,
            'cleaned_text': text.lower().strip(),
            'entities': {},
            'sentiment': {},
            'linguistic_features': {}
        }
        
        if self.nlp:
            doc = self.nlp(text)
            
            # Extract entities
            analysis['entities'] = {
                ent.text: {
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_) or ent.label_
                } for ent in doc.ents
            }
            
            # Extract linguistic features
            analysis['linguistic_features'] = {
                'tokens': len(doc),
                'sentences': len(list(doc.sents)),
                'has_question': '?' in text,
                'has_negation': any(token.dep_ == 'neg' for token in doc),
                'main_verbs': [token.lemma_ for token in doc if token.pos_ == 'VERB'],
                'main_nouns': [token.lemma_ for token in doc if token.pos_ == 'NOUN']
            }
        
        # Sentiment analysis
        if hasattr(self, 'sentiment_analyzer'):
            try:
                sentiment_result = self.sentiment_analyzer(text)[0]
                analysis['sentiment'] = {
                    'label': sentiment_result['label'],
                    'score': sentiment_result['score']
                }
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {str(e)}")
                analysis['sentiment'] = {'label': 'NEUTRAL', 'score': 0.5}
        
        return analysis
    
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
                    scores[intent] = max(0, similarity)  # Ensure non-negative
                else:
                    scores[intent] = 0.0
        
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {str(e)}")
        
        return scores
    
    def _calculate_pattern_scores(self, text: str, analysis: Dict) -> Dict[str, float]:
        """Calculate pattern-based scores"""
        scores = {}
        text_lower = text.lower()
        
        for intent, data in self.intent_embeddings.items():
            score = 0.0
            patterns = data.get('patterns', {})
            
            # Keyword matching
            keywords = patterns.get('keywords', [])
            if keywords:
                keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
                score += (keyword_matches / len(keywords)) * 0.4
            
            # Entity matching
            entities = patterns.get('entities', [])
            user_entities = set(analysis.get('entities', {}).values())
            if entities and user_entities:
                entity_matches = len(set(entities) & user_entities)
                score += (entity_matches / len(entities)) * 0.3
            
            # Linguistic feature matching
            linguistic = analysis.get('linguistic_features', {})
            
            # Question detection
            if intent in ['product_inquiry', 'price_inquiry', 'shipping_info'] and linguistic.get('has_question'):
                score += 0.2
            
            # Negation detection for complaints
            if intent == 'complaint' and linguistic.get('has_negation'):
                score += 0.2
            
            # Verb matching for actions
            main_verbs = linguistic.get('main_verbs', [])
            if intent == 'order_cancel' and any(verb in ['cancel', 'stop', 'halt'] for verb in main_verbs):
                score += 0.3
            
            if intent == 'return_request' and any(verb in ['return', 'refund', 'exchange'] for verb in main_verbs):
                score += 0.3
            
            scores[intent] = min(1.0, score)  # Cap at 1.0
        
        return scores
    
    def _combine_scores(self, semantic_scores: Dict, pattern_scores: Dict, context: Dict = None) -> Dict[str, float]:
        """Combine different scoring methods"""
        combined_scores = {}
        
        all_intents = set(semantic_scores.keys()) | set(pattern_scores.keys())
        
        for intent in all_intents:
            semantic_score = semantic_scores.get(intent, 0.0)
            pattern_score = pattern_scores.get(intent, 0.0)
            
            # Weighted combination
            combined_score = (semantic_score * 0.6) + (pattern_score * 0.4)
            
            # Context boost
            if context and context.get('last_intent') == intent:
                combined_score *= 1.1  # Small boost for context continuity
            
            combined_scores[intent] = combined_score
        
        # Normalize scores
        max_score = max(combined_scores.values()) if combined_scores else 1.0
        if max_score > 0:
            combined_scores = {k: v/max_score for k, v in combined_scores.items()}
        
        return combined_scores
    
    def _generate_explanation(self, text: str, analysis: Dict, predicted_intent: str, scores: Dict) -> str:
        """Generate human-readable explanation of the prediction"""
        
        explanations = []
        
        # Main prediction
        confidence = scores[predicted_intent]
        explanations.append(f"Predicted intent: {predicted_intent} (confidence: {confidence:.2f})")
        
        # Key factors
        entities = analysis.get('entities', {})
        if entities:
            explanations.append(f"Detected entities: {', '.join(entities.keys())}")
        
        sentiment = analysis.get('sentiment', {})
        if sentiment.get('label') != 'NEUTRAL':
            explanations.append(f"Sentiment: {sentiment['label']} ({sentiment['score']:.2f})")
        
        linguistic = analysis.get('linguistic_features', {})
        if linguistic.get('has_question'):
            explanations.append("Question detected")
        
        if linguistic.get('has_negation'):
            explanations.append("Negation detected")
        
        # Alternative intents
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[1][1] > 0.3:
            explanations.append(f"Alternative: {sorted_scores[1][0]} ({sorted_scores[1][1]:.2f})")
        
        return " | ".join(explanations)
    
    def update_context(self, user_id: str, intent: str, entities: Dict):
        """Update conversation context"""
        context_entry = {
            'user_id': user_id,
            'intent': intent,
            'entities': entities,
            'timestamp': torch.tensor(0).item()  # Simple timestamp
        }
        
        self.context_history.append(context_entry)
        
        # Keep only recent context (last 10 interactions per user)
        user_history = [c for c in self.context_history if c['user_id'] == user_id]
        if len(user_history) > 10:
            # Remove oldest entries for this user
            self.context_history = [c for c in self.context_history 
                                  if c['user_id'] != user_id or c in user_history[-10:]]
    
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
                'patterns': v['patterns']
            } for k, v in self.intent_embeddings.items()},
            'context_history': self.context_history
        }
        
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'intelligent_classifier.json'), 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, path: str):
        """Load model state"""
        model_path = os.path.join(path, 'intelligent_classifier.json')
        if os.path.exists(model_path):
            with open(model_path, 'r') as f:
                model_data = json.load(f)
            
            # Restore embeddings
            for intent, data in model_data['intent_embeddings'].items():
                self.intent_embeddings[intent] = {
                    'examples': data['examples'],
                    'patterns': data['patterns']
                }
                if data['embedding']:
                    self.intent_embeddings[intent]['embedding'] = np.array(data['embedding'])
            
            self.context_history = model_data.get('context_history', [])
            logger.info(f"Loaded intelligent classifier with {len(self.intent_embeddings)} intents")