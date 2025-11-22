"""
Intelligent response generator with context-aware and dynamic responses
"""

import random
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class IntelligentResponseGenerator:
    """Advanced response generator with context awareness and personalization"""
    
    def __init__(self):
        self.response_templates = self._load_response_templates()
        self.context_modifiers = self._load_context_modifiers()
        self.personality_traits = {
            'helpful': 0.9,
            'friendly': 0.8,
            'professional': 0.9,
            'empathetic': 0.7,
            'proactive': 0.8
        }
    
    def _load_response_templates(self) -> Dict[str, Dict]:
        """Load comprehensive response templates"""
        return {
            'greeting': {
                'base_responses': [
                    "Hello! I'm here to help you with any questions or concerns you might have.",
                    "Hi there! Welcome! How can I assist you today?",
                    "Good day! I'm your AI customer assistant. What can I help you with?",
                    "Hello! I'm ready to help you find what you need or solve any issues."
                ],
                'context_responses': {
                    'returning_user': "Welcome back! How can I help you today?",
                    'frustrated_user': "Hello! I understand you might be having some difficulties. I'm here to help resolve any issues.",
                    'happy_user': "Hello! Great to see you in good spirits! How can I assist you today?"
                },
                'follow_up_actions': ['ask_specific_need', 'offer_popular_services']
            },
            
            'goodbye': {
                'base_responses': [
                    "Thank you for contacting us! Have a wonderful day!",
                    "It was great helping you today. Feel free to reach out anytime!",
                    "Goodbye! I hope I was able to help. Have a great day ahead!",
                    "Take care! Don't hesitate to contact us if you need anything else."
                ],
                'context_responses': {
                    'issue_resolved': "I'm glad we could resolve your issue! Have a great day!",
                    'issue_pending': "I've noted your concern and our team will follow up. Thank you for your patience!",
                    'satisfied_user': "Wonderful! I'm so happy I could help. Have an amazing day!"
                }
            },
            
            'order_status': {
                'base_responses': [
                    "I'd be happy to check your order status! Could you please provide your order number?",
                    "Let me look up your order information. What's your order number?",
                    "I can help you track your order. Please share your order number with me."
                ],
                'with_order_info': [
                    "Great! I found your order #{order_id}. Current status: {status}. {additional_info}",
                    "Your order #{order_id} is currently {status}. {tracking_info}",
                    "Here's the update on order #{order_id}: Status is {status}. {estimated_delivery}"
                ],
                'follow_up_actions': ['offer_tracking_updates', 'ask_other_concerns']
            },
            
            'order_cancel': {
                'base_responses': [
                    "I understand you'd like to cancel your order. Let me help you with that.",
                    "I can assist you with order cancellation. What's your order number?",
                    "No problem! I'll help you cancel your order. Could you provide the order details?"
                ],
                'context_responses': {
                    'recent_order': "I see you placed this order recently. I can definitely help cancel it.",
                    'shipped_order': "I notice your order has already shipped. Let me check if we can still cancel or if we need to process a return instead."
                },
                'follow_up_actions': ['confirm_cancellation', 'explain_refund_process']
            },
            
            'return_request': {
                'base_responses': [
                    "I'm here to help with your return request. Our return policy allows returns within 30 days.",
                    "I can guide you through the return process. What item would you like to return?",
                    "No worries! Returns are easy with us. Let me help you get started."
                ],
                'context_responses': {
                    'within_return_period': "Perfect! Your purchase is well within our 30-day return window.",
                    'outside_return_period': "I see this purchase was made over 30 days ago. Let me check what options we have for you.",
                    'defective_product': "I'm sorry to hear the product isn't working properly. We'll definitely take care of this for you."
                },
                'follow_up_actions': ['provide_return_label', 'explain_refund_timeline']
            },
            
            'product_inquiry': {
                'base_responses': [
                    "I'd be happy to provide product information! Which product are you interested in?",
                    "Great question! What specific product would you like to know more about?",
                    "I can help you learn about our products. What are you looking for?"
                ],
                'with_product_info': [
                    "Here's what I can tell you about {product_name}: {description}. Price: ${price}. {availability_status}",
                    "{product_name} is {description}. It's currently priced at ${price} and {availability_status}.",
                    "Great choice! {product_name} - {description}. Price: ${price}. {additional_details}"
                ],
                'follow_up_actions': ['suggest_similar_products', 'offer_purchase_assistance']
            },
            
            'complaint': {
                'base_responses': [
                    "I'm truly sorry to hear about this issue. I want to make sure we resolve this for you.",
                    "I understand your frustration, and I'm here to help fix this problem.",
                    "Thank you for bringing this to our attention. Let me see how I can help resolve this.",
                    "I apologize for any inconvenience. I'm committed to finding a solution for you."
                ],
                'context_responses': {
                    'repeat_issue': "I see this isn't the first time you've contacted us about this. Let me escalate this to ensure it gets resolved.",
                    'urgent_issue': "I understand this is urgent for you. Let me prioritize getting this resolved quickly.",
                    'billing_issue': "I completely understand your concern about billing. Let me connect you with our billing specialists right away."
                },
                'follow_up_actions': ['escalate_to_human', 'offer_compensation', 'schedule_callback']
            },
            
            'compliment': {
                'base_responses': [
                    "Thank you so much for your kind words! It really makes my day to hear that.",
                    "I'm thrilled that I could help! Your feedback means a lot to us.",
                    "That's wonderful to hear! I'm so glad you're satisfied with our service.",
                    "Thank you! I'm happy I could provide the help you needed."
                ],
                'follow_up_actions': ['ask_for_review', 'offer_additional_help']
            },
            
            'technical_support': {
                'base_responses': [
                    "I'm sorry you're experiencing technical difficulties. Let me help troubleshoot this.",
                    "Technical issues can be frustrating. I'll guide you through some solutions.",
                    "I can help resolve this technical problem. Let's start with some basic troubleshooting."
                ],
                'context_responses': {
                    'website_issue': "I understand you're having trouble with our website. Let me provide some quick fixes.",
                    'app_issue': "App problems can be annoying. Let's get this working for you.",
                    'login_issue': "Login troubles are common. I have several solutions we can try."
                },
                'follow_up_actions': ['provide_step_by_step_guide', 'escalate_to_tech_team']
            },
            
            'account_help': {
                'base_responses': [
                    "I can help you with your account. What specific assistance do you need?",
                    "Account management is easy! What would you like to update or change?",
                    "I'm here to help with your account. What can I assist you with?"
                ],
                'context_responses': {
                    'password_reset': "Password resets are simple! I'll guide you through the process.",
                    'profile_update': "Updating your profile is straightforward. Let me help you with that.",
                    'security_concern': "I take security seriously. Let me help secure your account."
                },
                'follow_up_actions': ['send_reset_link', 'verify_identity', 'update_security_settings']
            },
            
            'low_confidence': {
                'base_responses': [
                    "I want to make sure I understand you correctly. Could you please rephrase your question?",
                    "I'm not entirely sure what you're looking for. Can you provide a bit more detail?",
                    "Let me make sure I can help you properly. Could you clarify what you need assistance with?",
                    "I want to give you the best help possible. Can you tell me more about what you're trying to do?"
                ],
                'follow_up_actions': ['offer_common_topics', 'suggest_human_agent']
            }
        }
    
    def _load_context_modifiers(self) -> Dict[str, Dict]:
        """Load context-based response modifiers"""
        return {
            'sentiment_modifiers': {
                'POSITIVE': {
                    'prefix': ["Great!", "Wonderful!", "Fantastic!"],
                    'tone': 'enthusiastic'
                },
                'NEGATIVE': {
                    'prefix': ["I understand your concern.", "I'm sorry to hear that.", "Let me help resolve this."],
                    'tone': 'empathetic'
                },
                'NEUTRAL': {
                    'prefix': ["", "Certainly!", "Of course!"],
                    'tone': 'professional'
                }
            },
            'urgency_modifiers': {
                'high': {
                    'prefix': "I understand this is urgent.",
                    'priority_words': ["immediately", "right away", "as soon as possible"]
                },
                'medium': {
                    'prefix': "I'll help you with this.",
                    'priority_words': ["shortly", "soon", "quickly"]
                },
                'low': {
                    'prefix': "I'm happy to help.",
                    'priority_words': ["when convenient", "at your pace"]
                }
            },
            'user_type_modifiers': {
                'new_user': {
                    'approach': 'explanatory',
                    'additional_info': True
                },
                'returning_user': {
                    'approach': 'direct',
                    'personalization': True
                },
                'vip_user': {
                    'approach': 'premium',
                    'priority_treatment': True
                }
            }
        }
    
    def generate_response(self, message: str, intent: str, confidence: float, 
                         entities: Dict, sentiment: Dict, context: Dict = None, 
                         user_id: str = None) -> Dict[str, Any]:
        """Generate intelligent, context-aware response"""
        
        try:
            # Determine response strategy based on confidence
            if confidence < 0.3:
                return self._generate_low_confidence_response(message, entities, sentiment)
            
            # Get base response template
            template = self.response_templates.get(intent, self.response_templates['low_confidence'])
            
            # Select appropriate response based on context
            response_text = self._select_contextual_response(template, context, sentiment, entities)
            
            # Apply personality and tone modifications
            response_text = self._apply_personality_modifications(response_text, sentiment, intent)
            
            # Add dynamic elements
            response_text = self._add_dynamic_elements(response_text, entities, intent, user_id)
            
            # Generate follow-up actions
            actions = self._generate_follow_up_actions(intent, template, entities, context)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(intent, entities, context)
            
            # Determine response type
            response_type = self._determine_response_type(intent, confidence, entities)
            
            return {
                'response': response_text,
                'response_type': response_type,
                'confidence_used': confidence,
                'actions': actions,
                'suggestions': suggestions,
                'personalization_applied': bool(context),
                'sentiment_considered': sentiment.get('label', 'NEUTRAL') != 'NEUTRAL'
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._generate_fallback_response()
    
    def _select_contextual_response(self, template: Dict, context: Dict, 
                                   sentiment: Dict, entities: Dict) -> str:
        """Select the most appropriate response based on context"""
        
        # Check for specific context responses
        if context and 'context_responses' in template:
            context_responses = template['context_responses']
            
            # Check for specific context keys
            for context_key, response in context_responses.items():
                if self._matches_context(context_key, context, sentiment, entities):
                    return response
        
        # Fall back to base responses
        base_responses = template.get('base_responses', ["I'm here to help!"])
        return random.choice(base_responses)
    
    def _matches_context(self, context_key: str, context: Dict, 
                        sentiment: Dict, entities: Dict) -> bool:
        """Check if context matches a specific key"""
        
        context_matchers = {
            'returning_user': lambda: context.get('conversation_count', 0) > 1,
            'frustrated_user': lambda: sentiment.get('label') == 'NEGATIVE',
            'happy_user': lambda: sentiment.get('label') == 'POSITIVE',
            'recent_order': lambda: 'order' in entities or 'ORDER' in str(entities),
            'urgent_issue': lambda: any(word in str(entities).lower() for word in ['urgent', 'asap', 'immediately']),
            'repeat_issue': lambda: context.get('conversation_count', 0) > 3
        }
        
        matcher = context_matchers.get(context_key)
        return matcher() if matcher else False
    
    def _apply_personality_modifications(self, response: str, sentiment: Dict, intent: str) -> str:
        """Apply personality traits to the response"""
        
        # Add empathy for negative sentiments
        if sentiment.get('label') == 'NEGATIVE' and self.personality_traits['empathetic'] > 0.5:
            empathy_phrases = [
                "I understand how frustrating this must be.",
                "I can imagine this is concerning for you.",
                "I'm sorry you're experiencing this."
            ]
            if not any(phrase in response for phrase in empathy_phrases):
                response = random.choice(empathy_phrases) + " " + response
        
        # Add proactive elements for certain intents
        if intent in ['order_status', 'product_inquiry'] and self.personality_traits['proactive'] > 0.7:
            proactive_additions = {
                'order_status': " I can also set up tracking notifications if you'd like.",
                'product_inquiry': " I can also show you similar products or current promotions."
            }
            response += proactive_additions.get(intent, "")
        
        # Ensure professional tone
        if self.personality_traits['professional'] > 0.8:
            response = response.replace("can't", "cannot").replace("won't", "will not")
        
        return response
    
    def _add_dynamic_elements(self, response: str, entities: Dict, intent: str, user_id: str) -> str:
        """Add dynamic, personalized elements to the response"""
        
        # Add time-based greetings
        current_hour = datetime.now().hour
        if intent == 'greeting':
            if 5 <= current_hour < 12:
                response = response.replace("Hello!", "Good morning!")
            elif 12 <= current_hour < 17:
                response = response.replace("Hello!", "Good afternoon!")
            elif 17 <= current_hour < 22:
                response = response.replace("Hello!", "Good evening!")
        
        # Add entity-specific information
        if entities:
            for entity_text, entity_info in entities.items():
                if entity_info['label'] == 'PERSON':
                    response = response.replace("you", f"you, {entity_text}")
                elif entity_info['label'] == 'ORG':
                    response += f" I see you mentioned {entity_text}."
        
        # Add user-specific elements
        if user_id and user_id != 'default':
            response = response.replace("How can I help you", f"How can I help you today")
        
        return response
    
    def _generate_follow_up_actions(self, intent: str, template: Dict, 
                                   entities: Dict, context: Dict) -> List[str]:
        """Generate contextual follow-up actions"""
        
        actions = template.get('follow_up_actions', [])
        
        # Add intent-specific actions
        intent_actions = {
            'order_status': ['Check other orders', 'Set up tracking alerts'],
            'product_inquiry': ['View similar products', 'Add to wishlist', 'Compare products'],
            'complaint': ['Escalate to manager', 'Schedule callback', 'File formal complaint'],
            'return_request': ['Generate return label', 'Schedule pickup', 'Process refund'],
            'technical_support': ['Contact tech team', 'Schedule remote assistance', 'Access help guides']
        }
        
        specific_actions = intent_actions.get(intent, [])
        return list(set(actions + specific_actions))[:3]  # Limit to 3 actions
    
    def _generate_suggestions(self, intent: str, entities: Dict, context: Dict) -> List[str]:
        """Generate helpful suggestions"""
        
        base_suggestions = {
            'greeting': [
                "Check your order status",
                "Browse our products",
                "View current promotions",
                "Contact customer support"
            ],
            'order_status': [
                "Set up delivery notifications",
                "Change delivery address",
                "View order history"
            ],
            'product_inquiry': [
                "Read customer reviews",
                "Compare with similar products",
                "Check availability in stores"
            ],
            'complaint': [
                "Speak with a manager",
                "Request a callback",
                "Submit written feedback"
            ],
            'low_confidence': [
                "Try rephrasing your question",
                "Browse our help center",
                "Chat with a human agent"
            ]
        }
        
        suggestions = base_suggestions.get(intent, [
            "Browse our help center",
            "Contact customer support",
            "View your account"
        ])
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _determine_response_type(self, intent: str, confidence: float, entities: Dict) -> str:
        """Determine the type of response for UI handling"""
        
        if confidence < 0.3:
            return 'clarification_needed'
        elif intent == 'complaint':
            return 'escalation_candidate'
        elif intent in ['order_status', 'return_request'] and entities:
            return 'action_required'
        elif intent == 'compliment':
            return 'positive_feedback'
        elif intent in ['greeting', 'goodbye']:
            return 'conversational'
        else:
            return 'informational'
    
    def _generate_low_confidence_response(self, message: str, entities: Dict, sentiment: Dict) -> Dict[str, Any]:
        """Generate response for low confidence predictions"""
        
        template = self.response_templates['low_confidence']
        response = random.choice(template['base_responses'])
        
        # Add helpful context based on entities
        if entities:
            response += f" I noticed you mentioned {', '.join(entities.keys())}. "
        
        # Suggest common topics
        common_topics = [
            "order status and tracking",
            "returns and refunds", 
            "product information",
            "account management",
            "technical support"
        ]
        
        response += f"I can help with: {', '.join(common_topics[:3])}."
        
        return {
            'response': response,
            'response_type': 'clarification_needed',
            'confidence_used': 0.0,
            'actions': ['clarify_intent', 'suggest_human_agent'],
            'suggestions': [
                "Try rephrasing your question",
                "Choose from common topics above",
                "Speak with a human agent"
            ],
            'personalization_applied': False,
            'sentiment_considered': False
        }
    
    def _generate_fallback_response(self) -> Dict[str, Any]:
        """Generate fallback response for errors"""
        return {
            'response': "I'm experiencing some technical difficulties right now. Please try again in a moment, or contact our support team for immediate assistance.",
            'response_type': 'error',
            'confidence_used': 0.0,
            'actions': ['retry', 'contact_support'],
            'suggestions': [
                "Try your request again",
                "Contact customer support",
                "Visit our help center"
            ],
            'personalization_applied': False,
            'sentiment_considered': False
        }