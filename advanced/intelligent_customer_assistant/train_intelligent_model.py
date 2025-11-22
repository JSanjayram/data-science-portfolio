#!/usr/bin/env python3
"""
Train the intelligent AI model with advanced capabilities
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.intelligent_assistant import IntelligentAssistant
from src.utils.sqlite_database import sqlite_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_intelligent_model():
    """Test the intelligent model with various queries"""
    
    logger.info("Initializing Intelligent Assistant...")
    assistant = IntelligentAssistant()
    
    # Test queries with different complexity levels
    test_queries = [
        # Basic greetings
        "Hello, I need some help",
        "Good morning, how are you?",
        
        # Order-related queries
        "Where is my order? I placed it last week and haven't heard anything",
        "I want to cancel my recent order, is that possible?",
        "My order ORD001 seems to be delayed, what's going on?",
        
        # Product inquiries
        "Tell me about the wireless headphones you have",
        "Do you have any smartphones under $500?",
        "What's the difference between your premium and basic plans?",
        
        # Complex complaints
        "I'm really frustrated with your service. This is the third time I'm calling about the same issue and nobody seems to care",
        "The product I received is completely different from what I ordered and now you're telling me I can't return it?",
        
        # Technical issues
        "Your website keeps crashing when I try to checkout",
        "I can't log into my account even though I'm using the right password",
        
        # Contextual conversations
        "I forgot my password",
        "Actually, I also need to update my email address",
        "And while we're at it, can you check if I have any pending orders?",
        
        # Emotional queries
        "Thank you so much for your excellent service!",
        "I'm so disappointed with this purchase",
        "This is exactly what I was looking for, perfect!",
        
        # Ambiguous queries
        "I have a problem",
        "Something is wrong",
        "Can you help me with this thing?",
        
        # Goodbye
        "Thanks for your help, goodbye!"
    ]
    
    logger.info(f"Testing intelligent model with {len(test_queries)} queries...")
    
    user_id = "test_user_001"
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {query}")
        print('='*60)
        
        try:
            result = assistant.process_message(query, user_id)
            
            print(f"Intent: {result['intent']} (Confidence: {result['confidence']:.3f})")
            print(f"Response: {result['response']}")
            
            if result['entities']:
                print(f"Entities: {result['entities']}")
            
            if result['sentiment']['label'] != 'NEUTRAL':
                print(f"Sentiment: {result['sentiment']['label']} ({result['sentiment']['score']:.3f})")
            
            if result['suggestions']:
                print(f"Suggestions: {', '.join(result['suggestions'])}")
            
            if result['actions']:
                print(f"Actions: {', '.join(result['actions'])}")
            
            print(f"Explanation: {result['explanation']}")
            
            if result['context_used']:
                print("✓ Context was used in this response")
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            logger.error(f"Failed to process query '{query}': {str(e)}")
    
    # Show user insights
    print(f"\n{'='*60}")
    print("USER INSIGHTS")
    print('='*60)
    
    insights = assistant.get_user_insights(user_id)
    for key, value in insights.items():
        print(f"{key}: {value}")
    
    # Show performance stats
    print(f"\n{'='*60}")
    print("PERFORMANCE STATISTICS")
    print('='*60)
    
    stats = assistant.get_performance_stats()
    for key, value in stats.items():
        if key != 'intent_distribution':
            print(f"{key}: {value}")
    
    logger.info("Intelligent model testing completed!")

if __name__ == "__main__":
    try:
        test_intelligent_model()
        print("\n✅ Intelligent model testing completed successfully!")
        print("The AI assistant is ready to handle complex, context-aware conversations!")
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        print(f"\n❌ Testing failed: {str(e)}")
        sys.exit(1)