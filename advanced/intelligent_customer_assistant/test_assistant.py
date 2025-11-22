# test_assistant.py
#!/usr/bin/env python3
"""
Quick test script for the customer assistant
Tests the assistant service directly without the web interface
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.assistant_service import AssistantService

def test_assistant():
    """Test the assistant service directly"""
    print("🤖 Testing Intelligent Customer Assistant...")
    print("=" * 60)
    
    try:
        # Initialize assistant
        print("Initializing assistant service...")
        assistant = AssistantService()
        assistant.initialize_model()
        print("✅ Assistant initialized successfully!")
        
        # Test queries
        test_queries = [
            "I forgot my password and can't login",
            "Where is my order? It hasn't arrived yet",
            "I was charged twice for my subscription",
            "Do you have the new smartphone in stock?",
            "The app keeps crashing when I try to checkout",
            "How can I return a product I bought last week?",
            "I need to update my email address in my account",
            "What's the weather like today?"  # Should have low confidence
        ]
        
        print(f"\n📝 Testing {len(test_queries)} sample queries...")
        print("=" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            result = assistant.generate_response(query)
            print(f"   ✅ Intent: {result['predicted_intent']}")
            print(f"   📊 Confidence: {result['confidence']:.3f}")
            print(f"   🎯 Status: {result['status']}")
            print(f"   💬 Response: {result['response']}")
            print("-" * 50)
        
        # Show statistics
        stats = assistant.get_performance_stats()
        print("\n" + "📈" * 20)
        print("PERFORMANCE STATISTICS")
        print("📈" * 20)
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_assistant()