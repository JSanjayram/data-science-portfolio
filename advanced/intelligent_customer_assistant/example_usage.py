import sys
import os

from src.services.assistant_service import AssistantService
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    # Initialize services
    assistant = AssistantService()
    data_loader = DataLoader()
    data_preprocessor = DataPreprocessor()
    
    # Create and preprocess data
    df = data_loader.create_synthetic_dataset()
    df_processed = data_preprocessor.preprocess_dataframe(df, 'customer_query')
    
    # Train model
    assistant.initialize_model('logistic_regression')
    assistant.train_model(df_processed)
    
    # Test the assistant
    test_queries = [
        "I forgot my password",
        "Where is my order?",
        "I was charged twice"
    ]
    
    for query in test_queries:
        result = assistant.generate_response(query)
        print(f"Query: {query}")
        print(f"Intent: {result['predicted_intent']} (Confidence: {result['confidence']:.3f})")
        print(f"Response: {result['response']}")
        print("-" * 50)
    
    # Show statistics
    stats = assistant.get_performance_stats()
    print("\nPerformance Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == '__main__':
    main()